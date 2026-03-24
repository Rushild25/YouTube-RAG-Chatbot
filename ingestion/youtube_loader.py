from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any

import requests
from langchain_core.documents import Document
from yt_dlp import YoutubeDL


def _pick_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tracks:
        return None
    # Prefer json3 because it preserves timing and text segments cleanly.
    for track in tracks:
        if track.get("ext") == "json3":
            return track
    # Fallback to the first available subtitle/caption track.
    return tracks[0]


def _parse_json3(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    events = payload.get("events", [])
    lines: list[dict[str, Any]] = []

    for event in events:
        segments = event.get("segs") or []
        content = "".join(seg.get("utf8", "") for seg in segments).strip()
        if not content:
            continue

        start = float(event.get("tStartMs", 0.0)) / 1000.0
        duration = float(event.get("dDurationMs", 0.0)) / 1000.0
        lines.append({"text": content, "start": start, "duration": duration})

    return lines


def _parse_vtt(text: str) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    time_pattern = re.compile(
        r"(?P<s_h>\d{2}):(?P<s_m>\d{2}):(?P<s_s>\d{2})\.(?P<s_ms>\d{3})\s+-->\s+"
        r"(?P<e_h>\d{2}):(?P<e_m>\d{2}):(?P<e_s>\d{2})\.(?P<e_ms>\d{3})"
    )

    for block in blocks:
        parts = block.splitlines()
        if len(parts) < 2:
            continue

        match = time_pattern.search(parts[0]) or (time_pattern.search(parts[1]) if len(parts) > 1 else None)
        if not match:
            continue

        text_lines = parts[2:] if time_pattern.search(parts[0]) else parts[3:]
        content = " ".join(item.strip() for item in text_lines if item.strip()).strip()
        if not content:
            continue

        start = (
            int(match.group("s_h")) * 3600
            + int(match.group("s_m")) * 60
            + int(match.group("s_s"))
            + int(match.group("s_ms")) / 1000.0
        )
        end = (
            int(match.group("e_h")) * 3600
            + int(match.group("e_m")) * 60
            + int(match.group("e_s"))
            + int(match.group("e_ms")) / 1000.0
        )
        lines.append({"text": content, "start": start, "duration": max(0.01, end - start)})

    return lines


@dataclass
class YoutubeLoaderDL:
    youtube_url: str
    add_video_info: bool = True
    language: list[str] | None = None

    @classmethod
    def from_youtube_url(
        cls,
        youtube_url: str,
        add_video_info: bool = True,
        language: list[str] | None = None,
    ) -> "YoutubeLoaderDL":
        return cls(youtube_url=youtube_url, add_video_info=add_video_info, language=language or ["en"])

    def _extract_info(self) -> dict[str, Any]:
        with YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            return ydl.extract_info(self.youtube_url, download=False)

    def _select_caption_track(self, info: dict[str, Any]) -> tuple[dict[str, Any], str]:
        subtitles = info.get("subtitles") or {}
        auto_captions = info.get("automatic_captions") or {}
        caption_groups = [subtitles, auto_captions]

        preferred = self.language or ["en"]

        for captions in caption_groups:
            for lang in preferred:
                if lang in captions:
                    track = _pick_track(captions.get(lang, []))
                    if track:
                        return track, lang

        for captions in caption_groups:
            for lang, tracks in captions.items():
                track = _pick_track(tracks)
                if track:
                    return track, lang

        raise RuntimeError("No subtitle/caption tracks found via yt_dlp")

    def load(self) -> list[Document]:
        info = self._extract_info()
        track, lang = self._select_caption_track(info)

        track_url = track.get("url")
        if not track_url:
            raise RuntimeError("Selected subtitle track has no URL")

        response = requests.get(track_url, timeout=30)
        response.raise_for_status()
        raw_text = response.text

        ext = str(track.get("ext", "")).lower()
        if ext == "json3":
            entries = _parse_json3(raw_text)
        else:
            entries = _parse_vtt(raw_text)

        video_id = str(info.get("id", "unknown"))
        title = str(info.get("title", ""))

        documents: list[Document] = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            start = float(entry.get("start", 0.0))
            duration = float(entry.get("duration", 0.0))
            metadata = {
                "video_id": video_id,
                "source": self.youtube_url,
                "language": lang,
                "start_seconds": start,
                "duration_seconds": duration,
            }
            if self.add_video_info:
                metadata.update(
                    {
                        "title": title,
                        "channel": info.get("channel"),
                        "uploader": info.get("uploader"),
                    }
                )

            documents.append(Document(page_content=text, metadata=metadata))

        if not documents:
            raise RuntimeError("No transcript text could be parsed from subtitle track")

        return documents


def fetch_transcript(youtube_url: str) -> tuple[list[Document], str, str, str]:
    """Fetch transcript documents and return docs + language + video_id."""
    loader = YoutubeLoaderDL.from_youtube_url(
        youtube_url=youtube_url,
        add_video_info=True,
        language=["en", "en-US", "en-GB"],
    )
    docs = loader.load()
    first_meta = docs[0].metadata if docs else {}
    language_code = str(first_meta.get("language", "unknown"))
    language_label = language_code
    video_id = str(first_meta.get("video_id", "unknown"))
    return docs, language_code, language_label, video_id
