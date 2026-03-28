from __future__ import annotations
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse
from typing import Any
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    host = parsed.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/").split("/")[0]
        if video_id:
            return video_id

    query = parse_qs(parsed.query)
    if "v" in query and query["v"]:
        return query["v"][0]

    # Support urls like /shorts/<id> and /embed/<id>.
    path_parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in path_parts:
            idx = path_parts.index(marker)
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]

    raise RuntimeError("Could not extract YouTube video id from URL")


def _get_entries(video_id: str, preferred_languages: list[str]) -> tuple[list[dict[str, Any]], str]:
    api = YouTubeTranscriptApi()
    if hasattr(api, "list"):
        transcripts = api.list(video_id)
    elif hasattr(api, "list_transcripts"):
        transcripts = api.list_transcripts(video_id)
    else:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)

    available_languages = [item.language_code for item in transcripts]

    if preferred_languages:
        try:
            transcript = transcripts.find_transcript(preferred_languages)
        except Exception:
            transcript = transcripts.find_transcript(available_languages)
    else:
        transcript = transcripts.find_transcript(available_languages)

    fetched = transcript.fetch()
    entries = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
    return entries, str(transcript.language_code)


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
        # return cls(youtube_url=youtube_url, add_video_info=add_video_info, language=language or ["en"])
        return cls(youtube_url=youtube_url, add_video_info=add_video_info, language=language)

    def _extract_info(self) -> dict[str, Any]:
        video_id = _extract_video_id(self.youtube_url)
        return {
            "id": video_id,
            "title": "",
            "channel": None,
            "uploader": None,
        }

    def load(self) -> list[Document]:
        info = self._extract_info()
        preferred = self.language or []
        entries, lang = _get_entries(str(info.get("id", "")), preferred)

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
    # Prefer any available track first for multilingual processing.
    loader = YoutubeLoaderDL.from_youtube_url(
        youtube_url=youtube_url,
        add_video_info=True,
        # language=["en", "en-US", "en-GB"],
        language=None,
    )
    docs = loader.load()
    first_meta = docs[0].metadata if docs else {}
    language_code = str(first_meta.get("language", "unknown"))
    language_label = language_code
    video_id = str(first_meta.get("video_id", "unknown"))
    return docs, language_code, language_label, video_id
