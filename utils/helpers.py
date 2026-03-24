from __future__ import annotations
import hashlib
from urllib.parse import parse_qs, urlparse


def extract_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from common URL formats."""
    if not url or not url.strip():
        raise ValueError("Empty YouTube URL")

    raw = url.strip()
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    def _clean(value: str | None) -> str | None:
        if not value:
            return None
        return value.split("?")[0].split("&")[0].split("#")[0].strip() or None

    if "youtu.be" in host:
        return _clean(path.split("/")[0]) or _raise_bad_url(url)

    if "youtube.com" in host or "m.youtube.com" in host:
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] == "watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
            return _clean(candidate) or _raise_bad_url(url)
        if len(parts) >= 2 and parts[0] in {"embed", "shorts", "live"}:
            return _clean(parts[1]) or _raise_bad_url(url)

    raise ValueError(f"Unsupported YouTube URL: {url}")


def build_chunk_id(video_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{video_id}::chunk-{chunk_index:06d}::{digest}"


def _raise_bad_url(url: str) -> str:
    raise ValueError(f"Could not extract video id from URL: {url}")
