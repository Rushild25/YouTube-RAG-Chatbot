from __future__ import annotations
import html
import re
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class TranscriptLine:
    text: str
    start: float
    duration: float


_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_BRACKETED_CUE_RE = re.compile(r"\[(.{1,40}?)\]")


def _clean_transcript_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _BRACKETED_CUE_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def normalize_transcript_lines(raw_items: list[dict] | list[Document]) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    previous_text = ""

    if raw_items and isinstance(raw_items[0], Document):
        for item in raw_items:
            text = _clean_transcript_text(str(item.page_content or ""))
            if not text:
                continue
            if text == previous_text:
                continue
            start = float(item.metadata.get("start_seconds", 0.0))
            duration = float(item.metadata.get("duration_seconds", 0.0))
            lines.append(TranscriptLine(text=text, start=start, duration=duration))
            previous_text = text
        return lines

    for item in raw_items:
        text = _clean_transcript_text(str(item.get("text", "")))
        if not text:
            continue
        if text == previous_text:
            continue
        lines.append(
            TranscriptLine(
                text=text,
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
            )
        )
        previous_text = text
    return lines
