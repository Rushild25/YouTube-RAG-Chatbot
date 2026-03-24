from __future__ import annotations
from dataclasses import dataclass
from langchain_core.documents import Document

@dataclass
class TranscriptLine:
    text: str
    start: float
    duration: float


def normalize_transcript_lines(raw_items: list[dict] | list[Document]) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []

    if raw_items and isinstance(raw_items[0], Document):
        for item in raw_items:
            text = str(item.page_content or "").strip()
            if not text:
                continue
            start = float(item.metadata.get("start_seconds", 0.0))
            duration = float(item.metadata.get("duration_seconds", 0.0))
            lines.append(TranscriptLine(text=text, start=start, duration=duration))
        return lines

    for item in raw_items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        lines.append(
            TranscriptLine(
                text=text,
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
            )
        )
    return lines
