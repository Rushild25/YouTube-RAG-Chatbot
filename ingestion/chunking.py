from __future__ import annotations
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.transcript_processor import TranscriptLine

@dataclass
class Chunk:
    text: str
    start_time: float
    end_time: float

def chunk_transcript(lines: list[TranscriptLine], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if not lines:
        return []

    full_text = "\n".join(line.text for line in lines)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    parts = [part.strip() for part in splitter.split_text(full_text) if part.strip()]

    # Map split chunks to approximate transcript timeline by text-position ratio.
    total_chars = max(1, len(full_text))
    total_start = lines[0].start
    total_end = max(line.start + line.duration for line in lines)
    total_span = max(0.01, total_end - total_start)

    chunks: list[Chunk] = []
    cursor = 0
    for part in parts:
        pos = full_text.find(part, cursor)
        if pos < 0:
            pos = cursor
        end_pos = min(total_chars, pos + len(part))

        start_ratio = pos / total_chars
        end_ratio = max(start_ratio, end_pos / total_chars)

        start_time = total_start + (total_span * start_ratio)
        end_time = total_start + (total_span * end_ratio)
        if end_time <= start_time:
            end_time = start_time + 0.01

        chunks.append(Chunk(text=part, start_time=start_time, end_time=end_time))
        cursor = end_pos

    return chunks
