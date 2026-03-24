from __future__ import annotations
from dataclasses import dataclass
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.transcript_processor import TranscriptLine
import tiktoken

@dataclass
class Chunk:
    text: str


_TOKEN_FALLBACK_RE = re.compile(r"\S+")
_ENCODER = tiktoken.get_encoding("cl100k_base") if tiktoken else None


def _token_length(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=()))
    return len(_TOKEN_FALLBACK_RE.findall(text))

def chunk_transcript(lines: list[TranscriptLine], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if not lines:
        return []

    full_text = "\n".join(line.text for line in lines)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )
    parts = [part.strip() for part in splitter.split_text(full_text) if part.strip()]
    return [Chunk(text=part) for part in parts]
