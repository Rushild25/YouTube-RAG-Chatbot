from __future__ import annotations
import hashlib

def build_chunk_id(video_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{video_id}::chunk-{chunk_index:06d}::{digest}"
