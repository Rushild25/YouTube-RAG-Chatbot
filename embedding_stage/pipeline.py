from __future__ import annotations
import os
from datetime import datetime
from typing import Any
import json
import numpy as np
from .config import CHUNKS_FILE, DEFAULT_EMBED_MODEL,EMBEDDINGS_FILE,INDEX_META_FILE,VECTOR_STORE_RECORDS_FILE
from .embedder import build_embedder
from .io_utils import load_translated_chunks, write_json, write_jsonl

def _safe_video_id(raw_video_id: str | None) -> str:
    value = (raw_video_id or "unknown_video").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in value)
    return cleaned or "unknown_video"

def _normalize_chunk_record(chunk: dict[str, Any], chunk_index: int, default_video_id: str | None, source_language: str | None, target_language: str | None) -> dict[str, Any]:
    start_time = float(chunk.get("start_time", 0.0))
    end_time = float(chunk.get("end_time", start_time))
    duration = max(0.0, end_time - start_time)

    video_id = chunk.get("video_id") or default_video_id or "unknown_video"
    safe_video_id = _safe_video_id(str(video_id))
    unique_chunk_id = f"{safe_video_id}::chunk-{chunk_index:06d}"

    return {
        "chunk_id": f"chunk-{chunk_index:06d}",
        "chunk_index": chunk_index,
        "text": str(chunk.get("text", "")).strip(),
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "video_id": chunk.get("video_id"),
        "video_url": chunk.get("video_url", ""),
        "translation_status": chunk.get("translation_status"),
        "source_language": source_language,
        "target_language": target_language
    }


def build_embedding_artifacts(input_json_path: str, output_dir: str, model_name: str = DEFAULT_EMBED_MODEL,) -> dict[str, Any]:
    """Build local embedding artifacts from translated chunks JSON.

        Output files:
      - embeddings.npy: dense vector matrix [n_chunks, dim]
      - chunks.jsonl: normalized chunk payloads for indexing
            - vector_store_records.jsonl: one record per chunk with vector + metadata
      - index_metadata.json: build metadata and source references
    """
    metadata, chunks = load_translated_chunks(input_json_path)

    source_language = metadata.get("source_language")
    target_language = metadata.get("target_language")
    default_video_id = metadata.get("video_id")

    normalized_chunks = [
        _normalize_chunk_record(chunk, idx, default_video_id, source_language, target_language)
        for idx, chunk in enumerate(chunks)
        if str(chunk.get("text", "")).strip()
    ]

    if not normalized_chunks:
        raise ValueError("No non-empty chunks found in input JSON.")

    os.makedirs(output_dir, exist_ok=True)

    texts = [chunk["text"] for chunk in normalized_chunks]
    embedder, backend = build_embedder(model_name=model_name)
    embeddings = embedder.embed_documents(texts)

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape[0] != len(normalized_chunks):
        raise ValueError("Embedding output row count does not match chunk count.")

    embeddings_path = os.path.join(output_dir, EMBEDDINGS_FILE)
    chunks_path = os.path.join(output_dir, CHUNKS_FILE)
    vector_records_path = os.path.join(output_dir, VECTOR_STORE_RECORDS_FILE)
    meta_path = os.path.join(output_dir, INDEX_META_FILE)

    np.save(embeddings_path, embeddings)
    write_jsonl(chunks_path, normalized_chunks)

    with open(vector_records_path, "w", encoding="utf-8") as file:
        for chunk, vector in zip(normalized_chunks, embeddings):
            record = {
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": vector.tolist(),
                "metadata": {
                    "chunk_index": chunk["chunk_index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "duration": chunk["duration"],
                    "video_id": chunk["video_id"],
                    "video_url": chunk["video_url"],
                    "translation_status": chunk["translation_status"],
                    "source_language": chunk["source_language"],
                    "target_language": chunk["target_language"]
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    artifact_meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_json_path": input_json_path,
        "output_dir": output_dir,
        "embedding_backend": backend,
        "embedding_model": model_name if backend == "huggingface" else "tfidf-fallback",
        "chunk_count": len(normalized_chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "video_id": metadata.get("video_id"),
        "video_url": metadata.get("video_url"),
        "source_language": metadata.get("source_language"),
        "target_language": metadata.get("target_language"),
        "translation_status": metadata.get("translation_status"),
    }
    write_json(meta_path, artifact_meta)

    return {
        "embeddings_path": embeddings_path,
        "chunks_path": chunks_path,
        "vector_records_path": vector_records_path,
        "meta_path": meta_path,
        "meta": artifact_meta,
    }
