from __future__ import annotations

import json
import os

import numpy as np

from .config import CHUNKS_FILE, EMBEDDINGS_FILE


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (matrix_norm * query_norm)


class LocalEmbeddingRetriever:
    """Lightweight local retriever for quick validation before DB indexing."""

    def __init__(self, artifact_dir: str) -> None:
        embeddings_path = os.path.join(artifact_dir, EMBEDDINGS_FILE)
        chunks_path = os.path.join(artifact_dir, CHUNKS_FILE)

        self.embeddings = np.load(embeddings_path)
        self.chunks: list[dict] = []

        with open(chunks_path, "r", encoding="utf-8") as file:
            for line in file:
                self.chunks.append(json.loads(line))

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        scores = _cosine_similarity(query_vector, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for index in top_indices:
            item = dict(self.chunks[index])
            item["score"] = float(scores[index])
            results.append(item)

        return results
