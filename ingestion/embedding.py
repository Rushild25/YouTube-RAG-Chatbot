from __future__ import annotations
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = HuggingFaceEmbeddings(model_name=model_name)

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        vectors = self._model.embed_documents(texts)
        matrix = np.asarray(vectors, dtype=np.float32)
        # Normalize for cosine-sim consistency in vector search.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix

    def embed_query(self, query: str) -> np.ndarray:
        vector = np.asarray(self._model.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
