from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import DEFAULT_EMBED_MODEL


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        pass


class HuggingFaceSentenceEmbedder:
    """Sentence-transformer embedder via LangChain HuggingFace integration."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings

        self.model_name = model_name
        self._embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._embedder.embed_documents(texts)
        return np.asarray(vectors, dtype=np.float32)


class TfidfFallbackEmbedder:
    """Zero-dependency fallback if sentence-transformers stack is unavailable."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(max_features=4096)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        return matrix.toarray().astype(np.float32)


def build_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> tuple[Embedder, str]:
    """Create primary embedder; fallback to TF-IDF if runtime deps are missing."""
    try:
        return HuggingFaceSentenceEmbedder(model_name=model_name), "huggingface"
    except Exception:
        return TfidfFallbackEmbedder(), "tfidf-fallback"
