from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "youtube_chat_chunks").strip()

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    llm_model: str = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta").strip()

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    top_k: int = int(os.getenv("TOP_K", "5"))


SETTINGS = Settings()
