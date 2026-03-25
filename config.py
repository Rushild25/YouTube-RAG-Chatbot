import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

@dataclass(frozen=True)
class Settings:
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_TOKEN")
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "youtube_chat_chunks").strip()

    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    llm_model: str = os.getenv("LLM_MODEL")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", 650))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", 64))
    top_k: int = int(os.getenv("TOP_K", 5))


SETTINGS = Settings()