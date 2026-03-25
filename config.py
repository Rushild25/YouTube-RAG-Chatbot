from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass(frozen=True)
class Settings:
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_TOKEN")
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION")

    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    llm_model: str = os.getenv("LLM_MODEL")

    chunk_size: int = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP"))
    embedding_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE"))
    top_k: int = int(os.getenv("TOP_K"))


SETTINGS = Settings()