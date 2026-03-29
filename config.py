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

    llm_provider: str = os.getenv("LLM_PROVIDER", "huggingface").strip().lower()
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    groq_llm_model: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile").strip()

    transcript_source_mode: str = os.getenv("TRANSCRIPT_SOURCE_MODE", "transcript-api").strip().lower()

    groq_whisper_model: str = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()
    temp_audio_dir: str = os.getenv("TEMP_AUDIO_DIR", "./tmp_audio").strip()
    ffmpeg_path: str | None = os.getenv("FFMPEG_PATH") or None


SETTINGS = Settings()