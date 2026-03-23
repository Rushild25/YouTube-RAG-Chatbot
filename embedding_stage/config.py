from __future__ import annotations
import os

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_BATCH_SIZE = 64

# Files written by embedding pipeline
EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.jsonl"
INDEX_META_FILE = "index_metadata.json"
VECTOR_STORE_RECORDS_FILE = "vector_store_records.jsonl"

VECTOR_DB_PROVIDER = os.getenv("VECTOR_DB_PROVIDER", "qdrant").strip().lower()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("ODRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "youtube_chat_chunks").strip()
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "cosine").strip().lower()
QDRANT_CREATE_IF_MISSING = os.getenv("QDRANT_CREATE_IF_MISSING", "true").strip().lower() in {
    "1", "true", "yes", "on"
}