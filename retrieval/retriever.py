from __future__ import annotations
from ingestion.embedding import EmbeddingService
from vectorstore.qdrant_client import QdrantVectorStore


class Retriever:
    def __init__(self, embedding_service: EmbeddingService, vectorstore: QdrantVectorStore) -> None:
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore

    def retrieve(self, question: str, top_k: int, video_id: str | None = None) -> list[dict]:
        query_vec = self.embedding_service.embed_query(question).tolist()
        hits = self.vectorstore.query(query_vector=query_vec, top_k=top_k, video_id=video_id)

        results: list[dict] = []
        for hit in hits:
            payload = hit["payload"]
            results.append(
                {
                    "score": hit["score"],
                    "video_id": payload.get("video_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                }
            )
        return results
