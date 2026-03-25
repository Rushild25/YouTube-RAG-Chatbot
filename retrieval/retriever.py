from __future__ import annotations
from vectorstore.qdrant_client import QdrantVectorStore


class Retriever:
    def __init__(self, vectorstore: QdrantVectorStore) -> None:
        self.vectorstore = vectorstore

    def retrieve(self, question: str, top_k: int, video_id: str | None = None) -> list[dict]:
        retriever = self.vectorstore.as_retriever(top_k=top_k, video_id=video_id)
        docs = retriever.invoke(question)

        return [
            {
                "video_id": doc.metadata.get("video_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
            }
            for doc in docs
        ]
