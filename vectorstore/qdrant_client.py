from __future__ import annotations
import uuid
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore as LangChainQdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams
from config import SETTINGS


class QdrantVectorStore:
    def __init__(self, embeddings: Embeddings) -> None:
        self.client = QdrantClient(path="./qdrant_data")
        self.collection = SETTINGS.qdrant_collection
        self.embeddings = embeddings

        self._ensure_collection()
        self.store = LangChainQdrantVectorStore(
            client=self.client,
            collection_name=self.collection,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        probe_vector = self.embeddings.embed_query("dimension_probe")
        vector_dim = len(probe_vector)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    def upsert_documents(self, documents: list[Document], ids: list[str]) -> None:
        if not documents:
            return
        uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, source_id)) for source_id in ids]
        self.store.add_documents(documents=documents, ids=uuid_ids)

    def as_retriever(self, top_k: int, video_id: str | None = None):
        search_kwargs: dict = {"k": top_k}
        if video_id:
            search_kwargs["filter"] = Filter(
                must=[FieldCondition(key="metadata.video_id", match=MatchValue(value=video_id))]
            )
        return self.store.as_retriever(search_kwargs=search_kwargs)
