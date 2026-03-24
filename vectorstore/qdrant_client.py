from __future__ import annotations
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import SETTINGS


class QdrantVectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(path="./qdrant_data")
        self.collection = SETTINGS.qdrant_collection

    def ensure_collection(self, vector_dim: int) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
        )

    def upsert(self, records: list[dict]) -> None:
        if not records:
            return

        vector_dim = len(records[0]["embedding"])
        self.ensure_collection(vector_dim)

        points: list[models.PointStruct] = []
        for item in records:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(item["id"])))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=item["embedding"],
                    payload=item["metadata"],
                )
            )

        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def query(self, query_vector: list[float], top_k: int, video_id: str | None = None) -> list[dict]:
        q_filter = None
        if video_id:
            q_filter = models.Filter(
                must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
            )

        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                query_filter=q_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            hits = response.points

        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload or {},
            }
            for hit in hits
        ]
