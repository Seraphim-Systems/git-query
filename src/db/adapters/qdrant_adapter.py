from typing import Dict, Any, List, Optional
from db.adapters.interfaces import VectorRepository
from db.clients import get_qdrant_client


class QdrantAdapter(VectorRepository):
    def __init__(self, client=None):
        self.client = client or get_qdrant_client()

    def list_collections(self) -> List[str]:
        cols = self.client.get_collections()
        return [c.name for c in cols.collections]

    def upsert_points(self, collection: str, points: List[Dict[str, Any]]):
        # Expect points as dicts with id/vector/payload
        from qdrant_client.models import PointStruct

        pts = [
            PointStruct(
                id=p.get("id"), vector=p["vector"], payload=p.get("payload", {})
            )
            for p in points
        ]
        self.client.upsert(collection_name=collection, points=pts)
        return {"upserted": len(pts)}

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            filter=filter or {},
        )
        # Return a normalized dict with hits
        return {"results": [r.dict() for r in res]}
