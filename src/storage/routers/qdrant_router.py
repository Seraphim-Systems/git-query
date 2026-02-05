"""
Qdrant API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from qdrant_client.models import PointStruct
from ..models.qdrant_models import QdrantQuery, QdrantInsert
from ..services.db_clients import get_qdrant_client
from ..auth import get_api_key

router = APIRouter(prefix="/api/qdrant", tags=["Qdrant"])


@router.post("/search", dependencies=[Depends(get_api_key)])
async def search_qdrant(query: QdrantQuery):
    """Search Qdrant vector database (requires API key)"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        results = qdrant_client.search(
            collection_name=query.collection,
            query_vector=query.vector,
            limit=query.limit,
            score_threshold=query.score_threshold,
        )

        return {
            "collection": query.collection,
            "count": len(results),
            "results": [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/insert", dependencies=[Depends(get_api_key)])
async def insert_qdrant(insert: QdrantInsert):
    """Insert points into Qdrant (requires API key)"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        points = [
            PointStruct(
                id=point.get("id"),
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in insert.points
        ]

        qdrant_client.upsert(collection_name=insert.collection, points=points)

        return {"collection": insert.collection, "inserted_count": len(points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@router.get("/collections", dependencies=[Depends(get_api_key)])
async def list_qdrant_collections():
    """List all Qdrant collections (requires API key)"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "vectors_count": col.vectors_count,
                    "points_count": col.points_count,
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )
