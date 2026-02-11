"""
Qdrant API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List
from qdrant_client.models import PointStruct
from db.models import QdrantQuery, QdrantInsert
from db.clients import get_qdrant_client
from storage.auth import get_api_key

router = APIRouter(prefix="/qdrant", tags=["Qdrant"])


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


@router.post("/{collection}/search", dependencies=[Depends(get_api_key)])
async def search_collection(
    collection: str,
    query: Dict[str, Any] = Body(
        ...,
        example={
            "vector": [0.1, 0.2, 0.3],
            "limit": 10,
            "filter": {},
            "with_payload": True,
        },
    ),
):
    """Search for similar vectors in a Qdrant collection"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        vector = query.get("vector")
        limit = query.get("limit", 10)
        score_threshold = query.get("score_threshold")

        results = qdrant_client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            score_threshold=score_threshold,
        )

        return {
            "collection": collection,
            "count": len(results),
            "results": [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/{collection}/bulk", dependencies=[Depends(get_api_key)])
async def bulk_upsert_vectors(
    collection: str,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "points": [
                {"id": "1", "vector": [0.1, 0.2], "payload": {"name": "item1"}},
                {"id": "2", "vector": [0.3, 0.4], "payload": {"name": "item2"}},
            ],
            "wait": True,
        },
    ),
):
    """
    Bulk upsert points into a Qdrant collection.

    Optimized for loading large vector datasets. Uses upsert so it's idempotent.

    Args:
        collection: Collection name
        payload: {
            "points": [{"id": "1", "vector": [...], "payload": {...}}],
            "wait": True  # Wait for operation to complete
        }
    """
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        points_data = payload.get("points", [])
        wait = payload.get("wait", True)

        points = [
            PointStruct(
                id=point.get("id"),
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in points_data
        ]

        qdrant_client.upsert(collection_name=collection, points=points, wait=wait)

        return {
            "collection": collection,
            "upserted": len(points),
            "status": "completed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk upsert failed: {str(e)}")
