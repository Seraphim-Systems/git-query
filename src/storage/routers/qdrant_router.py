"""
Qdrant API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any
from qdrant_client.models import VectorParams, Distance
from src.db.models import QdrantQuery, QdrantInsert
from src.db.clients import get_qdrant_client
from src.db.adapters.qdrant_adapter import QdrantAdapter
from src.processing.qdrant_helpers import serialize_collection_description
from src.storage.auth import get_api_key

router = APIRouter(prefix="/qdrant", tags=["Qdrant"])


# Modern RESTful aliases (kept in addition to legacy routes for compatibility)


@router.post("/collections/{collection}/points", dependencies=[Depends(get_api_key)])
async def create_points(collection: str, payload: Dict[str, Any] = Body(...)):
    """Modern alias for bulk upsert: POST /collections/{collection}/points"""
    return await bulk_upsert_vectors_impl(collection, payload)


@router.post("/collections/{collection}/search", dependencies=[Depends(get_api_key)])
async def search_collection_modern(collection: str, query: Dict[str, Any] = Body(...)):
    """Modern alias for searching a collection"""
    return await search_collection_impl(collection, query)


@router.delete("/collections/{collection}", dependencies=[Depends(get_api_key)])
async def delete_collection_modern(collection: str):
    """Modern alias for deleting collection"""
    return await delete_collection_impl(collection)


@router.delete("/collections/{collection}/points", dependencies=[Depends(get_api_key)])
async def delete_points_modern(collection: str, payload: Dict[str, Any] = Body(...)):
    """Modern alias for deleting points in a collection"""
    return await delete_points_impl(collection, payload)


@router.post("/search", dependencies=[Depends(get_api_key)])
async def search_qdrant(query: QdrantQuery):
    """Search Qdrant vector database (requires API key)"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        adapter = QdrantAdapter(qdrant_client)
        return adapter.search(query.collection, query.vector, limit=query.limit, filter=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/insert", dependencies=[Depends(get_api_key)])
async def insert_qdrant(insert: QdrantInsert):
    """Insert points into Qdrant (requires API key)"""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        adapter = QdrantAdapter(qdrant_client)
        points = [
            {
                "id": point.get("id"),
                "vector": point["vector"],
                "payload": point.get("payload", {}),
            }
            for point in insert.points
        ]
        res = adapter.upsert_points(insert.collection, points, wait=True)
        return {
            "collection": insert.collection,
            "inserted_count": res.get("upserted", 0),
        }
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
        items = []
        for col in getattr(collections, "collections", []):
            try:
                items.append(serialize_collection_description(col))
            except Exception:
                # Ensure a best-effort fallback so a single unexpected item
                # doesn't cause the entire endpoint to fail.
                items.append({"name": str(col), "vectors_count": 0, "points_count": 0})

        return {"collections": items}
    except Exception as e:
        # Provide a clearer error message while avoiding leaking internal
        # client types/structures in the response.
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


async def search_collection_impl(
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
        filt = query.get("filter")

        adapter = QdrantAdapter(qdrant_client)
        return adapter.search(collection, vector, limit=limit, filter=filt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def bulk_upsert_vectors_impl(
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

        adapter = QdrantAdapter(qdrant_client)
        points = [
            {
                "id": point.get("id"),
                "vector": point["vector"],
                "payload": point.get("payload", {}),
            }
            for point in points_data
        ]

        try:
            res = adapter.upsert_points(collection, points, wait=wait)
        except Exception as e:
            # Try to create collection if missing and retry once
            msg = str(e)
            msg_lower = msg.lower()
            # Cover a few common phrasings returned by different qdrant
            # client/http responses (e.g. "doesn't exist", "does not exist",
            # "not found"). This makes the auto-create fallback more
            # robust across qdrant versions and client libraries.
            if (
                "doesn't exist" in msg_lower
                or "does not exist" in msg_lower
                or "not found" in msg_lower
                or ("collection" in msg_lower and "does not" in msg_lower)
            ):
                if points and isinstance(points[0].get("vector"), (list, tuple)):
                    vec_size = len(points[0]["vector"])
                else:
                    vec_size = 768
                try:
                    qdrant_client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
                    )
                except Exception:
                    raise

                res = adapter.upsert_points(collection, points, wait=wait)
            else:
                raise

        return {
            "collection": collection,
            "upserted": res.get("upserted", len(points)),
            "status": "completed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk upsert failed: {str(e)}")


async def delete_collection_impl(collection: str):
    """Delete an entire Qdrant collection."""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        adapter = QdrantAdapter(qdrant_client)
        return adapter.delete_collection(collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete collection failed: {str(e)}")


async def delete_points_impl(collection: str, payload: Dict[str, Any] = Body(...)):
    """Delete specific points in a Qdrant collection by ids or by filter.

    Payload examples:
    - {"ids": ["pt1","pt2"]}
    - {"filter": {...}}  # will be passed through to client if supported
    """
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        ids = payload.get("ids")
        filt = payload.get("filter")
        adapter = QdrantAdapter(qdrant_client)
        try:
            return adapter.delete_points(collection, ids=ids, filter=filt)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete points failed: {str(e)}")
