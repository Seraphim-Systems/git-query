"""
Qdrant API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any
from qdrant_client.models import VectorParams, Distance
from src.db.models import QdrantQuery, QdrantInsert
from src.db.clients import get_qdrant_client
from src.processing.qdrant_helpers import serialize_collection_description
from src.storage.auth import get_api_key

router = APIRouter(prefix="/qdrant", tags=["Qdrant"])


# Modern RESTful aliases (kept in addition to legacy routes for compatibility)


@router.post("/collections/{collection}/points", dependencies=[Depends(get_api_key)])
async def create_points(collection: str, payload: Dict[str, Any] = Body(...)):
    """Modern alias for bulk upsert: POST /collections/{collection}/points"""
    return await bulk_upsert_vectors(collection, payload)


@router.post("/collections/{collection}/search", dependencies=[Depends(get_api_key)])
async def search_collection_modern(collection: str, query: Dict[str, Any] = Body(...)):
    """Modern alias for searching a collection"""
    return await search_collection(collection, query)


@router.delete("/collections/{collection}", dependencies=[Depends(get_api_key)])
async def delete_collection_modern(collection: str):
    """Modern alias for deleting collection"""
    return await delete_collection(collection)


@router.delete("/collections/{collection}/points", dependencies=[Depends(get_api_key)])
async def delete_points_modern(collection: str, payload: Dict[str, Any] = Body(...)):
    """Modern alias for deleting points in a collection"""
    return await delete_points(collection, payload)


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
            {
                "id": point.get("id"),
                "vector": point["vector"],
                "payload": point.get("payload", {}),
            }
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

        # Use plain dicts for points to avoid client-model/version serialization
        # differences that can cause server-side format errors.
        points = [
            {
                "id": point.get("id"),
                "vector": point["vector"],
                "payload": point.get("payload", {}),
            }
            for point in points_data
        ]

        try:
            qdrant_client.upsert(collection_name=collection, points=points, wait=wait)
        except Exception as e:
            # If the collection doesn't exist, create it using the incoming
            # vector dimensionality then retry the upsert once.
            msg = str(e)
            if (
                "doesn't exist" in msg
                or "Not found" in msg
                or "Collection" in msg
                and "doesn't exist" in msg
            ):
                # Infer vector size from first point if available
                if points and isinstance(points[0].get("vector"), (list, tuple)):
                    vec_size = len(points[0]["vector"])
                else:
                    vec_size = 768

                try:
                    qdrant_client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=vec_size, distance=Distance.COSINE
                        ),
                    )
                except Exception:
                    # If create_collection fails, re-raise original error
                    raise

                # Retry upsert once
                qdrant_client.upsert(
                    collection_name=collection, points=points, wait=wait
                )
            else:
                raise

        return {
            "collection": collection,
            "upserted": len(points),
            "status": "completed",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk upsert failed: {str(e)}")


@router.delete("/{collection}", dependencies=[Depends(get_api_key)])
async def delete_collection(collection: str):
    """Delete an entire Qdrant collection."""
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        # Preferred method on client
        if hasattr(qdrant_client, "delete_collection"):
            result = qdrant_client.delete_collection(collection_name=collection)
            return {"deleted": bool(result)}

        # Fallback to HTTP-style call if client uses a different API
        if hasattr(qdrant_client, "_client") and hasattr(
            qdrant_client._client, "delete_collection"
        ):
            result = qdrant_client._client.delete_collection(collection_name=collection)
            return {"deleted": bool(result)}

        raise HTTPException(
            status_code=501, detail="Delete collection not supported by client"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Delete collection failed: {str(e)}"
        )


@router.post("/{collection}/delete_points", dependencies=[Depends(get_api_key)])
async def delete_points(collection: str, payload: Dict[str, Any] = Body(...)):
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
        selector = None
        # Try client-native deletion APIs
        if ids:
            # Many qdrant client versions accept a `points_selector` or `points` kwarg
            try:
                if hasattr(qdrant_client, "delete"):
                    qdrant_client.delete(
                        collection_name=collection, points_selector={"ids": ids}
                    )
                    return {"deleted": len(ids)}
                # fallback to delete_points if present
                if hasattr(qdrant_client, "delete_points"):
                    qdrant_client.delete_points(collection_name=collection, ids=ids)
                    return {"deleted": len(ids)}
            except Exception:
                # fall through to attempt HTTP API
                pass

        # If filter provided, try passing through
        if payload.get("filter"):
            try:
                qdrant_client.delete(
                    collection_name=collection, filter=payload.get("filter")
                )
                return {"deleted": "filter_applied"}
            except Exception:
                pass

        raise HTTPException(
            status_code=400, detail="Unsupported delete request payload"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete points failed: {str(e)}")
