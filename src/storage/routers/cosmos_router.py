"""
Cosmos DB API endpoints (Mongo-compatible wire protocol)

These endpoints mirror the MongoDB router but operate against the configured
Cosmos DB (using the Mongo API). They reuse the same Pydantic models as
MongoDB endpoints for compatibility.
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List
from src.db.models import MongoQuery, MongoInsert
from src.db.clients import get_cosmos_client
from src.storage.auth import get_api_key

router = APIRouter(prefix="/cosmos", tags=["CosmosDB"])


@router.post("/query", dependencies=[Depends(get_api_key)])
async def query_cosmos(query: MongoQuery):
    """Query Cosmos DB collections (requires API key)."""
    client = get_cosmos_client()
    if not client:
        raise HTTPException(status_code=503, detail="CosmosDB not available")

    try:
        db = client[query.database]
        collection = db[query.collection]

        cursor = (
            collection.find(query.filter, query.projection)
            .limit(query.limit)
            .skip(query.skip)
        )

        if query.sort:
            cursor = cursor.sort(list(query.sort.items()))

        documents = list(cursor)
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"count": len(documents), "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/insert", dependencies=[Depends(get_api_key)])
async def insert_cosmos(insert: MongoInsert):
    """Insert documents into Cosmos DB (requires API key)"""
    client = get_cosmos_client()
    if not client:
        raise HTTPException(status_code=503, detail="CosmosDB not available")

    try:
        db = client[insert.database]
        collection = db[insert.collection]

        result = collection.insert_many(insert.documents)

        return {
            "inserted_count": len(result.inserted_ids),
            "inserted_ids": [str(id) for id in result.inserted_ids],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@router.get("/collections", dependencies=[Depends(get_api_key)])
async def list_cosmos_collections(database: str = "gitquery"):
    """List all collections in a Cosmos DB database (requires API key)"""
    client = get_cosmos_client()
    if not client:
        raise HTTPException(status_code=503, detail="CosmosDB not available")

    try:
        db = client[database]
        collections = db.list_collection_names()
        return {"database": database, "collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@router.post("/{collection}/query", dependencies=[Depends(get_api_key)])
async def query_collection(
    collection: str,
    query: Dict[str, Any] = Body(
        ...,
        example={
            "filter": {},
            "projection": None,
            "limit": 100,
            "skip": 0,
            "sort": None,
        },
    ),
):
    """Query a specific Cosmos DB collection."""
    client = get_cosmos_client()
    if not client:
        raise HTTPException(status_code=503, detail="CosmosDB not available")

    try:
        db = client[query.get("database", "gitquery")]
        coll = db[collection]

        cursor = (
            coll.find(query.get("filter", {}), query.get("projection"))
            .limit(int(query.get("limit", 100)))
            .skip(int(query.get("skip", 0)))
        )

        if query.get("sort"):
            cursor = cursor.sort(list(query.get("sort").items()))

        documents = list(cursor)
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"count": len(documents), "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/{collection}/bulk", dependencies=[Depends(get_api_key)])
async def bulk_upsert_collection(collection: str, payload: Dict[str, Any] = Body(...)):
    """Bulk upsert into a Cosmos DB collection using bulk write operations."""
    client = get_cosmos_client()
    if not client:
        raise HTTPException(status_code=503, detail="CosmosDB not available")

    try:
        db = client[payload.get("database", "gitquery")]
        coll = db[collection]

        ordered = bool(payload.get("ordered", False))
        upsert = bool(payload.get("upsert", True))
        documents = payload.get("documents", [])

        # Prepare bulk operations (replace_one with upsert for upsert behavior)
        from pymongo import ReplaceOne

        operations = []
        for doc in documents:
            if "_id" in doc:
                operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=upsert))
            else:
                operations.append(ReplaceOne(doc, doc, upsert=True))

        result = coll.bulk_write(operations, ordered=ordered)

        return {
            "matched": getattr(result, "matched_count", None),
            "modified": getattr(result, "modified_count", None),
            "upserted": getattr(result, "upserted_count", None),
            "inserted": len(documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk upsert failed: {str(e)}")
