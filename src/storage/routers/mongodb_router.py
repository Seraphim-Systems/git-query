"""
MongoDB API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any, List
from db.models import MongoQuery, MongoInsert
from services.db_clients import get_mongo_client
from auth import get_api_key
from pymongo import UpdateOne

router = APIRouter(prefix="/mongodb", tags=["MongoDB"])


@router.post("/query", dependencies=[Depends(get_api_key)])
async def query_mongodb(query: MongoQuery):
    """
    Query MongoDB collections (requires API key).

    WARNING: This endpoint allows arbitrary filter queries. While MongoDB's
    query language is safe from injection when using the official driver,
    unrestricted queries can enable expensive operations (e.g., full collection
    scans with complex filters) that may cause performance issues or denial of
    service. Consider:
    - Adding query complexity limits for production use
    - Restricting certain operators ($where, $expr)
    - Requiring authentication for complex queries
    - Monitoring query execution times

    Args:
        query: MongoQuery object containing database, collection, filter, and options
    """
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[query.database]
        collection = db[query.collection]

        cursor = (
            collection.find(query.filter, query.projection)
            .limit(query.limit)
            .skip(query.skip)
        )

        if query.sort:
            cursor = cursor.sort(list(query.sort.items()))

        documents = list(cursor)

        # Convert ObjectId to string
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"count": len(documents), "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/insert", dependencies=[Depends(get_api_key)])
async def insert_mongodb(insert: MongoInsert):
    """Insert documents into MongoDB (requires API key)"""
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[insert.database]
        collection = db[insert.collection]

        result = collection.insert_many(insert.documents)

        return {
            "inserted_count": len(result.inserted_ids),
            "inserted_ids": [str(id) for id in result.inserted_ids],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@router.get("/collections", dependencies=[Depends(get_api_key)])
async def list_mongodb_collections(database: str = "gitquery"):
    """List all collections in a MongoDB database (requires API key)"""
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[database]
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
    """
    Query a specific MongoDB collection.

    Supports filter, projection, limit, skip, and sort operations.
    """
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client["gitquery"]
        coll = db[collection]

        filter_query = query.get("filter", {})
        projection = query.get("projection")
        limit = query.get("limit", 100)
        skip = query.get("skip", 0)
        sort = query.get("sort")

        cursor = coll.find(filter_query, projection).limit(limit).skip(skip)

        if sort:
            cursor = cursor.sort(list(sort.items()))

        documents = list(cursor)

        # Convert ObjectId to string
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"count": len(documents), "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/{collection}/bulk", dependencies=[Depends(get_api_key)])
async def bulk_upsert_collection(
    collection: str,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "documents": [{"_id": "1", "data": "value"}],
            "ordered": False,
            "upsert": True,
        },
    ),
):
    """
    Bulk insert or upsert documents into a MongoDB collection.

    Optimized for loading large datasets. If upsert=True, will update existing
    documents with matching _id and insert new ones.

    Args:
        collection: Collection name
        payload: {
            "documents": List of documents to insert/upsert
            "ordered": Whether to stop on first error (default: False)
            "upsert": Whether to upsert instead of insert (default: True)
        }
    """
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client["gitquery"]
        coll = db[collection]

        documents = payload.get("documents", [])
        ordered = payload.get("ordered", False)
        upsert = payload.get("upsert", True)

        if not documents:
            return {"inserted": 0, "updated": 0, "errors": []}

        inserted_count = 0
        updated_count = 0
        errors = []

        if upsert:
            # Use bulk_write with UpdateOne operations for upsert
            operations = []
            for doc in documents:
                if "_id" not in doc:
                    # Generate ID if not provided
                    operations.append(
                        UpdateOne(
                            {"_id": doc.get("_id", str(id(doc)))},
                            {"$set": doc},
                            upsert=True,
                        )
                    )
                else:
                    operations.append(
                        UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
                    )

            result = coll.bulk_write(operations, ordered=ordered)
            inserted_count = result.upserted_count
            updated_count = result.modified_count
        else:
            # Use insert_many for plain inserts
            try:
                result = coll.insert_many(documents, ordered=ordered)
                inserted_count = len(result.inserted_ids)
            except Exception as e:
                errors.append(str(e))

        return {"inserted": inserted_count, "updated": updated_count, "errors": errors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk operation failed: {str(e)}")
