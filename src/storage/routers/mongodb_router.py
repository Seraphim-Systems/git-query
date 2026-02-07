"""
MongoDB API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from models.mongodb_models import MongoQuery, MongoInsert
from services.db_clients import get_mongo_client
from auth import get_api_key

router = APIRouter(prefix="/api/mongodb", tags=["MongoDB"])


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
