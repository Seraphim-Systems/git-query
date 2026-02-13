"""
MongoDB API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, Any
from src.db.models import MongoQuery, MongoInsert
from src.db.clients import get_mongo_client
from src.storage.auth import get_api_key
from pymongo import UpdateOne
from bson import ObjectId

router = APIRouter(prefix="/mongodb", tags=["MongoDB"])


# Modern RESTful aliases


@router.post("/collections/{collection}/query", dependencies=[Depends(get_api_key)])
async def query_collection_modern(collection: str, query: Dict[str, Any] = Body(...)):
    """Modern alias for querying a collection."""
    return await query_collection_impl(collection, query)


@router.post("/collections/{collection}/bulk", dependencies=[Depends(get_api_key)])
async def bulk_upsert_collection_modern(
    collection: str, payload: Dict[str, Any] = Body(...)
):
    """Modern alias for bulk upsert into a collection."""
    return await bulk_upsert_collection_impl(collection, payload)


@router.delete(
    "/collections/{collection}/documents", dependencies=[Depends(get_api_key)]
)
async def delete_from_collection_modern(
    collection: str, payload: Dict[str, Any] = Body(...)
):
    """Modern alias for deleting documents from a collection."""
    return await _delete_from_collection_impl(collection, payload)


@router.get(
    "/collections/{collection}/documents/{doc_id}", dependencies=[Depends(get_api_key)]
)
async def get_document_by_id(collection: str, doc_id: str):
    """Fetch a single document by its _id (string)."""
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    try:
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection(collection)
        try:
            oid = ObjectId(doc_id)
        except Exception:
            oid = doc_id
        doc = coll.find_one({"_id": oid})
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return {"document": doc}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@router.delete(
    "/collections/{collection}/documents/{doc_id}", dependencies=[Depends(get_api_key)]
)
async def delete_document_by_id(collection: str, doc_id: str):
    """Delete a single document by its _id."""
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    try:
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection(collection)
        try:
            oid = ObjectId(doc_id)
        except Exception:
            oid = doc_id
        result = coll.delete_one({"_id": oid})
        return {"deleted": int(result.deleted_count)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


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
        db = mongo_client.get_database(query.database)
        collection = db.get_collection(query.collection)

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
        db = mongo_client.get_database(insert.database)
        collection = db.get_collection(insert.collection)

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
        db = mongo_client.get_database(database)
        collections = db.list_collection_names()
        return {"database": database, "collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@router.delete("/collections/{collection}", dependencies=[Depends(get_api_key)])
async def drop_mongodb_collection(collection: str, database: str = "gitquery"):
    """Drop a MongoDB collection. Returns whether the collection was dropped.

    Note: dropping a collection removes its data and metadata. This is a
    destructive operation; callers must ensure they intend to remove the
    collection entirely.
    """
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client.get_database(database)
        # Use PyMongo's drop_collection which returns None; check existence
        # beforehand to provide a clear boolean response.
        exists = collection in db.list_collection_names()
        if not exists:
            return {"collection": collection, "dropped": False, "reason": "not_found"}

        db.drop_collection(collection)
        return {"collection": collection, "dropped": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drop failed: {str(e)}")


async def query_collection_impl(
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
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection(collection)

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


async def bulk_upsert_collection_impl(
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
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection(collection)

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


def _cast_id_if_needed(filter_query: Dict[str, Any]):
    if "_id" in filter_query and isinstance(filter_query["_id"], str):
        val = filter_query["_id"]
        try:
            filter_query["_id"] = ObjectId(val)
        except Exception:
            pass


async def _delete_from_collection_impl(collection: str, payload: Dict[str, Any]):
    """Internal implementation for deleting documents from a collection.

    This is intentionally not exposed as a POST route; callers should use the
    modern `DELETE /collections/{collection}/documents` route which accepts a
    JSON body describing the filter.
    """
    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection(collection)

        filter_query = payload.get("filter", {})
        many = bool(payload.get("many", True))

        _cast_id_if_needed(filter_query)

        if many:
            result = coll.delete_many(filter_query)
            deleted = result.deleted_count
        else:
            result = coll.delete_one(filter_query)
            deleted = result.deleted_count

        return {"deleted": int(deleted)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
