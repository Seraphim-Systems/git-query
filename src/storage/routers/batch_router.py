"""
Batch operation endpoints
"""

from fastapi import APIRouter, Depends
from qdrant_client.models import PointStruct
from models.batch_models import BatchInsert
from services.db_clients import get_mongo_client, get_redis_client, get_qdrant_client
from auth import get_api_key

router = APIRouter(prefix="/batch", tags=["Batch Operations"])


@router.post("/insert", dependencies=[Depends(get_api_key)])
async def batch_insert(batch: BatchInsert):
    """
    Batch insert data into multiple databases (requires API key).

    WARNING: This operation does NOT use transactions. If one database operation
    succeeds and another fails, you will have partial data across systems, which
    can lead to data inconsistency issues.

    While implementing distributed transactions across MongoDB, Qdrant, and Redis
    is complex, you should be aware of this behavior:
    - Check the 'errors' array in the response for any failures
    - Implement retry logic for failed operations in your application
    - Consider implementing a rollback mechanism for MongoDB/Qdrant if needed
    - Design your data model to handle eventual consistency

    Args:
        batch: BatchInsert object containing data for MongoDB, Qdrant, and/or Redis

    Returns:
        Results object with successful operations and any errors that occurred
    """
    results = {"mongodb": [], "qdrant": [], "redis": []}
    errors = []

    mongo_client = get_mongo_client()
    redis_client = get_redis_client()
    qdrant_client = get_qdrant_client()

    # MongoDB batch insert
    if batch.mongodb_data and mongo_client:
        for insert_op in batch.mongodb_data:
            try:
                db = mongo_client[insert_op.database]
                collection = db[insert_op.collection]
                result = collection.insert_many(insert_op.documents)
                results["mongodb"].append(
                    {
                        "collection": insert_op.collection,
                        "inserted_count": len(result.inserted_ids),
                    }
                )
            except Exception as e:
                errors.append(f"MongoDB insert failed: {str(e)}")

    # Qdrant batch insert
    if batch.qdrant_data and qdrant_client:
        for insert_op in batch.qdrant_data:
            try:
                points = [
                    PointStruct(
                        id=point.get("id"),
                        vector=point["vector"],
                        payload=point.get("payload", {}),
                    )
                    for point in insert_op.points
                ]
                qdrant_client.upsert(
                    collection_name=insert_op.collection, points=points
                )
                results["qdrant"].append(
                    {"collection": insert_op.collection, "inserted_count": len(points)}
                )
            except Exception as e:
                errors.append(f"Qdrant insert failed: {str(e)}")

    # Redis batch insert
    if batch.redis_data and redis_client:
        for item in batch.redis_data:
            try:
                key = item["key"]
                value = item["value"]
                expire = item.get("expire")

                if expire:
                    redis_client.setex(key, expire, value)
                else:
                    redis_client.set(key, value)
                results["redis"].append({"key": key, "status": "success"})
            except Exception as e:
                errors.append(f"Redis insert failed: {str(e)}")

    return {"results": results, "errors": errors if errors else None}
