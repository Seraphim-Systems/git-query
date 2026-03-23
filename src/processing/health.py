"""Health check endpoint for processing service."""

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis
from qdrant_client import QdrantClient

from processing.config import settings

app = FastAPI(title="Processing Service Health")


@app.get("/health")
async def health_check():
    """Check if all database connections are working."""
    health = {
        "status": "healthy",
        "mongodb": False,
        "redis": False,
        "qdrant": False,
    }

    try:
        mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        await mongo_client.admin.command("ping")
        health["mongodb"] = True
        mongo_client.close()
    except Exception as exc:
        health["status"] = "unhealthy"
        health["mongodb_error"] = str(exc)

    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        health["redis"] = True
        await redis_client.close()
    except Exception as exc:
        health["status"] = "unhealthy"
        health["redis_error"] = str(exc)

    try:
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )
        qdrant_client.get_collections()
        health["qdrant"] = True
        qdrant_client.close()
    except Exception as exc:
        health["status"] = "unhealthy"
        health["qdrant_error"] = str(exc)

    return health


@app.get("/stats")
async def get_stats():
    """Get processing statistics from Redis."""
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        stats = await redis_client.hgetall("processing:stats")
        await redis_client.close()
        return stats or {"message": "No stats available yet"}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/collections")
async def get_collection_stats():
    """Get MongoDB collection statistics."""
    try:
        mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        db = mongo_client[settings.mongodb_db]

        raw_count = await db[settings.source_collection].count_documents({})
        cleaned_count = await db[settings.dest_collection].count_documents({})

        pipeline = [
            {
                "$group": {
                    "_id": "$processing_status",
                    "count": {"$sum": 1},
                }
            }
        ]

        status_breakdown = await db[settings.source_collection].aggregate(pipeline).to_list(None)
        mongo_client.close()

        return {
            "raw_repositories": raw_count,
            "cleaned_repositories": cleaned_count,
            "status_breakdown": {
                item["_id"] or "pending": item["count"]
                for item in status_breakdown
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/qdrant")
async def get_qdrant_stats():
    """Get Qdrant collection statistics."""
    try:
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )

        collection_info = qdrant_client.get_collection(settings.vector_collection)
        qdrant_client.close()

        return {
            "collection": settings.vector_collection,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
        }
    except Exception as exc:
        return {"error": str(exc)}
