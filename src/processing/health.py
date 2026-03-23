"""Health check endpoint for processing service"""

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis
from qdrant_client import QdrantClient
from processing.config import settings
from processing.pipelines.ingestion import DataIngestion
from processing.pipelines.preparation import run_preparation_batch
from processing.pipelines.transformation import DataTransformer

app = FastAPI(title="Processing Service Health")


class PipelineRunRequest(BaseModel):
    batch_size: int = Field(default=500, ge=1, le=5000)
    max_batches: int = Field(default=1, ge=1, le=200)
    mark_processed: bool = True


@app.post("/pipeline/run")
async def run_pipeline_once(payload: PipelineRunRequest):
    """Run raw->cleaned data preparation pipeline on the server.

    This endpoint intentionally handles data preparation and Mongo persistence
    only. Model training is managed separately.
    """
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    db = mongo_client[settings.mongodb_db]
    ingestion = DataIngestion(db)
    transformer = DataTransformer()

    aggregate = {
        "batches_requested": payload.max_batches,
        "batches_executed": 0,
        "fetched": 0,
        "cleaned": 0,
        "saved": 0,
        "errors": 0,
    }

    try:
        for _ in range(payload.max_batches):
            result = await run_preparation_batch(
                ingestion=ingestion,
                transformer=transformer,
                limit=payload.batch_size,
                mark_processed=payload.mark_processed,
            )
            stats = result["stats"]

            aggregate["batches_executed"] += 1
            aggregate["fetched"] += stats["fetched"]
            aggregate["cleaned"] += stats["cleaned"]
            aggregate["saved"] += stats["saved"]
            aggregate["errors"] += stats["errors"]

            if stats["fetched"] == 0:
                break

        pending_count = await db[settings.source_collection].count_documents(
            {
                "$or": [
                    {"processing_status": {"$exists": False}},
                    {"processing_status": "pending"},
                ]
            }
        )

        cleaned_count = await db[settings.dest_collection].count_documents({})

        return {
            "status": "ok",
            "pipeline": "data_preparation",
            "target_collection": settings.dest_collection,
            "summary": aggregate,
            "pending_source_records": pending_count,
            "cleaned_collection_total": cleaned_count,
        }
    finally:
        mongo_client.close()


@app.get("/health")
async def health_check():
    """Check if all database connections are working"""
    health = {
        "status": "healthy",
        "mongodb": False,
        "redis": False,
        "qdrant": False
    }
    
    try:
        # Check MongoDB
        mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        await mongo_client.admin.command('ping')
        health["mongodb"] = True
        mongo_client.close()
    except Exception as e:
        health["status"] = "unhealthy"
        health["mongodb_error"] = str(e)
    
    try:
        # Check Redis
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        health["redis"] = True
        await redis_client.close()
    except Exception as e:
        health["status"] = "unhealthy"
        health["redis_error"] = str(e)
    
    try:
        # Check Qdrant
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        qdrant_client.get_collections()
        health["qdrant"] = True
        qdrant_client.close()
    except Exception as e:
        health["status"] = "unhealthy"
        health["qdrant_error"] = str(e)
    
    return health


@app.get("/stats")
async def get_stats():
    """Get processing statistics from Redis"""
    try:
        redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
        stats = await redis_client.hgetall("processing:stats")
        await redis_client.close()
        return stats or {"message": "No stats available yet"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/collections")
async def get_collection_stats():
    """Get MongoDB collection statistics"""
    try:
        mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        db = mongo_client[settings.mongodb_db]
        
        # Get counts from both collections
        raw_count = await db[settings.source_collection].count_documents({})
        cleaned_count = await db[settings.dest_collection].count_documents({})
        
        # Get processing status breakdown
        pipeline = [
            {
                "$group": {
                    "_id": "$processing_status",
                    "count": {"$sum": 1}
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
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/qdrant")
async def get_qdrant_stats():
    """Get Qdrant collection statistics"""
    try:
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        
        collection_info = qdrant_client.get_collection(settings.vector_collection)
        
        qdrant_client.close()
        
        return {
            "collection": settings.vector_collection,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status
        }
    except Exception as e:
        return {"error": str(e)}
