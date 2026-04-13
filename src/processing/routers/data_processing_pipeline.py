"""Pipeline execution API routes."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient

from processing.config import settings
from processing.pipelines.ingestion import DataIngestion
from processing.pipelines.preparation import run_preparation_batch
from processing.pipelines.transformation import DataTransformer

router = APIRouter(prefix="/data_processing_pipeline", tags=["data_processing_pipeline"])


class PipelineRunRequest(BaseModel):
    batch_size: int = Field(default=500, ge=1, le=5000)
    max_batches: int = Field(default=1, ge=1, le=200)
    mark_processed: bool = True


@router.post("/run")
async def run_pipeline_once(payload: PipelineRunRequest):
    """Run raw->cleaned data preparation pipeline on the server."""
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
            "pipeline": "data_processing_pipeline",
            "target_collection": settings.dest_collection,
            "summary": aggregate,
            "pending_source_records": pending_count,
            "cleaned_collection_total": cleaned_count,
        }
    finally:
        mongo_client.close()
