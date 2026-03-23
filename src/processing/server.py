"""HTTP API app for processing service."""

from processing.health import app
from processing.routers.data_processing_pipeline import router as pipeline_router

app.include_router(pipeline_router)
