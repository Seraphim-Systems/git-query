"""FastAPI application for the recommendation service."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from .config import settings
from .models import (
    RecommendationRequest,
    RecommendationResponse,
    UserInteraction,
    EvaluationMetrics,
    RepositoryResult,
)
from .database import db_manager
from .engines import BaselineEngine, HybridRetrievalEngine, PersonalizedEngine
from .services import (
    EmbeddingService,
    RerankerService,
    PersonalizationService,
    ABTestService,
    ModelRegistryService,
)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    await db_manager.connect()

    # Initialize services
    app.state.embedding_service = EmbeddingService()
    app.state.reranker_service = RerankerService()
    app.state.personalization_service = PersonalizationService()
    app.state.ab_test_service = ABTestService()
    app.state.registry_service = ModelRegistryService()

    # Initialize engines
    app.state.engines = {
        "baseline": BaselineEngine(),
        "hybrid": HybridRetrievalEngine(
            embedding_service=app.state.embedding_service,
            reranker_service=app.state.reranker_service,
        ),
        "personalized": PersonalizedEngine(
            embedding_service=app.state.embedding_service,
            reranker_service=app.state.reranker_service,
        ),
    }

    # Load active models
    await app.state.embedding_service.load_active_model()
    await app.state.reranker_service.load_active_model()

    yield

    # Shutdown
    await db_manager.close()


app = FastAPI(
    title="Git-Query Recommendation API",
    description="AI-powered repository recommendation system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow_origins=["*"] and allow_credentials=True cannot be combined
# (browsers reject it per the CORS spec). For a public API, omit credentials.
# To support credentialed requests, replace ["*"] with explicit origin list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Root Redirect =====

@app.get("/")
async def root():
    """Redirect to frontend webserver."""
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8080")
    return RedirectResponse(url=frontend_url)


# ===== Health Check =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "recommender",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ===== Recommendation Endpoints =====

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get repository recommendations.

    This endpoint:
    1. Determines which variant to use (A/B testing)
    2. Generates recommendations using the appropriate engine
    3. Returns ranked results
    """
    start_time = time.time()

    try:
        # Determine variant
        ab_test_service: ABTestService = app.state.ab_test_service
        variant = await ab_test_service.get_variant_for_user(
            request.user_id, request.variant
        )

        # Get appropriate engine
        engine = app.state.engines.get(variant, app.state.engines["baseline"])

        # Generate recommendations
        results = await engine.recommend(request)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build response
        response = RecommendationResponse(
            query=request.query,
            user_id=request.user_id,
            results=results,
            total_candidates=len(results),
            processing_time_ms=processing_time_ms,
            variant=variant,
            personalized=variant == "personalized" and request.user_id is not None,
            filters_applied={
                "language": request.language,
                "min_stars": request.min_stars,
                "license": request.license,
            },
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/recommend/explain/{repo_id}")
async def explain_recommendation(repo_id: str, request: RecommendationRequest):
    """Explain why a repository was recommended."""
    try:
        # Get variant
        ab_test_service: ABTestService = app.state.ab_test_service
        variant = await ab_test_service.get_variant_for_user(
            request.user_id, request.variant
        )

        # Get engine and explanation
        engine = app.state.engines.get(variant, app.state.engines["baseline"])
        explanation = await engine.explain(repo_id, request)

        return explanation

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


# ===== Interaction Tracking =====

@app.post("/interaction")
async def log_interaction(
    interaction: UserInteraction,
    background_tasks: BackgroundTasks,
):
    """
    Log a user interaction.

    This is used for:
    - Learning user preferences
    - Evaluating recommendation quality
    - A/B test analysis
    """
    try:
        # Log interaction
        interaction_id = await db_manager.log_interaction(interaction)

        # Update user preferences in background
        if interaction.user_id:
            background_tasks.add_task(
                update_user_preferences_task,
                interaction,
            )

        return {
            "status": "success",
            "interaction_id": interaction_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {str(e)}")


async def update_user_preferences_task(interaction: UserInteraction):
    """Background task to update user preferences."""
    try:
        personalization_service: PersonalizationService = app.state.personalization_service

        # Fetch repo data from database
        repos = await db_manager.search_repositories(
            {"_id": interaction.repo_id}, limit=1
        )
        if not repos:
            repos = await db_manager.search_repositories(
                {"id": interaction.repo_id}, limit=1
            )

        if repos:
            await personalization_service.update_preferences_from_interaction(
                interaction, repos[0]
            )
    except Exception as e:
        logger.error("Failed to update preferences: %s", e)


# ===== User Preferences =====

@app.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get user preferences."""
    prefs = await db_manager.get_user_preferences(user_id)
    if not prefs:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return prefs


# ===== Metrics & Evaluation =====

@app.get("/metrics/{variant}")
async def get_metrics(variant: str):
    """Get latest evaluation metrics for a variant."""
    metrics = await db_manager.get_latest_metrics(variant)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"No metrics found for variant: {variant}")
    return metrics


@app.get("/ab-test")
async def get_active_ab_test():
    """Get currently active A/B test configuration."""
    ab_test = await db_manager.get_active_ab_test()
    if not ab_test:
        return {"status": "no_active_test"}
    return ab_test


# ===== Admin Endpoints =====

@app.post("/admin/cache/clear")
async def clear_cache():
    """Clear recommendation cache."""
    try:
        cleared = 0
        async for key in db_manager.redis_client.scan_iter(match="reco:*"):
            await db_manager.redis_client.delete(key)
            cleared += 1
        return {"status": "success", "message": f"Cleared {cleared} cache entries"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/admin/engines")
async def list_engines():
    """List all available recommendation engines."""
    engines = []
    for name, engine in app.state.engines.items():
        engines.append(engine.get_metadata())
    return {"engines": engines}


# ===== Model Management Endpoints =====

@app.get("/admin/models")
async def list_models(
    model_type: Optional[str] = None, 
    status: Optional[str] = None
):
    """List all registered models."""
    registry: ModelRegistryService = app.state.registry_service
    models = await registry.list_models(model_type, status)
    return {"models": models}


@app.post("/admin/models/reload")
async def reload_models(variant: str = "default"):
    """Reload active models from the registry."""
    try:
        await app.state.embedding_service.load_active_model(variant)
        await app.state.reranker_service.load_active_model(variant)
        return {
            "status": "success",
            "message": "Models reloaded",
            "active_embedding": app.state.embedding_service.current_model_id,
            "active_reranker": app.state.reranker_service.current_model_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")


@app.post("/admin/models/promote/{model_id}")
async def promote_model(model_id: str):
    """Promote a model to active status."""
    registry: ModelRegistryService = app.state.registry_service
    success = await registry.promote_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found or promotion failed")
    return {"status": "success", "message": f"Model {model_id} promoted to active"}
