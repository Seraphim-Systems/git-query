"""FastAPI application for the recommendation service."""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import time
from typing import Optional
from datetime import datetime, timezone
from typing import List, Optional

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
    LanguagePreferenceService,
)

logger = logging.getLogger(__name__)


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
    app.state.language_preference_service = LanguagePreferenceService()
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

    # Load active models in parallel
    await asyncio.gather(
        app.state.embedding_service.load_active_model(),
        app.state.reranker_service.load_active_model(),
    )

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
    """Redirect to frontend nginx server.

    In production: Uses SVC_NGINX_SERVER_NAME from GitHub secrets
    In development: Defaults to localhost:8080
    """
    nginx_server = os.getenv("SVC_NGINX_SERVER_NAME", "")
    if nginx_server:
        frontend_url = f"https://{nginx_server}"
    else:
        frontend_url = "http://localhost:8080"

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
        variant = await ab_test_service.get_variant_for_user(request.user_id, request.variant)

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
        logger.error("Recommendation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@app.post("/recommend/explain/{repo_id}")
async def explain_recommendation(repo_id: str, request: RecommendationRequest):
    """Explain why a repository was recommended."""
    try:
        # Get variant
        ab_test_service: ABTestService = app.state.ab_test_service
        variant = await ab_test_service.get_variant_for_user(request.user_id, request.variant)

        # Get engine and explanation
        engine = app.state.engines.get(variant, app.state.engines["baseline"])
        explanation = await engine.explain(repo_id, request)

        return explanation

    except Exception as e:
        logger.error("Explanation failed for repo %s: %s", repo_id, e, exc_info=True)
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

        # Update user preferences in background — pass service explicitly to
        # avoid a closure over app.state that could break on hot-reload.
        if interaction.user_id:
            background_tasks.add_task(
                update_user_preferences_task,
                interaction,
                app.state.personalization_service,
            )

        return {
            "status": "success",
            "interaction_id": interaction_id,
        }

    except Exception as e:
        logger.error("Failed to log interaction: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to log interaction: {str(e)}")


async def update_user_preferences_task(
    interaction: UserInteraction,
    personalization_service: PersonalizationService,
):
    """Background task to update user preferences."""
    try:
        # Fetch repo data — single round-trip using $or across both _id and id fields
        repo_map = await db_manager.get_repositories_by_repo_ids([interaction.repo_id])
        repos = list(repo_map.values())

        if repos:
            await personalization_service.update_preferences_from_interaction(interaction, repos[0])
    except Exception as e:
        logger.error("Failed to update preferences: %s", e, exc_info=True)


# ===== User Preferences =====


@app.get("/preferences/{user_id}")
async def get_user_preferences(user_id: str):
    """Get user preferences."""
    prefs = await db_manager.get_user_preferences(user_id)
    if not prefs:
        raise HTTPException(status_code=404, detail="User preferences not found")
    return prefs


# ===== Language Preference Endpoints =====


@app.post("/preferences/{user_id}/languages", status_code=200)
async def set_language_preferences(user_id: str, languages: List[str]):
    """
    Set the user's explicitly preferred programming languages.

    Pass a list of language names (case-insensitive).  These languages will
    receive a score boost and always appear near the top of personalization
    signals even before the user has interacted with repos in that language.

    Example body: ["Python", "Rust", "TypeScript"]
    """
    try:
        svc: LanguagePreferenceService = app.state.language_preference_service
        prefs = await svc.set_explicit_languages(user_id, languages)
        return {
            "status": "success",
            "user_id": user_id,
            "explicit_languages": prefs.explicit_languages,
            "language_scores": prefs.language_preferences,
        }
    except Exception as e:
        logger.error("Failed to set language preferences for %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preferences/{user_id}/languages")
async def get_language_preferences(user_id: str, top_n: int = 5):
    """
    Get a user's top preferred programming languages, ranked by preference score.

    Each entry includes whether the language was explicitly declared by the user
    or inferred from their interaction history.
    """
    svc: LanguagePreferenceService = app.state.language_preference_service
    languages = await svc.get_top_languages(user_id, top_n=top_n)
    if not languages:
        raise HTTPException(
            status_code=404,
            detail=f"No language preferences found for user '{user_id}'",
        )
    return {"user_id": user_id, "languages": languages}


@app.delete("/preferences/{user_id}/languages/{language}", status_code=200)
async def remove_language_preference(user_id: str, language: str):
    """
    Remove a specific language from a user's preferences.

    Removes both the explicit declaration and the learned score so the language
    will no longer influence recommendations.
    """
    try:
        svc: LanguagePreferenceService = app.state.language_preference_service
        prefs = await svc.remove_language(user_id, language)
        return {
            "status": "success",
            "user_id": user_id,
            "removed_language": language.lower(),
            "remaining_explicit": prefs.explicit_languages,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to remove language preference: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
        keys = [key async for key in db_manager.redis_client.scan_iter(match="reco:*")]
        if keys:
            await db_manager.redis_client.delete(*keys)
        return {"status": "success", "message": f"Cleared {len(keys)} cache entries"}
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
async def list_models(model_type: Optional[str] = None, status: Optional[str] = None):
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


@app.post("/admin/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    variant: str = Form("default"),
    metrics: str = Form("{}"),
):
    """Upload a trained model file and register it in the model registry.

    Called by the local training pipeline to push a model artifact to the server
    before promoting it. Saves the file to /app/models/ and creates a candidate
    entry in MongoDB so that /admin/models/promote/{model_id} can find it.
    """
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from .models import ModelMetadata

    models_dir = Path("/app/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or f"{model_id}.pkl"
    model_path = models_dir / filename

    content = await file.read()
    model_path.write_bytes(content)
    logger.info("Uploaded model file saved to %s (%d bytes)", model_path, len(content))

    try:
        parsed_metrics = json.loads(metrics)
    except Exception:
        parsed_metrics = {}

    metadata = ModelMetadata(
        model_id=model_id,
        model_type="reranker",
        variant=variant,
        version="1.0.0",
        path=filename,
        hyperparameters={},
        metrics={k: float(v) for k, v in parsed_metrics.items() if isinstance(v, (int, float))},
        trained_at=datetime.now(UTC),
        is_active=False,
        status="candidate",
    )
    registry: ModelRegistryService = app.state.registry_service
    await registry.register_model(metadata)
    logger.info("Registered uploaded model %s in registry", model_id)

    return {"status": "success", "model_id": model_id, "path": str(model_path)}
