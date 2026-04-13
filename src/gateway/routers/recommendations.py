"""Recommendations router - proxies to recommender service."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional
import httpx

from src.gateway.middleware.session import get_current_user, get_user_preferences

router = APIRouter()


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    recommendations: list
    user_id: str
    personalized: bool


class RecommendationRequest(BaseModel):
    """Recommendation request model."""

    query: str = ""
    top_k: int = 10
    enable_personalization: bool = True
    language: Optional[str] = None
    min_stars: Optional[int] = None
    license: Optional[str] = None
    max_age_days: Optional[int] = None
    variant: Optional[str] = None


@router.post("/", response_model=RecommendationResponse)
async def post_recommendations(
    request: Request,
    rec_request: RecommendationRequest,
    user_id: str = Depends(get_current_user),
    preferences=Depends(get_user_preferences),
):
    """
    Get recommendations via POST - simple proxy to recommender service.
    Frontend sends search queries here.
    """
    from src.shared.config import settings

    # Forward request to recommender service
    payload = {
        "user_id": user_id,
        "query": rec_request.query,
        "preferences": preferences.model_dump(),
        "top_k": rec_request.top_k,
        "enable_personalization": rec_request.enable_personalization,
    }

    # Add optional filters if provided
    if rec_request.language:
        payload["language"] = rec_request.language
    if rec_request.min_stars:
        payload["min_stars"] = rec_request.min_stars
    if rec_request.license:
        payload["license"] = rec_request.license
    if rec_request.max_age_days:
        payload["max_age_days"] = rec_request.max_age_days
    if rec_request.variant:
        payload["variant"] = rec_request.variant

    # Call recommender service
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{settings.recommender_url}/recommend", json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            # Fallback to empty results on error
            result = {"results": [], "items": []}

    # Return standardized response
    items = result.get("results", result.get("items", []))
    return RecommendationResponse(
        recommendations=items,
        user_id=user_id,
        personalized=result.get("personalized", True),
    )


@router.get("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    user_id: str = Depends(get_current_user),
    preferences=Depends(get_user_preferences),
    limit: int = 10,
    query: str = "",
):
    """
    Get recommendations via GET - simple proxy to recommender service.
    """
    from src.shared.config import settings

    payload = {
        "user_id": user_id,
        "query": query,
        "preferences": preferences.model_dump(),
        "top_k": limit,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{settings.recommender_url}/recommend", json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError:
            result = {"results": [], "items": []}

    items = result.get("results", result.get("items", []))
    return RecommendationResponse(
        recommendations=items,
        user_id=user_id,
        personalized=True,
    )


class FeedbackRequest(BaseModel):
    """Feedback request model."""

    repo_id: str
    action: str  # click, save, dismiss, thumbs_up, thumbs_down, view, star, clone
    query: str = ""
    variant: str = "hybrid"
    position_in_results: Optional[int] = None
    metadata: dict = Field(default_factory=dict)


@router.post("/feedback")
async def record_feedback(
    request: Request,
    feedback: FeedbackRequest,
    user_id: str = Depends(get_current_user),
):
    """Record user feedback on recommendations."""
    user_service = request.app.state.user_service

    await user_service.record_interaction(
        user_id=user_id,
        repo_id=feedback.repo_id,
        action=feedback.action,
        query=feedback.query,
        variant=feedback.variant,
        position_in_results=feedback.position_in_results,
        metadata=feedback.metadata,
    )

    return {
        "status": "recorded",
        "repo_id": feedback.repo_id,
        "action": feedback.action,
        "variant": feedback.variant,
    }
