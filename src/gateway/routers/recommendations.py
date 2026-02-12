"""Recommendations router."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
import httpx

from src.gateway.middleware.session import get_current_user, get_user_preferences

router = APIRouter()


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    recommendations: list
    user_id: str
    personalized: bool


@router.get("/", response_model=RecommendationResponse)
async def get_recommendations(
    request: Request,
    user_id: str = Depends(get_current_user),
    preferences=Depends(get_user_preferences),
    limit: int = 10,
    context: str = None,
):
    """
    Get personalized recommendations.

    User context is automatically injected from session.
    Preferences are used to filter/rank results.
    """
    from src.shared.config import settings

    # Prepare request payload with user context
    payload = {
        "user_id": user_id,
        "preferences": preferences.model_dump(),
        "context": context,
        "limit": limit,
    }

    # Call internal Recommender API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.recommender_url}/recommend", json=payload, timeout=30.0
            )
            response.raise_for_status()
            recommendations = response.json()
        except httpx.HTTPError as e:
            # Fallback to empty recommendations
            recommendations = {"items": [], "error": str(e)}

    # Record implicit signal (viewed recommendations)
    user_service = request.app.state.user_service
    await user_service.record_interaction(
        user_id=user_id,
        repo_id="recommendations_viewed",
        action="view",
        metadata={"limit": limit, "context": context},
    )

    return RecommendationResponse(
        recommendations=recommendations.get("items", []),
        user_id=user_id,
        personalized=True,
    )


class FeedbackRequest(BaseModel):
    """Feedback request model."""

    repo_id: str
    action: str  # click, star, clone, dismiss


@router.post("/feedback")
async def record_feedback(
    request: Request,
    feedback: FeedbackRequest,
    user_id: str = Depends(get_current_user),
):
    """Record user feedback on recommendations."""
    user_service = request.app.state.user_service

    await user_service.record_interaction(
        user_id=user_id, repo_id=feedback.repo_id, action=feedback.action
    )

    # TODO: Trigger async task to update user embedding
    # from src.processing.tasks.embedding import update_user_embedding
    # update_user_embedding.delay(user_id)

    return {
        "status": "recorded",
        "repo_id": feedback.repo_id,
        "action": feedback.action,
    }
