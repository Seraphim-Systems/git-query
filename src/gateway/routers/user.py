"""User router."""

from fastapi import APIRouter, Depends, Request

from src.gateway.middleware.session import get_current_user
from src.gateway.services.user_service import UserPreferences

router = APIRouter()


@router.get("/preferences", response_model=UserPreferences)
async def get_preferences(request: Request, user_id: str = Depends(get_current_user)):
    """Get current user preferences."""
    user_service = request.app.state.user_service
    preferences = await user_service.get_user_preferences(user_id)
    return preferences


@router.put("/preferences", response_model=UserPreferences)
async def update_preferences(
    request: Request,
    preferences: UserPreferences,
    user_id: str = Depends(get_current_user),
):
    """Update user preferences."""
    user_service = request.app.state.user_service
    updated = await user_service.update_preferences(user_id, preferences.model_dump())
    return updated


@router.get("/interactions")
async def get_interactions(
    request: Request, user_id: str = Depends(get_current_user), limit: int = 100
):
    """Get user interaction history."""
    user_service = request.app.state.user_service
    interactions = await user_service.get_interaction_history(user_id, limit)
    return {"interactions": interactions, "count": len(interactions)}


@router.get("/profile")
async def get_profile(request: Request, user_id: str = Depends(get_current_user)):
    """Get user profile."""
    user_service = request.app.state.user_service
    user = await user_service.get_user(user_id)

    if not user:
        return {"error": "User not found"}

    # Remove sensitive data
    user.pop("password_hash", None)
    user.pop("_id", None)

    return user
