"""User router."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from src.gateway.middleware.session import get_current_user
from src.gateway.services.user_service import UserPreferences

router = APIRouter()


class UserChatsPayload(BaseModel):
    """Bulk chat sessions payload."""

    chats: list[dict[str, Any]] = Field(default_factory=list)


class UserSavedReposPayload(BaseModel):
    """Bulk saved repositories payload."""

    repos: list[dict[str, Any]] = Field(default_factory=list)


class UserFoldersPayload(BaseModel):
    """Bulk folders payload."""

    folders: list[dict[str, Any]] = Field(default_factory=list)


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


@router.get("/chats")
async def get_chats(
    request: Request, user_id: str = Depends(get_current_user), limit: int = 100
):
    """Get persisted chat sessions."""
    user_service = request.app.state.user_service
    chats = await user_service.get_user_chats(user_id, limit)
    return {"chats": chats, "count": len(chats)}


@router.put("/chats")
async def put_chats(
    request: Request,
    payload: UserChatsPayload,
    user_id: str = Depends(get_current_user),
):
    """Replace persisted chat sessions."""
    user_service = request.app.state.user_service
    chats = await user_service.replace_user_chats(user_id, payload.chats)
    return {"chats": chats, "count": len(chats)}


@router.get("/saved-repos")
async def get_saved_repos(
    request: Request, user_id: str = Depends(get_current_user), limit: int = 200
):
    """Get saved repositories."""
    user_service = request.app.state.user_service
    repos = await user_service.get_saved_repos(user_id, limit)
    return {"repos": repos, "count": len(repos)}


@router.put("/saved-repos")
async def put_saved_repos(
    request: Request,
    payload: UserSavedReposPayload,
    user_id: str = Depends(get_current_user),
):
    """Replace saved repositories."""
    user_service = request.app.state.user_service
    repos = await user_service.replace_saved_repos(user_id, payload.repos)
    return {"repos": repos, "count": len(repos)}


@router.get("/folders")
async def get_folders(
    request: Request, user_id: str = Depends(get_current_user), limit: int = 200
):
    """Get saved folders."""
    user_service = request.app.state.user_service
    folders = await user_service.get_user_folders(user_id, limit)
    return {"folders": folders, "count": len(folders)}


@router.put("/folders")
async def put_folders(
    request: Request,
    payload: UserFoldersPayload,
    user_id: str = Depends(get_current_user),
):
    """Replace saved folders."""
    user_service = request.app.state.user_service
    folders = await user_service.replace_user_folders(user_id, payload.folders)
    return {"folders": folders, "count": len(folders)}


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
