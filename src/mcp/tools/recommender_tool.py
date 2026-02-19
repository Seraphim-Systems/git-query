"""Recommender tools for the MCP server.

These tools act as a thin wrapper around the recommender service (port 8095),
routing requests from the LLM/client through MCP and returning the results.
"""

import logging
from typing import Any, Optional

import httpx

from src.mcp.config import settings

logger = logging.getLogger(__name__)


def _get_recommender_client() -> httpx.AsyncClient:
    """Return an async HTTP client pointed at the recommender service."""
    return httpx.AsyncClient(base_url=settings.recommender_url, timeout=30.0)


async def recommend_repositories(
    query: str,
    user_id: Optional[str] = None,
    language: Optional[str] = None,
    min_stars: Optional[int] = None,
    license: Optional[str] = None,
    max_age_days: Optional[int] = None,
    top_k: int = 10,
    enable_personalization: bool = True,
    variant: Optional[str] = None,
) -> dict[str, Any]:
    """
    Find GitHub repositories that match a natural-language query.

    Calls the recommender service and returns ranked repository results
    with relevance scores and metadata.

    Args:
        query: Natural-language description of what the user is looking for.
        user_id: Optional user identifier — enables personalised results.
        language: Filter results to a specific programming language (e.g. "Python").
        min_stars: Only return repositories with at least this many stars.
        license: Filter by licence type (e.g. "MIT").
        max_age_days: Exclude repos not updated within this many days.
        top_k: Number of results to return (1–50, default 10).
        enable_personalization: Apply user-preference personalisation when available.
        variant: A/B test variant to use ("baseline", "hybrid", "personalized").

    Returns:
        Dictionary with ranked repository results and metadata.
    """
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "enable_personalization": enable_personalization,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    if language is not None:
        payload["language"] = language
    if min_stars is not None:
        payload["min_stars"] = min_stars
    if license is not None:
        payload["license"] = license
    if max_age_days is not None:
        payload["max_age_days"] = max_age_days
    if variant is not None:
        payload["variant"] = variant

    async with _get_recommender_client() as client:
        try:
            response = await client.post("/recommend", json=payload)
            response.raise_for_status()
            data = response.json()

            # Flatten results into a clean, LLM-friendly structure
            results = []
            for repo in data.get("results", []):
                results.append(
                    {
                        "rank": repo.get("rank"),
                        "name": repo.get("full_name") or repo.get("name"),
                        "description": repo.get("description"),
                        "url": repo.get("url"),
                        "language": repo.get("language"),
                        "stars": repo.get("stars"),
                        "forks": repo.get("forks"),
                        "license": repo.get("license"),
                        "score": round(repo.get("score", 0.0), 4),
                    }
                )

            return {
                "query": data.get("query"),
                "results": results,
                "total_candidates": data.get("total_candidates"),
                "processing_time_ms": data.get("processing_time_ms"),
                "variant": data.get("variant"),
                "personalized": data.get("personalized"),
                "filters_applied": data.get("filters_applied"),
            }

        except httpx.HTTPStatusError as e:
            logger.error("Recommender returned HTTP error: %s", e)
            return {"error": f"Recommender service error: {e.response.status_code}", "detail": e.response.text}
        except httpx.RequestError as e:
            logger.error("Could not reach recommender service: %s", e)
            return {"error": "Recommender service is unreachable", "detail": str(e)}


async def log_repository_interaction(
    user_id: str,
    query: str,
    repo_id: str,
    interaction_type: str,
    position_in_results: Optional[int] = None,
    variant: str = "baseline",
) -> dict[str, Any]:
    """
    Log a user interaction with a recommended repository.

    This feeds back into the personalisation engine so future recommendations
    improve over time.

    Args:
        user_id: The user who interacted.
        query: The original search query that produced the result.
        repo_id: The repository ID that was interacted with.
        interaction_type: One of: click, save, dismiss, thumbs_up, thumbs_down, view.
        position_in_results: 0-based position of the repo in the result list.
        variant: Which recommendation variant was shown.

    Returns:
        Confirmation of the logged interaction.
    """
    valid_types = {"click", "save", "dismiss", "thumbs_up", "thumbs_down", "view"}
    if interaction_type not in valid_types:
        return {
            "error": f"Invalid interaction_type '{interaction_type}'. Must be one of: {', '.join(sorted(valid_types))}"
        }

    payload: dict[str, Any] = {
        "user_id": user_id,
        "query": query,
        "repo_id": repo_id,
        "interaction_type": interaction_type,
        "variant": variant,
    }
    if position_in_results is not None:
        payload["position_in_results"] = position_in_results

    async with _get_recommender_client() as client:
        try:
            response = await client.post("/interaction", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("Recommender returned HTTP error while logging interaction: %s", e)
            return {"error": f"Recommender service error: {e.response.status_code}", "detail": e.response.text}
        except httpx.RequestError as e:
            logger.error("Could not reach recommender service: %s", e)
            return {"error": "Recommender service is unreachable", "detail": str(e)}


async def get_user_preferences(user_id: str) -> dict[str, Any]:
    """
    Retrieve the learned preference profile for a user.

    Shows which languages and topics the user has shown interest in,
    based on their interaction history.

    Args:
        user_id: The user whose preferences to fetch.

    Returns:
        Dictionary with language and topic preference scores.
    """
    async with _get_recommender_client() as client:
        try:
            response = await client.get(f"/preferences/{user_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"user_id": user_id, "message": "No preference profile found for this user yet."}
            logger.error("Recommender returned HTTP error fetching preferences: %s", e)
            return {"error": f"Recommender service error: {e.response.status_code}", "detail": e.response.text}
        except httpx.RequestError as e:
            logger.error("Could not reach recommender service: %s", e)
            return {"error": "Recommender service is unreachable", "detail": str(e)}


# ---------------------------------------------------------------------------
# Tool definitions (MCP protocol metadata)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "recommend_repositories",
        "description": (
            "Find GitHub repositories that best match a natural-language query. "
            "Returns a ranked list of repositories with scores, metadata and optional "
            "personalisation for a specific user."
        ),
        "parameters": [
            {"name": "query", "type": "string", "description": "Natural-language description of what to find", "required": True},
            {"name": "user_id", "type": "string", "description": "User identifier for personalised results", "required": False},
            {"name": "language", "type": "string", "description": "Filter by programming language (e.g. Python)", "required": False},
            {"name": "min_stars", "type": "integer", "description": "Minimum number of GitHub stars", "required": False},
            {"name": "license", "type": "string", "description": "Filter by licence (e.g. MIT)", "required": False},
            {"name": "max_age_days", "type": "integer", "description": "Exclude repos not updated in this many days", "required": False},
            {"name": "top_k", "type": "integer", "description": "Number of results (1–50, default 10)", "required": False, "default": 10},
            {"name": "enable_personalization", "type": "boolean", "description": "Apply user-preference personalisation", "required": False, "default": True},
            {"name": "variant", "type": "string", "description": "A/B variant: baseline | hybrid | personalized", "required": False},
        ],
    },
    {
        "name": "log_repository_interaction",
        "description": (
            "Log a user interaction (click, save, thumbs up/down, etc.) with a recommended "
            "repository. This feedback improves future personalised recommendations."
        ),
        "parameters": [
            {"name": "user_id", "type": "string", "description": "The user who interacted", "required": True},
            {"name": "query", "type": "string", "description": "The original search query", "required": True},
            {"name": "repo_id", "type": "string", "description": "The repository ID", "required": True},
            {"name": "interaction_type", "type": "string", "description": "One of: click, save, dismiss, thumbs_up, thumbs_down, view", "required": True},
            {"name": "position_in_results", "type": "integer", "description": "0-based position of repo in result list", "required": False},
            {"name": "variant", "type": "string", "description": "Which recommendation variant was shown", "required": False, "default": "baseline"},
        ],
    },
    {
        "name": "get_user_preferences",
        "description": (
            "Retrieve a user's learned preference profile — which languages and topics "
            "they tend to be interested in, derived from their interaction history."
        ),
        "parameters": [
            {"name": "user_id", "type": "string", "description": "The user whose preferences to fetch", "required": True},
        ],
    },
]

