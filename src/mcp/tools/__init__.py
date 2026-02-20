"""Tool registry for MCP server."""
from typing import Callable, Any
from .example_tool import get_recommendation, search_items
from .recommender_tool import (
    recommend_repositories,
    log_repository_interaction,
    get_user_preferences,
)

# Tool registry - maps tool names to their implementations
TOOL_REGISTRY: dict[str, Callable] = {
    "get_recommendation": get_recommendation,
    "search_items": search_items,
    # Recommender tools
    "recommend_repositories": recommend_repositories,
    "log_repository_interaction": log_repository_interaction,
    "get_user_preferences": get_user_preferences,
}


def get_tool(tool_name: str) -> Callable | None:
    """Get a tool by name."""
    return TOOL_REGISTRY.get(tool_name)


def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their metadata."""
    from .example_tool import TOOL_DEFINITIONS as EXAMPLE_DEFS
    from .recommender_tool import TOOL_DEFINITIONS as RECOMMENDER_DEFS

    return EXAMPLE_DEFS + RECOMMENDER_DEFS


__all__ = ["get_tool", "list_tools", "TOOL_REGISTRY"]
