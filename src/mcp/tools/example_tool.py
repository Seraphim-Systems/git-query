"""Example tools for the MCP server."""

from typing import Any


async def get_recommendation(user_id: str, category: str = "general") -> dict[str, Any]:
    """
    Get personalized recommendations for a user.

    Args:
        user_id: The user identifier
        category: The category of recommendations (default: general)

    Returns:
        Dictionary with recommendations
    """
    # TODO: Implement actual recommendation logic
    return {
        "user_id": user_id,
        "category": category,
        "recommendations": [
            {"id": "1", "name": "Item 1", "score": 0.95},
            {"id": "2", "name": "Item 2", "score": 0.87},
            {"id": "3", "name": "Item 3", "score": 0.82},
        ],
        "generated_at": "2026-02-04T00:00:00Z",
    }


async def search_items(query: str, limit: int = 10) -> dict[str, Any]:
    """
    Search for items in the system.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    # TODO: Implement actual search logic
    return {
        "query": query,
        "limit": limit,
        "results": [
            {"id": "1", "name": f"Result for '{query}' #1", "relevance": 0.98},
            {"id": "2", "name": f"Result for '{query}' #2", "relevance": 0.85},
        ],
        "total_count": 2,
    }


# Tool definitions for MCP protocol
TOOL_DEFINITIONS = [
    {
        "name": "get_recommendation",
        "description": "Get personalized recommendations for a user",
        "parameters": [
            {
                "name": "user_id",
                "type": "string",
                "description": "The user identifier",
                "required": True,
            },
            {
                "name": "category",
                "type": "string",
                "description": "The category of recommendations",
                "required": False,
                "default": "general",
            },
        ],
    },
    {
        "name": "search_items",
        "description": "Search for items in the system",
        "parameters": [
            {
                "name": "query",
                "type": "string",
                "description": "Search query string",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of results to return",
                "required": False,
                "default": 10,
            },
        ],
    },
]
