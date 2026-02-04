"""Tool registry for MCP server."""
from typing import Callable, Any
from .example_tool import get_recommendation, search_items

# Tool registry - maps tool names to their implementations
TOOL_REGISTRY: dict[str, Callable] = {
    "get_recommendation": get_recommendation,
    "search_items": search_items,
}


def get_tool(tool_name: str) -> Callable | None:
    """Get a tool by name."""
    return TOOL_REGISTRY.get(tool_name)


def list_tools() -> list[dict[str, Any]]:
    """List all available tools with their metadata."""
    from .example_tool import TOOL_DEFINITIONS
    return TOOL_DEFINITIONS


__all__ = ["get_tool", "list_tools", "TOOL_REGISTRY"]

