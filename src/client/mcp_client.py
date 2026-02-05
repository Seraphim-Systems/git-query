"""MCP Client for communicating with MCP server."""

import logging
from typing import Any
import httpx

from src.client.config import settings

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for MCP server communication."""

    def __init__(self, base_url: str = None):
        """Initialize MCP client."""
        self.base_url = base_url or settings.mcp_server_url
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools from MCP server."""
        try:
            response = await self.client.post("/tools/list")
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list tools from MCP server: {e}")
            return []

    async def execute_tool(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a tool on the MCP server."""
        try:
            response = await self.client.post(
                "/tools/execute",
                json={"tool_name": tool_name, "parameters": parameters},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def health_check(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"MCP server health check failed: {e}")
            return False

    async def close(self):
        """Close the client connection."""
        await self.client.aclose()


# Global MCP client instance
mcp_client = MCPClient()
