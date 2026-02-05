"""Chatbot implementation using Pydantic AI."""

import logging
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from src.client.config import settings
from src.client.mcp_client import mcp_client

logger = logging.getLogger(__name__)


# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant for a recommendation system.
You can help users find recommendations and search for items.

You have access to tools that can:
- Get personalized recommendations for users
- Search for items in the system

When a user asks for recommendations or searches, use the appropriate tools to help them.
Be friendly, concise, and helpful in your responses."""


class ChatbotDependencies:
    """Dependencies for the chatbot agent."""

    def __init__(self, user_id: str = None):
        self.user_id = user_id
        self.tool_calls: list[dict[str, Any]] = []


# Initialize the Pydantic AI agent
model = OpenAIModel(settings.model_name, api_key=settings.openai_api_key)
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=ChatbotDependencies,
    retries=2,
)


@agent.tool
async def get_recommendation(
    ctx: RunContext[ChatbotDependencies], user_id: str, category: str = "general"
) -> dict:
    """
    Get personalized recommendations for a user.

    Args:
        ctx: The run context
        user_id: The user identifier
        category: The category of recommendations
    """
    logger.info(
        "Tool called: get_recommendation(user_id=%s, category=%s)", user_id, category
    )

    result = await mcp_client.execute_tool(
        "get_recommendation", {"user_id": user_id, "category": category}
    )

    # Track tool call
    ctx.deps.tool_calls.append(
        {
            "tool": "get_recommendation",
            "parameters": {"user_id": user_id, "category": category},
            "result": result,
        }
    )

    return result


@agent.tool
async def search_items(
    ctx: RunContext[ChatbotDependencies], query: str, limit: int = 10
) -> dict:
    """
    Search for items in the system.

    Args:
        ctx: The run context
        query: Search query string
        limit: Maximum number of results
    """
    logger.info("Tool called: search_items(query=%s, limit=%d)", query, limit)

    result = await mcp_client.execute_tool(
        "search_items", {"query": query, "limit": limit}
    )

    # Track tool call
    ctx.deps.tool_calls.append(
        {
            "tool": "search_items",
            "parameters": {"query": query, "limit": limit},
            "result": result,
        }
    )

    return result


async def chat(message: str, user_id: str = None) -> tuple[str, list[dict]]:
    """
    Process a chat message and return a response.

    Args:
        message: User's message
        user_id: Optional user ID

    Returns:
        Tuple of (response text, list of tool calls)
    """
    deps = ChatbotDependencies(user_id=user_id)

    try:
        result = await agent.run(message, deps=deps)
        return result.data, deps.tool_calls
    except Exception as e:
        logger.error("Error processing chat message: %s", e)
        return f"I apologize, but I encountered an error: {str(e)}", deps.tool_calls
