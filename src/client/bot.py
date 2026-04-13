"""Chatbot implementation using Pydantic AI."""

import logging
from typing import Any, Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.client.config import settings
from src.client.mcp_client import mcp_client

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful AI assistant for Git-Query, a GitHub repository recommendation system.

You help users discover relevant GitHub repositories based on natural language queries.

You have access to the following tools:
- recommend_repositories: Find GitHub repos matching a query, with optional filters (language, stars, licence, etc.)
- log_repository_interaction: Record when a user clicks, saves, or rates a repository
- get_user_preferences: View a user's learned language/topic preferences
- query_repository_data: Run read-only MongoDB queries for repository data
- explain_repository: Explain a repository from URL/full_name using DB + GitHub metadata
- get_recommendation: Legacy recommendation tool
- search_items: Generic item search

When a user asks to find, search, or recommend repositories, use recommend_repositories.
Always present results clearly with the repo name, description, stars and URL.
Be friendly, concise, and helpful."""


class ChatbotDependencies:
    """Dependencies for the chatbot agent."""

    def __init__(self, user_id: str = None):
        self.user_id = user_id
        self.tool_calls: list[dict[str, Any]] = []


# Initialize the Pydantic AI agent
model = OpenAIModel(
    settings.model_name,
    provider=OpenAIProvider(api_key=settings.resolved_openai_api_key),
)
agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=ChatbotDependencies,
    retries=2,
)


@agent.tool
async def recommend_repositories(
    ctx: RunContext[ChatbotDependencies],
    query: str,
    language: Optional[str] = None,
    min_stars: Optional[int] = None,
    repo_license: Optional[str] = None,
    max_age_days: Optional[int] = None,
    top_k: int = 10,
    enable_personalization: bool = True,
    variant: Optional[str] = None,
) -> dict:
    """
    Find GitHub repositories that match a natural-language query.

    Args:
        ctx: The run context
        query: Natural-language description of what to find
        language: Filter by programming language (e.g. Python)
        min_stars: Minimum number of GitHub stars
        repo_license: Filter by licence type (e.g. MIT)
        max_age_days: Exclude repos not updated in this many days
        top_k: Number of results to return (1-50)
        enable_personalization: Apply user-preference personalisation
        variant: A/B variant: baseline | hybrid | personalized
    """
    user_id = ctx.deps.user_id
    params: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "enable_personalization": enable_personalization,
    }
    if user_id:
        params["user_id"] = user_id
    if language:
        params["language"] = language
    if min_stars is not None:
        params["min_stars"] = min_stars
    if repo_license:
        params["license"] = repo_license
    if max_age_days is not None:
        params["max_age_days"] = max_age_days
    if variant:
        params["variant"] = variant

    logger.info("Tool called: recommend_repositories(query=%s)", query)
    result = await mcp_client.execute_tool("recommend_repositories", params)
    ctx.deps.tool_calls.append({"tool": "recommend_repositories", "parameters": params, "result": result})
    return result


@agent.tool
async def log_repository_interaction(
    ctx: RunContext[ChatbotDependencies],
    query: str,
    repo_id: str,
    interaction_type: str,
    position_in_results: Optional[int] = None,
    variant: str = "baseline",
) -> dict:
    """
    Log a user interaction with a recommended repository (click, save, thumbs_up, thumbs_down, view, dismiss).

    Args:
        ctx: The run context
        query: The original search query
        repo_id: The repository ID that was interacted with
        interaction_type: One of: click, save, dismiss, thumbs_up, thumbs_down, view
        position_in_results: 0-based position in the result list
        variant: Which recommendation variant was shown
    """
    user_id = ctx.deps.user_id or "anonymous"
    params: dict[str, Any] = {
        "user_id": user_id,
        "query": query,
        "repo_id": repo_id,
        "interaction_type": interaction_type,
        "variant": variant,
    }
    if position_in_results is not None:
        params["position_in_results"] = position_in_results

    logger.info(
        "Tool called: log_repository_interaction(repo_id=%s, type=%s)",
        repo_id,
        interaction_type,
    )
    result = await mcp_client.execute_tool("log_repository_interaction", params)
    ctx.deps.tool_calls.append({"tool": "log_repository_interaction", "parameters": params, "result": result})
    return result


@agent.tool
async def get_user_preferences(ctx: RunContext[ChatbotDependencies]) -> dict:
    """
    Retrieve the current user's learned preference profile (favourite languages, topics).
    """
    user_id = ctx.deps.user_id
    if not user_id:
        return {"message": "No user_id provided — cannot fetch preferences."}

    logger.info("Tool called: get_user_preferences(user_id=%s)", user_id)
    result = await mcp_client.execute_tool("get_user_preferences", {"user_id": user_id})
    ctx.deps.tool_calls.append(
        {
            "tool": "get_user_preferences",
            "parameters": {"user_id": user_id},
            "result": result,
        }
    )
    return result


@agent.tool
async def query_repository_data(
    ctx: RunContext[ChatbotDependencies],
    collection: str,
    criteria: Optional[dict[str, Any]] = None,
    projection: Optional[list[str]] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    limit: int = 20,
    skip: int = 0,
    database: Optional[str] = None,
) -> dict:
    """Run a read-only MongoDB query for repository-related data."""
    params: dict[str, Any] = {
        "collection": collection,
        "sort_order": sort_order,
        "limit": limit,
        "skip": skip,
    }
    if criteria is not None:
        params["filters"] = criteria
    if projection is not None:
        params["projection"] = projection
    if sort_by is not None:
        params["sort_by"] = sort_by
    if database is not None:
        params["database"] = database

    logger.info("Tool called: query_repository_data(collection=%s)", collection)
    result = await mcp_client.execute_tool("query_repository_data", params)
    ctx.deps.tool_calls.append({"tool": "query_repository_data", "parameters": params, "result": result})
    return result


@agent.tool
async def explain_repository(
    ctx: RunContext[ChatbotDependencies],
    repo_url: Optional[str] = None,
    full_name: Optional[str] = None,
    include_database_context: bool = True,
    include_readme_excerpt: bool = True,
) -> dict:
    """Explain what a repository is about from DB and GitHub metadata."""
    params: dict[str, Any] = {
        "include_database_context": include_database_context,
        "include_readme_excerpt": include_readme_excerpt,
    }
    if repo_url is not None:
        params["repo_url"] = repo_url
    if full_name is not None:
        params["full_name"] = full_name

    logger.info("Tool called: explain_repository(repo=%s)", full_name or repo_url)
    result = await mcp_client.execute_tool("explain_repository", params)
    ctx.deps.tool_calls.append({"tool": "explain_repository", "parameters": params, "result": result})
    return result


@agent.tool
async def get_recommendation(ctx: RunContext[ChatbotDependencies], user_id: str, category: str = "general") -> dict:
    """Get personalized recommendations for a user (legacy tool)."""
    logger.info("Tool called: get_recommendation(user_id=%s, category=%s)", user_id, category)
    result = await mcp_client.execute_tool("get_recommendation", {"user_id": user_id, "category": category})
    ctx.deps.tool_calls.append(
        {
            "tool": "get_recommendation",
            "parameters": {"user_id": user_id, "category": category},
            "result": result,
        }
    )
    return result


@agent.tool
async def search_items(ctx: RunContext[ChatbotDependencies], query: str, limit: int = 10) -> dict:
    """Search for items in the system (legacy tool)."""
    logger.info("Tool called: search_items(query=%s, limit=%d)", query, limit)
    result = await mcp_client.execute_tool("search_items", {"query": query, "limit": limit})
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
        user_id: Optional user ID for personalisation

    Returns:
        Tuple of (response text, list of tool calls made)
    """
    deps = ChatbotDependencies(user_id=user_id)
    result = await agent.run(message, deps=deps)
    return result.output, deps.tool_calls
