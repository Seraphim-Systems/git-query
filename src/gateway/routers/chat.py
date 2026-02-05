"""Chat router - proxies to MCP server."""

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
import httpx

from src.gateway.middleware.session import get_current_user, get_user_preferences

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str
    context: dict = {}


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    user_id: str


@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    user_id: str = Depends(get_current_user),
    preferences=Depends(get_user_preferences),
):
    """
    Chat endpoint - proxies to MCP server with user context.
    """
    from src.gateway.config import settings

    # Enrich request with user context
    payload = {
        "message": chat_request.message,
        "user_id": user_id,
        "preferences": preferences.model_dump(),
        "context": chat_request.context,
    }

    # Call internal MCP server
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{settings.mcp_server_url}/chat", json=payload, timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            result = {"response": f"Error communicating with chat service: {str(e)}"}

    return ChatResponse(response=result.get("response", "No response"), user_id=user_id)
