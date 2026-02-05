"""Pydantic models for ChatbotClient."""
from typing import Optional, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message."""
    role: str
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ChatSession(BaseModel):
    """Chat session with history."""
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    user_id: Optional[str] = None

