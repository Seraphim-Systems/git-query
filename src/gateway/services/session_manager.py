"""Session management service using Redis."""

from typing import Optional
import secrets
from datetime import datetime
from pydantic import BaseModel
from redis.asyncio import Redis


class SessionData(BaseModel):
    """Session data model."""

    user_id: str
    created_at: datetime
    last_active: datetime
    ip_address: str
    user_agent: str


class SessionManager:
    """Manages user sessions in Redis."""

    def __init__(self, redis: Redis, ttl: int = 86400):
        """
        Initialize session manager.

        Args:
            redis: Redis client
            ttl: Session TTL in seconds (default: 24 hours)
        """
        self.redis = redis
        self.session_ttl = ttl

    async def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """
        Create a new session.

        Args:
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)

        session_data = SessionData(
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        await self.redis.setex(f"session:{session_id}", self.session_ttl, session_data.model_dump_json())

        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve and refresh session.

        Args:
            session_id: Session identifier

        Returns:
            Session data if valid, None if expired/invalid
        """
        key = f"session:{session_id}"
        data = await self.redis.get(key)

        if not data:
            return None

        session = SessionData.model_validate_json(data)

        # Update last_active timestamp and refresh TTL
        session.last_active = datetime.utcnow()
        await self.redis.setex(key, self.session_ttl, session.model_dump_json())

        return session

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session (logout).

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        result = await self.redis.delete(f"session:{session_id}")
        return result > 0

    async def get_user_sessions(self, user_id: str) -> list[str]:
        """
        Get all active sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of session IDs
        """
        session_ids = []
        pattern = "session:*"

        async for key in self.redis.scan_iter(match=pattern):
            data = await self.redis.get(key)
            if data:
                try:
                    session = SessionData.model_validate_json(data)
                    if session.user_id == user_id:
                        # Extract session_id from key "session:xxx"
                        session_id = key.split(":")[-1] if ":" in key else key
                        session_ids.append(session_id)
                except Exception:
                    continue

        return session_ids

    async def delete_all_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions deleted
        """
        session_ids = await self.get_user_sessions(user_id)
        if not session_ids:
            return 0

        keys = [f"session:{sid}" for sid in session_ids]
        return await self.redis.delete(*keys)
