"""Session middleware for request authentication."""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from src.gateway.middleware.shared import PUBLIC_PATHS

logger = logging.getLogger(__name__)


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware to validate sessions and inject user context."""

    async def dispatch(self, request: Request, call_next):
        """Process request and validate session."""

        # Skip auth for public endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # If API key middleware has already validated an API key for a
        # backend service mounted under `/api/...`, skip user session checks
        # because these requests are authenticated by service keys.
        if request.url.path.startswith("/api/") and getattr(
            request.state, "api_key", None
        ):
            return await call_next(request)

        # Extract session from cookie or Authorization header
        session_id = request.cookies.get("session_id")

        if not session_id:
            # Try Authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_id = auth_header.split(" ")[1]

        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No session found. Please login.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate session
        session_manager = request.app.state.session_manager
        session = await session_manager.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session. Please login again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Attach user context to request state
        request.state.user_id = session.user_id
        request.state.session_id = session_id
        request.state.ip_address = session.ip_address

        # Fetch and attach user preferences
        try:
            user_service = request.app.state.user_service
            preferences = await user_service.get_user_preferences(session.user_id)
            request.state.preferences = preferences
        except Exception as e:
            logger.error("Error loading user preferences: %s", e)
            # Continue with default preferences
            from src.gateway.services.user_service import UserPreferences

            request.state.preferences = UserPreferences()

        # Process request
        response = await call_next(request)

        return response


# Dependency functions for route handlers
async def get_current_user(request: Request) -> str:
    """
    Dependency to get current user ID from request state.

    Returns:
        User ID
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated"
        )
    return request.state.user_id


async def get_user_preferences(request: Request):
    """
    Dependency to get user preferences from request state.

    Returns:
        User preferences
    """
    if not hasattr(request.state, "preferences"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User preferences not loaded",
        )
    return request.state.preferences


async def get_session_id(request: Request) -> str:
    """
    Dependency to get session ID from request state.

    Returns:
        Session ID
    """
    if not hasattr(request.state, "session_id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No active session"
        )
    return request.state.session_id
