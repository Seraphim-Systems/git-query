"""Session middleware for request authentication."""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from src.gateway.middleware.shared import (
    PUBLIC_PATHS,
    SESSION_PATHS,
    GATEWAY_API_PREFIXES,
)

logger = logging.getLogger(__name__)


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware to validate sessions (JWT or Redis) and inject user context."""

    async def dispatch(self, request: Request, call_next):
        """Process request and validate session."""

        # Frontend requests (not a known API prefix) go straight to the proxy
        # handler; no session check needed.
        if not any(request.url.path.startswith(p) for p in GATEWAY_API_PREFIXES):
            return await call_next(request)

        # Skip auth for public API endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # Only enforce session authentication for user-facing session paths.
        # API/DB routes are handled by the API-key middleware.
        if not any(request.url.path.startswith(path) for path in SESSION_PATHS):
            return await call_next(request)

        # If API key middleware has already validated an API key for a
        # backend service mounted under `/api/...`, skip user session checks
        # because these requests are authenticated by service keys.
        if request.url.path.startswith("/api/") and getattr(request.state, "api_key", None):
            return await call_next(request)

        # Extract token from cookie or Authorization header
        token = request.cookies.get("session_id")

        if not token:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No session found. Please login.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # --- JWT validation (stateless, preferred) ---
        from src.gateway.services.jwt_service import verify_token

        jwt_payload = verify_token(token)
        if jwt_payload:
            request.state.user_id = jwt_payload["sub"]
            request.state.session_id = token
            request.state.ip_address = request.client.host if request.client else "unknown"
            request.state.is_admin = jwt_payload.get("is_admin", False)
        else:
            # --- Redis session fallback ---
            session_manager = request.app.state.session_manager
            session = await session_manager.get_session(token)

            if not session:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired session. Please login again.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            request.state.user_id = session.user_id
            request.state.session_id = token
            request.state.ip_address = session.ip_address
            request.state.is_admin = False

        # Fetch and attach user preferences
        try:
            user_service = request.app.state.user_service
            preferences = await user_service.get_user_preferences(request.state.user_id)
            request.state.preferences = preferences
        except Exception as e:
            logger.error("Error loading user preferences: %s", e)
            from src.gateway.services.user_service import UserPreferences

            request.state.preferences = UserPreferences()

        return await call_next(request)


# Dependency functions for route handlers
async def get_current_user(request: Request) -> str:
    """
    Dependency to get current user ID from request state.

    Returns:
        User ID
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated")
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No active session")
    return request.state.session_id


async def require_admin(request: Request) -> str:
    """
    Dependency to ensure the current user is an admin.

    Returns:
        User ID of the authenticated admin.
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not authenticated")
    if not getattr(request.state, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return request.state.user_id
