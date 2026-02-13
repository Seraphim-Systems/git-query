"""API Key authentication middleware - simplified per-service validation."""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from src.gateway.middleware.shared import PUBLIC_PATHS
from src.shared.config import settings

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API keys for database access."""

    async def dispatch(self, request: Request, call_next):
        """Process request and validate API key for DB endpoints."""
        # Skip auth for explicitly public endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # Enforce API key for all non-public routes
        api_key = self._extract_api_key(request)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "API key required. Provide via 'X-API-Key: <key>' or "
                    "'Authorization: Bearer <key>' header."
                ),
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate against configured API keys (any service key is acceptable)
        if not self._validate_service_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )

        # Inject key into request state for downstream handlers
        request.state.api_key = api_key

        return await call_next(request)

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from `Authorization: Bearer` or `X-API-Key` header."""
        # Prefer X-API-Key header (used by storage routers) then fallback to Bearer
        header_key = request.headers.get("X-API-Key")
        if header_key:
            return header_key

        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split(" ")
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
        return None

    def _validate_service_key(self, api_key: str, service: str) -> bool:
        """
        Validate API key for a specific service.
        Checks against environment variables or config.
        """
        # If a specific service is provided, prefer validating against that
        # service key. Otherwise, accept any configured service key as valid.
        candidate_keys = [
            getattr(settings, "mongodb_api_key", None),
            getattr(settings, "redis_api_key", None),
            getattr(settings, "qdrant_api_key", None),
            getattr(settings, "mcp_api_key", None),
        ]

        # Filter out None/placeholder values
        valid = [k for k in candidate_keys if k]

        if not valid:
            logger.warning("No API keys configured in settings; rejecting all requests")
            return False

        return api_key in valid
