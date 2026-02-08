"""API Key authentication middleware - simplified per-service validation."""

import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Public endpoints that don't require authentication
PUBLIC_PATHS = [
    "/health",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API keys for database access."""

    async def dispatch(self, request: Request, call_next):
        """Process request and validate API key for DB endpoints."""

        # Skip auth for public endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # All /api/v1/db/* routes require API key
        if request.url.path.startswith("/api/v1/db/"):
            api_key = self._extract_api_key(request)

            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required. Provide via 'Authorization: Bearer <key>' header.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Extract service from path: /api/v1/db/{service}/...
            path_parts = request.url.path.split("/")
            if len(path_parts) >= 4 and path_parts[3]:
                service = path_parts[3]  # mongodb, redis, qdrant, mcp

                # Validate API key for this service
                if not self._validate_service_key(api_key, service, request.app):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Invalid API key for service: {service}",
                    )

                # Inject service and key info into request state
                request.state.api_key = api_key
                request.state.service = service
                logger.debug(f"API key validated for service: {service}")

        return await call_next(request)

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from Authorization header."""
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split(" ")
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
        return None

    def _validate_service_key(self, api_key: str, service: str, app) -> bool:
        """
        Validate API key for a specific service.
        Checks against environment variables or config.
        """
        from src.gateway.config import settings

        # Map service names to config attributes
        service_keys = {
            "mongodb": getattr(settings, "mongodb_api_key", None),
            "redis": getattr(settings, "redis_api_key", None),
            "qdrant": getattr(settings, "qdrant_api_key", None),
            "mcp": getattr(settings, "mcp_api_key", None),
        }

        expected_key = service_keys.get(service)

        if not expected_key:
            logger.warning(f"No API key configured for service: {service}")
            return False

        return api_key == expected_key
