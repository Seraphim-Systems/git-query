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

        # Skip auth for public endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # Protect database and service endpoints under /api/{service}/
        # e.g. /api/mongodb/, /api/redis/, /api/qdrant/, /api/cosmos/, /api/mcp/
        if request.url.path.startswith("/api/"):
            # Extract service from path: /api/{service}/...
            path_parts = [p for p in request.url.path.split("/") if p]
            if len(path_parts) >= 2:
                service = path_parts[1]
                # Only enforce API keys for known backend services
                protected_services = {"mongodb", "redis", "qdrant", "cosmos", "mcp"}
                if service in protected_services:
                    api_key = self._extract_api_key(request)

                    if not api_key:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="API key required. Provide via 'Authorization: Bearer <key>' header.",
                            headers={"WWW-Authenticate": "Bearer"},
                        )

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
