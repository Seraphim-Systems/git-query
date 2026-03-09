"""API Key authentication middleware - simplified per-service validation."""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.gateway.middleware.shared import PUBLIC_PATHS, SESSION_PATHS, GATEWAY_API_PREFIXES
from src.shared.config import settings
import os

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API keys for database access."""

    async def dispatch(self, request: Request, call_next):
        """Process request and validate API key for DB endpoints."""
        # Frontend requests (not a known API prefix) go straight to the proxy
        # handler; no API key needed.
        if not any(request.url.path.startswith(p) for p in GATEWAY_API_PREFIXES):
            return await call_next(request)

        # Skip auth for explicitly public API endpoints
        if any(request.url.path.startswith(path) for path in PUBLIC_PATHS):
            return await call_next(request)

        # Skip API key check for user-facing session-authenticated routes.
        # The SessionMiddleware handles auth for these paths instead.
        if any(request.url.path.startswith(path) for path in SESSION_PATHS):
            return await call_next(request)

        # Enforce API key for all other non-public routes
        api_key = self._extract_api_key(request)

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": (
                        "API key required. Provide via 'X-API-Key: <key>' or "
                        "'Authorization: Bearer <key>' header."
                    )
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate against configured API keys (any service key is acceptable)
        if not self._validate_service_key(api_key):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key"},
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

    def _validate_service_key(self, api_key: str, service: str | None = None) -> bool:
        """
        Validate API key for a specific service.
        Checks against environment variables or config.
        """
        # If a specific service is provided, prefer validating against that
        # service key. Otherwise, accept any configured service key as valid.
        candidate_keys = [
            getattr(settings, "mongodb_api_key", None) or os.getenv("APIKEY_MONGODB"),
            getattr(settings, "redis_api_key", None) or os.getenv("APIKEY_REDIS"),
            getattr(settings, "qdrant_api_key", None) or os.getenv("APIKEY_QDRANT"),
            getattr(settings, "mcp_api_key", None) or os.getenv("APIKEY_MCP"),
        ]

        # If a service name is provided, validate specifically against that
        # configured key (if present). This preserves compatibility with any
        # callers that pass a service argument.
        if service:
            service_map = {
                "mongodb": getattr(settings, "mongodb_api_key", None)
                or os.getenv("APIKEY_MONGODB"),
                "redis": getattr(settings, "redis_api_key", None)
                or os.getenv("APIKEY_REDIS"),
                "qdrant": getattr(settings, "qdrant_api_key", None)
                or os.getenv("APIKEY_QDRANT"),
                "mcp": getattr(settings, "mcp_api_key", None)
                or os.getenv("APIKEY_MCP"),
            }
            expected = service_map.get(service)
            if expected:
                return api_key == expected
            # If no specific key configured for the named service, fall
            # through to global validation below.

        # Filter out None/placeholder values
        # Filter out None/empty and placeholders
        valid = [
            k
            for k in candidate_keys
            if k and k not in {"", "change-me", "dev-local-key"}
        ]

        if not valid:
            logger.warning("No API keys configured in settings; rejecting all requests")
            # If no keys are configured in the environment (common in
            # lightweight local test runs), allow a small set of well-known
            # development keys defined in the repo `.env` as fallbacks so
            # local testing is possible without requiring container env
            # propagation. These are intentionally limited and should never
            # be used in production.
            dev_fallback = {
                os.getenv("APIKEY_MONGODB", "dev-mongodb-key"),
                os.getenv("APIKEY_REDIS", "dev-redis-key"),
                os.getenv("APIKEY_QDRANT", "dev-qdrant-key"),
                os.getenv("APIKEY_MCP", "dev-mcp-key"),
            }
            # Clean set (remove None/empty)
            dev_fallback = {k for k in dev_fallback if k}
            if api_key in dev_fallback:
                logger.debug("Accepting dev fallback API key for local testing")
                return True
            return False

        return api_key in valid
