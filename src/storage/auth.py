"""Authentication utilities for database API.

This module reads per-service API keys from the shared configuration
`src.shared.config.settings`, falling back to legacy environment variables
for backwards compatibility.
"""

from typing import Optional, List
import logging
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from src.shared.config import settings
import os


# Security dependency
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


INVALID_PLACEHOLDERS = {"change-me", "dev-local-key", ""}


def _current_valid_api_keys() -> List[str]:
    """Compute the currently configured valid API keys from settings.

    This is evaluated at call time rather than import time so changes to the
    environment (for local dev) are picked up when the dependency runs.
    """
    # Prefer values from `settings`, but fall back to explicit environment
    # variables when pydantic's settings don't expose them (some pydantic
    # versions ignore the deprecated `env` Field argument).
    candidate_keys: List[Optional[str]] = [
        getattr(settings, "mongodb_api_key", None) or os.getenv("APIKEY_MONGODB"),
        getattr(settings, "redis_api_key", None) or os.getenv("APIKEY_REDIS"),
        getattr(settings, "qdrant_api_key", None) or os.getenv("APIKEY_QDRANT"),
    ]
    return [k for k in candidate_keys if k and k not in INVALID_PLACEHOLDERS]


def get_api_key(
    api_key: Optional[str] = Security(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
) -> str:
    """Validate API key for data ingestion endpoints at request time.

    Returns the provided API key if it matches one of the configured keys,
    otherwise raises HTTP 403. Computes valid keys on each call to ensure
    environment-driven values are used during local dev and container
    restarts.
    """
    valid = _current_valid_api_keys()
    # Debug: log counts and presence (do not log keys themselves)
    logger.debug(
        "get_api_key called: valid_keys=%d, has_x_api=%s, has_bearer=%s",
        len(valid),
        bool(api_key),
        bool(bearer),
    )
    if not valid:
        logger.warning(
            "No valid API keys found for storage services. "
            "Storage endpoints will reject requests until API keys are configured."
        )

    # Prefer explicit X-API-Key header
    if api_key and api_key in valid:
        logger.debug("X-API-Key provided and accepted")
        return api_key

    # Fallback to Authorization: Bearer <token>
    if bearer and getattr(bearer, "scheme", "").lower() == "bearer":
        token = getattr(bearer, "credentials", None)
        logger.debug("Bearer token present: %s", bool(token))
        if token and token in valid:
            logger.debug("Bearer token accepted")
            return token

    raise HTTPException(status_code=403, detail="Invalid or missing API key")
