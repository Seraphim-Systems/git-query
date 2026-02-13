"""Authentication utilities for database API.

This module reads per-service API keys from the shared configuration
`src.shared.config.settings`, falling back to legacy environment variables
for backwards compatibility.
"""

import os
from typing import Optional, List
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.shared.config import settings


# Gather candidate API keys from shared settings first, then fall back to
# legacy environment variables used in older deployments.

# Primary keys come from shared settings (which map to the new APIKEY_* env vars)
candidate_keys: List[Optional[str]] = [
    getattr(settings, "mongodb_api_key", None),
    getattr(settings, "redis_api_key", None),
    getattr(settings, "qdrant_api_key", None),
]


# Filter out insecure placeholder values and None
INVALID_PLACEHOLDERS = {"change-me", "dev-local-key", ""}
VALID_API_KEYS = [k for k in candidate_keys if k and k not in INVALID_PLACEHOLDERS]

if not VALID_API_KEYS:
    raise RuntimeError(
        "At least one API key must be set to a secure value. "
        "Configure via shared settings or environment variables."
    )


# Security dependency
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate API key for data ingestion endpoints.

    Returns the provided API key if it matches one of the configured keys,
    otherwise raises HTTP 403.
    """
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API key")
