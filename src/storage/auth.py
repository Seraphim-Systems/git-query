"""
Authentication utilities for database API
"""

import os
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# Configuration - Per-service API keys
APIKEY_MONGODB = os.getenv("MONGODB_API_KEY")
APIKEY_REDIS = os.getenv("REDIS_API_KEY")
APIKEY_QDRANT = os.getenv("QDRANT_API_KEY_AUTH")

# Backward compatibility: fall back to old DATA_INGESTION_API_KEY if new keys not set
LEGACY_API_KEY = os.getenv("DATA_INGESTION_API_KEY")

# Build list of valid API keys (excluding None and insecure defaults)
VALID_API_KEYS = []
for key_name, key_value in [
    ("MONGODB_API_KEY", APIKEY_MONGODB),
    ("REDIS_API_KEY", APIKEY_REDIS),
    ("QDRANT_API_KEY_AUTH", APIKEY_QDRANT),
    ("DATA_INGESTION_API_KEY (legacy)", LEGACY_API_KEY),
]:
    if key_value and key_value not in ["change-me", "dev-local-key"]:
        VALID_API_KEYS.append(key_value)

# Validate that at least one API key is set
if not VALID_API_KEYS:
    raise RuntimeError(
        "At least one API key must be set to a secure value. "
        "Set MONGODB_API_KEY, REDIS_API_KEY, or QDRANT_API_KEY_AUTH environment variables. "
        "Keys cannot be 'change-me' or 'dev-local-key'."
    )

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate API key for data ingestion endpoints"""
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API key")
