"""
Authentication utilities for database API
"""

import os
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# Configuration
DATA_INGESTION_API_KEY = os.getenv("DATA_INGESTION_API_KEY")

# Validate that API key is set and not using insecure default
if not DATA_INGESTION_API_KEY or DATA_INGESTION_API_KEY == "change-me":
    raise RuntimeError(
        "DATA_INGESTION_API_KEY must be set to a secure value and cannot be 'change-me'. "
        "Set this environment variable before starting the service."
    )

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate API key for data ingestion endpoints"""
    if api_key == DATA_INGESTION_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API key")
