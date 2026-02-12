"""Concrete DB client startup and lifecycle management.

This module provides a startup/shutdown lifecycle for all DB clients and
exposes accessor helpers. Services in `storage` re-export these for
backwards compatibility.
"""

import os
import logging
from typing import Optional
from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient

from src.db import config as db_config

logger = logging.getLogger(__name__)

# Config-driven values (allow alternate env names)
COSMOS_DB_URL = os.getenv("COSMOS_DB_URL")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
APIKEY_COSMODB = os.getenv("APIKEY_COSMODB")
COSMOS_DB_HOST = os.getenv("COSMOS_DB_HOST", "cosmos")
COSMOS_DB_PORT = os.getenv("COSMOS_DB_PORT", "10255")

# Other DB envs
MONGODB_URL = os.getenv("MONGODB_URL")
REDIS_URL = os.getenv("REDIS_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_USE_TLS = os.getenv("QDRANT_USE_TLS", "false").lower() in ("1", "true", "yes")

# Global clients
mongo_client: Optional[MongoClient] = None
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None
cosmos_client: Optional[MongoClient] = None


async def startup_db_clients():
    """Delegate initialization to db.inits.init_clients.startup_db_clients.

    The heavy-lifting is implemented in `db.inits.init_clients` to keep the
    startup logic isolated and easier to test. Import is done inside the
    function to avoid circular imports during module import time.
    """
    from src.db.inits.init_clients import startup_db_clients as _startup

    await _startup()


async def shutdown_db_clients():
    """Delegate shutdown to db.inits.init_clients.shutdown_db_clients.

    Import is deferred to avoid circular import issues at module import time.
    """
    from src.db.inits.init_clients import shutdown_db_clients as _shutdown

    await _shutdown()


def get_mongo_client() -> Optional[MongoClient]:
    return mongo_client


def get_redis_client() -> Optional[redis.Redis]:
    return redis_client


def get_qdrant_client() -> Optional[QdrantClient]:
    return qdrant_client


def get_cosmos_client() -> Optional[MongoClient]:
    return cosmos_client
