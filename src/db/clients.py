"""Concrete DB client startup and lifecycle management.

This module exposes startup/shutdown helpers and runtime accessors for
database clients. It prefers configuration from `src.shared.config.settings`.
"""

import logging
from typing import Optional
from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)

# Global clients (populated during startup)
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

    # Delegate to init module which will use shared settings
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
