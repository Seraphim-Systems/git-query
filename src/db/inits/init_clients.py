"""Database initialization helpers.

This module centralizes eager startup/shutdown logic for external database
clients. It intentionally keeps the implementation separate from
`db.clients` to make init logic easier to reason about and test. The
functions will assign the live client objects onto the `db.clients` module
attributes so the rest of the codebase can continue to access them via
`db.clients.get_*` helpers.
"""

import os
import logging
from typing import Optional

from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient

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


def _assign(module, name: str, value):
    try:
        setattr(module, name, value)
    except Exception:
        logger.exception("Failed to assign %s on %s", name, module)


async def startup_db_clients():
    """Initialize and assign database clients onto `db.clients`.

    This function is async because it may be awaited from FastAPI lifespans
    or other async startup hooks. It uses synchronous client constructors
    but performs connectivity checks where appropriate.
    """
    # Import here to avoid circular imports at module import time
    import src.db.clients as clients_mod

    # MongoDB
    if not MONGODB_URL:
        logger.error("MONGODB_URL environment variable is not set")
        raise RuntimeError("MONGODB_URL must be set to connect to MongoDB")

    try:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        _assign(clients_mod, "mongo_client", client)
        logger.info("✓ MongoDB connected successfully")
    except Exception as e:
        _assign(clients_mod, "mongo_client", None)
        logger.error(f"✗ MongoDB connection failed: {e}")
        raise RuntimeError(f"Failed to connect to MongoDB: {e}")

    # Cosmos (Mongo API)
    effective_cosmos_url = (
        COSMOS_DB_URL or f"mongodb://{COSMOS_DB_HOST}:{COSMOS_DB_PORT}"
    )
    effective_cosmos_key = COSMOS_DB_KEY or APIKEY_COSMODB

    try:
        client = MongoClient(
            effective_cosmos_url,
            password=effective_cosmos_key,
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True,
        )
        client.admin.command("ping")
        _assign(clients_mod, "cosmos_client", client)
        try:
            # mirror into any config instance if present
            from src.db.config import db_clients as db_clients_cfg

            db_clients_cfg._cosmos_client = client
        except Exception:
            logger.debug("db_config mirror unavailable")
        logger.info(
            f"✓ CosmosDB (Mongo API) connected successfully ({effective_cosmos_url})"
        )
    except Exception as e:
        _assign(clients_mod, "cosmos_client", None)
        logger.warning(f"✗ CosmosDB connection failed (optional): {e}")

    # Redis
    if not REDIS_URL:
        logger.error("REDIS_URL environment variable is not set")
        raise RuntimeError("REDIS_URL must be set to connect to Redis")

    try:
        rc = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        rc.ping()
        _assign(clients_mod, "redis_client", rc)
        logger.info("✓ Redis connected successfully")
    except Exception as e:
        _assign(clients_mod, "redis_client", None)
        logger.error(f"✗ Redis connection failed: {e}")
        raise RuntimeError(f"Failed to connect to Redis: {e}")

    # Qdrant (optional)
    try:
        scheme = "https" if QDRANT_USE_TLS else "http"
        url = f"{scheme}://{QDRANT_HOST}:{QDRANT_PORT}"
        qc = QdrantClient(url=url, api_key=QDRANT_API_KEY or None, timeout=5)
        qc.get_collections()
        _assign(clients_mod, "qdrant_client", qc)
        logger.info("✓ Qdrant connected successfully")
    except Exception as e:
        _assign(clients_mod, "qdrant_client", None)
        logger.warning(f"✗ Qdrant connection failed (optional service): {e}")


async def shutdown_db_clients():
    """Close and clear database clients assigned on `db.clients`."""
    import src.db.clients as clients_mod

    logger.info("Shutting down database clients...")

    try:
        if getattr(clients_mod, "mongo_client", None):
            clients_mod.mongo_client.close()
            logger.info("MongoDB client closed")
    except Exception as e:
        logger.error(f"Error closing MongoDB client: {e}")

    try:
        if getattr(clients_mod, "redis_client", None):
            clients_mod.redis_client.close()
            logger.info("Redis client closed")
    except Exception as e:
        logger.error(f"Error closing Redis client: {e}")

    try:
        if getattr(clients_mod, "qdrant_client", None) and hasattr(
            clients_mod.qdrant_client, "close"
        ):
            clients_mod.qdrant_client.close()
            logger.info("Qdrant client closed")
    except Exception as e:
        logger.error(f"Error closing Qdrant client: {e}")

    try:
        if getattr(clients_mod, "cosmos_client", None):
            clients_mod.cosmos_client.close()
            logger.info("Cosmos client closed")
    except Exception as e:
        logger.error(f"Error closing Cosmos client: {e}")

    # Clear mirror in db.config if present
    try:
        from src.db.config import db_clients as db_clients_cfg

        db_clients_cfg._cosmos_client = None
    except Exception:
        pass
