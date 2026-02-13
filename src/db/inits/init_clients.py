"""Database initialization helpers.

Centralizes eager startup/shutdown logic for external database clients.
It uses values from `src.shared.config.settings` where available and
falls back to environment variables for any missing values.
"""

import os
import logging

from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient

from src.shared.config import settings

logger = logging.getLogger(__name__)


def _assign(module, name: str, value):
    try:
        setattr(module, name, value)
    except Exception:
        logger.exception("Failed to assign %s on %s", name, module)


async def startup_db_clients():
    """Initialize and assign database clients onto `db.clients`.

    Uses synchronous client constructors but performs connectivity checks.
    """
    import src.db.clients as clients_mod

    # MongoDB
    mongodb_url = getattr(settings, "mongodb_url", None) or os.getenv("MONGODB_URL")
    if not mongodb_url:
        logger.error("MONGODB_URL is not configured")
        raise RuntimeError("MONGODB_URL must be set to connect to MongoDB")

    # Attempt to connect with retries and exponential backoff since containers
    # may still be starting when the gateway begins its startup sequence.
    import time
    import asyncio

    max_wait_seconds = int(os.getenv("DB_STARTUP_TIMEOUT", "60"))
    attempt = 0
    start = time.time()
    client = None
    last_exc = None
    while True:
        try:
            attempt += 1
            # Keep individual server selection timeout small so failures are
            # detected quickly, but allow multiple attempts up to max_wait_seconds.
            client = MongoClient(mongodb_url, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            _assign(clients_mod, "mongo_client", client)
            logger.info("MongoDB connected (attempt %d)", attempt)
            break
        except Exception as e:
            last_exc = e
            _assign(clients_mod, "mongo_client", None)
            elapsed = time.time() - start
            if elapsed >= max_wait_seconds:
                logger.error("MongoDB connection failed after %ds: %s", elapsed, e)
                raise RuntimeError(f"Failed to connect to MongoDB: {e}")
            # Exponential backoff (capped)
            backoff = min(1 * (2 ** (attempt - 1)), 8)
            logger.warning(
                "MongoDB not ready (attempt %d, waited %ds), retrying in %ds: %s",
                attempt,
                int(elapsed),
                backoff,
                e,
            )
            # Use asyncio.sleep to yield control while waiting
            try:
                await asyncio.sleep(backoff)
            except Exception:
                # In case the event loop is shutting down, propagate last exception
                raise RuntimeError(f"Failed to connect to MongoDB: {last_exc}")

    # Cosmos (Mongo API) - optional
    cosmos_url = os.getenv("COSMOS_DB_URL") or os.getenv("COSMOS_URL")
    cosmos_url = cosmos_url or os.getenv("API_COSMOS_URL")
    # If service exposes a dedicated COSMOS env, prefer it; otherwise skip
    if cosmos_url:
        cosmos_key = os.getenv("COSMOS_DB_KEY") or os.getenv("APIKEY_COSMODB")
        try:
            client = MongoClient(
                cosmos_url,
                password=cosmos_key,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True,
            )
            client.admin.command("ping")
            _assign(clients_mod, "cosmos_client", client)
            try:
                from src.db.config import db_clients as db_clients_cfg

                db_clients_cfg._cosmos_client = client
            except Exception:
                logger.debug("db_config mirror unavailable")
            logger.info("CosmosDB (Mongo API) connected: %s", cosmos_url)
        except Exception as e:
            _assign(clients_mod, "cosmos_client", None)
            logger.warning("CosmosDB connection failed (optional): %s", e)
    else:
        _assign(clients_mod, "cosmos_client", None)

    # Redis
    redis_url = getattr(settings, "redis_url", None) or os.getenv("REDIS_URL")
    if not redis_url:
        logger.error("REDIS_URL is not configured")
        raise RuntimeError("REDIS_URL must be set to connect to Redis")

    try:
        rc = redis.from_url(redis_url, decode_responses=True, socket_timeout=5)
        rc.ping()
        _assign(clients_mod, "redis_client", rc)
        logger.info("Redis connected")
    except Exception as e:
        _assign(clients_mod, "redis_client", None)
        logger.error("Redis connection failed: %s", e)
        raise RuntimeError(f"Failed to connect to Redis: {e}")

    # Qdrant (optional)
    qdrant_host = os.getenv("QDRANT_HOST", os.getenv("QDRANT_HTTP_HOST", "localhost"))
    qdrant_port = int(os.getenv("QDRANT_PORT", os.getenv("QDRANT_HTTP_PORT", "6333")))
    qdrant_api_key = getattr(settings, "qdrant_api_key", None) or os.getenv(
        "QDRANT_API_KEY"
    )
    qdrant_use_tls = os.getenv("QDRANT_USE_TLS", "false").lower() in (
        "1",
        "true",
        "yes",
    )

    try:
        scheme = "https" if qdrant_use_tls else "http"
        url = f"{scheme}://{qdrant_host}:{qdrant_port}"
        qc = QdrantClient(url=url, api_key=qdrant_api_key or None, timeout=10)
        qc.get_collections()
        _assign(clients_mod, "qdrant_client", qc)
        logger.info("Qdrant connected")
    except Exception as e:
        _assign(clients_mod, "qdrant_client", None)
        logger.warning("Qdrant connection failed (optional): %s", e)


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
