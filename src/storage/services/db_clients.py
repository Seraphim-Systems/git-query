"""
Database client management and initialization
"""

import os
import logging
from typing import Optional
from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient

# Configure logging
logger = logging.getLogger(__name__)

# Configuration - no default credentials for security
MONGODB_URL = os.getenv("MONGODB_URL")
REDIS_URL = os.getenv("REDIS_URL")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_USE_TLS = os.getenv("QDRANT_USE_TLS", "false").lower() in ("1", "true", "yes")

# Global database clients
mongo_client: Optional[MongoClient] = None
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None


async def startup_db_clients():
    """Initialize database clients on startup"""
    global mongo_client, redis_client, qdrant_client

    # Validate that required connection strings are provided
    if not MONGODB_URL:
        logger.error("MONGODB_URL environment variable is not set")
        raise RuntimeError("MONGODB_URL must be set to connect to MongoDB")

    if not REDIS_URL:
        logger.error("REDIS_URL environment variable is not set")
        raise RuntimeError("REDIS_URL must be set to connect to Redis")

    # Connect to MongoDB
    try:
        client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command("ping")
        mongo_client = client
        logger.info("✓ MongoDB connected successfully")
    except Exception as e:
        mongo_client = None
        logger.error(f"✗ MongoDB connection failed: {e}")
        raise RuntimeError(f"Failed to connect to MongoDB: {e}")

    # Connect to Redis
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        client.ping()
        redis_client = client
        logger.info("✓ Redis connected successfully")
    except Exception as e:
        redis_client = None
        logger.error(f"✗ Redis connection failed: {e}")
        raise RuntimeError(f"Failed to connect to Redis: {e}")

    # Connect to Qdrant (optional)
    try:
        scheme = "https" if QDRANT_USE_TLS else "http"
        url = f"{scheme}://{QDRANT_HOST}:{QDRANT_PORT}"
        client = QdrantClient(url=url, api_key=QDRANT_API_KEY or None, timeout=5)
        client.get_collections()
        qdrant_client = client
        logger.info("✓ Qdrant connected successfully")
    except Exception as e:
        qdrant_client = None
        logger.warning(f"✗ Qdrant connection failed (optional service): {e}")


async def shutdown_db_clients():
    """Close database connections on shutdown"""
    global mongo_client, redis_client, qdrant_client

    logger.info("Shutting down database clients...")

    if mongo_client:
        try:
            mongo_client.close()
            logger.info("MongoDB client closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB client: {e}")

    if redis_client:
        try:
            redis_client.close()
            logger.info("Redis client closed")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")

    if qdrant_client:
        try:
            # Note: Close method availability depends on qdrant-client version
            # Check if close method exists before calling
            if hasattr(qdrant_client, "close"):
                qdrant_client.close()
                logger.info("Qdrant client closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")


def get_mongo_client() -> Optional[MongoClient]:
    """Get MongoDB client"""
    return mongo_client


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client"""
    return redis_client


def get_qdrant_client() -> Optional[QdrantClient]:
    """Get Qdrant client"""
    return qdrant_client
