"""
Database client management and initialization
"""

import os
from typing import Optional
from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient

# Configuration
MONGODB_URL = os.getenv(
    "MONGODB_URL", "mongodb://admin:mongopass@localhost:27017/gitquery?authSource=admin"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://:redispass@localhost:6379")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Global database clients
mongo_client: Optional[MongoClient] = None
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None


async def startup_db_clients():
    """Initialize database clients on startup"""
    global mongo_client, redis_client, qdrant_client

    try:
        client = MongoClient(MONGODB_URL)
        # Test connection
        client.admin.command("ping")
        mongo_client = client
        print("✓ MongoDB connected")
    except Exception as e:
        mongo_client = None
        print(f"✗ MongoDB connection failed: {e}")

    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        redis_client = client
        print("✓ Redis connected")
    except Exception as e:
        redis_client = None
        print(f"✗ Redis connection failed: {e}")

    try:
        client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY
        )
        client.get_collections()
        qdrant_client = client
        print("✓ Qdrant connected")
    except Exception as e:
        qdrant_client = None
        print(f"✗ Qdrant connection failed: {e}")


async def shutdown_db_clients():
    """Close database connections on shutdown"""
    global mongo_client, redis_client, qdrant_client

    if mongo_client:
        mongo_client.close()
    if redis_client:
        redis_client.close()
    if qdrant_client:
        qdrant_client.close()


def get_mongo_client() -> Optional[MongoClient]:
    """Get MongoDB client"""
    return mongo_client


def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client"""
    return redis_client


def get_qdrant_client() -> Optional[QdrantClient]:
    """Get Qdrant client"""
    return qdrant_client
