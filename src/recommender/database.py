"""Database clients for the recommendation system."""

import asyncio
import os
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import redis.asyncio as redis
from datetime import datetime, timedelta
import json

from .config import settings
from .models import UserInteraction, UserPreferences, EvaluationMetrics, ABTestConfig, ModelMetadata
from src.db.config import db_clients


class DatabaseManager:
    """Manages connections to MongoDB, Qdrant, and Redis."""

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db = None

    async def connect(self):
        """Initialize all database connections using shared storage configuration."""
        config = db_clients.config
        
        # MongoDB
        # Use shared settings or env vars for the URL
        mongo_url = config.mongodb_url or settings.mongodb_url
        self.mongo_client = AsyncIOMotorClient(mongo_url)
        self.db = self.mongo_client[config.mongodb_db or "gitquery"]

        # Qdrant
        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or settings.qdrant_api_key,
        )

        # Redis
        redis_url = os.getenv("REDIS_URL") or settings.redis_url
        self.redis_client = await redis.from_url(redis_url)

        # Ensure collections exist
        await self._ensure_collections()

    async def _ensure_collections(self):
        """Ensure MongoDB collections and Qdrant collections exist."""
        # MongoDB indexes
        await self.db[settings.interactions_collection].create_index(
            [("user_id", 1), ("timestamp", -1)]
        )
        await self.db[settings.interactions_collection].create_index([("repo_id", 1)])
        await self.db[settings.user_prefs_collection].create_index([("user_id", 1)], unique=True)

        # Qdrant collection
        try:
            self.qdrant_client.get_collection(settings.qdrant_repos_collection)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=settings.qdrant_repos_collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension, distance=Distance.COSINE
                ),
            )

    async def close(self):
        """Close all database connections."""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()

    # ===== User Interactions =====

    async def log_interaction(self, interaction: UserInteraction) -> str:
        """Log a user interaction."""
        result = await self.db[settings.interactions_collection].insert_one(
            interaction.model_dump()
        )
        return str(result.inserted_id)

    async def get_user_interactions(
        self, user_id: str, limit: int = 100, days: int = 30
    ) -> List[UserInteraction]:
        """Get recent interactions for a user."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cursor = self.db[settings.interactions_collection].find(
            {"user_id": user_id, "timestamp": {"$gte": cutoff}}
        ).sort("timestamp", -1).limit(limit)

        interactions = []
        async for doc in cursor:
            doc.pop("_id", None)
            interactions.append(UserInteraction(**doc))
        return interactions

    # ===== User Preferences =====

    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Get user preferences."""
        doc = await self.db[settings.user_prefs_collection].find_one({"user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return UserPreferences(**doc)
        return None

    async def update_user_preferences(self, preferences: UserPreferences):
        """Update user preferences."""
        await self.db[settings.user_prefs_collection].update_one(
            {"user_id": preferences.user_id},
            {"$set": preferences.model_dump()},
            upsert=True,
        )

    # ===== Repositories =====

    async def search_repositories(
        self,
        query_filter: Dict[str, Any],
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search repositories with filters."""
        cursor = (
            self.db[settings.repos_collection]
            .find(query_filter)
            .skip(skip)
            .limit(limit)
        )

        repos = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            repos.append(doc)
        return repos

    # ===== Vector Search (Qdrant) =====

    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 100,
        filter_conditions: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar repositories using vector similarity."""
        results = self.qdrant_client.search(
            collection_name=settings.qdrant_repos_collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_conditions,
        )

        return [
            {
                "repo_id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    # ===== Caching (Redis) =====

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not settings.enable_cache:
            return None

        value = await self.redis_client.get(f"reco:{key}")
        if value:
            return json.loads(value)
        return None

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if not settings.enable_cache:
            return

        ttl = ttl or settings.cache_ttl_seconds
        await self.redis_client.setex(
            f"reco:{key}", ttl, json.dumps(value, default=str)
        )

    # ===== Metrics =====

    async def save_metrics(self, metrics: EvaluationMetrics):
        """Save evaluation metrics."""
        await self.db["evaluation_metrics"].insert_one(metrics.model_dump())

    async def get_latest_metrics(self, variant: str) -> Optional[EvaluationMetrics]:
        """Get latest metrics for a variant."""
        doc = await self.db["evaluation_metrics"].find_one(
            {"variant": variant}, sort=[("timestamp", -1)]
        )
        if doc:
            doc.pop("_id", None)
            return EvaluationMetrics(**doc)
        return None

    # ===== A/B Testing =====

    async def get_active_ab_test(self) -> Optional[ABTestConfig]:
        """Get currently active A/B test."""
        now = datetime.utcnow()
        doc = await self.db[settings.ab_tests_collection].find_one(
            {
                "is_active": True,
                "start_date": {"$lte": now},
                "$or": [{"end_date": None}, {"end_date": {"$gte": now}}],
            }
        )
        if doc:
            doc.pop("_id", None)
            return ABTestConfig(**doc)
        return None

    # ===== Model Management =====

    async def save_model_metadata(self, metadata: ModelMetadata):
        """Save model metadata."""
        await self.db[settings.models_collection].insert_one(metadata.model_dump())

    async def get_active_model(self, model_type: str, variant: str) -> Optional[ModelMetadata]:
        """Get active model for a variant."""
        doc = await self.db[settings.models_collection].find_one(
            {"model_type": model_type, "variant": variant, "is_active": True},
            sort=[("trained_at", -1)],
        )
        if doc:
            doc.pop("_id", None)
            return ModelMetadata(**doc)
        return None


# Global database manager instance
db_manager = DatabaseManager()

