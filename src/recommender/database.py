"""Database clients for the recommendation system."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, VectorParams

from .config import settings
from .models import ABTestConfig, EvaluationMetrics, ModelMetadata, UserInteraction, UserPreferences
from src.db.config import db_clients


class DatabaseManager:
    """Manages connections to MongoDB, Qdrant, and Redis."""

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        """Initialize database connections."""
        config = db_clients.config

        mongo_url = config.mongodb_url or settings.mongodb_url
        self.mongo_client = AsyncIOMotorClient(mongo_url)
        self.db = self.mongo_client[config.mongodb_db or "gitquery"]

        self.qdrant_client = QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or settings.qdrant_api_key,
        )

        self.redis_client = await redis.from_url(settings.redis_url)

        await self._ensure_collections()

    async def _ensure_collections(self):
        """Ensure MongoDB indexes and Qdrant collection exist."""
        await self.db[settings.interactions_collection].create_index([("user_id", 1), ("timestamp", -1)])
        await self.db[settings.interactions_collection].create_index([("repo_id", 1)])
        await self.db[settings.interactions_collection].create_index([("variant", 1), ("timestamp", -1)])
        await self.db[settings.user_prefs_collection].create_index([("user_id", 1)], unique=True)

        # Repository text index on the repositories collection.
        for collection_name in {settings.repos_collection}:
            try:
                await self.db[collection_name].create_index(
                    [
                        ("name", "text"),
                        ("description", "text"),
                        ("topics", "text"),
                    ],
                    name="repo_text_search",
                    weights={"name": 10, "topics": 5, "description": 1},
                    language_override="text_language",
                )
            except Exception as exc:
                logger.warning("Could not create text index on %s (may already exist): %s", collection_name, exc)

        await self.db["evaluation_metrics"].create_index([("variant", 1), ("timestamp", -1)])

        # Qdrant collection
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.qdrant_client.get_collection, settings.qdrant_repos_collection)
        except Exception:
            await loop.run_in_executor(
                None,
                lambda: self.qdrant_client.create_collection(
                    collection_name=settings.qdrant_repos_collection,
                    vectors_config=VectorParams(size=settings.embedding_dimension, distance=Distance.COSINE),
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
        result = await self.db[settings.interactions_collection].insert_one(interaction.model_dump())
        return str(result.inserted_id)

    async def get_user_interactions(self, user_id: str, limit: int = 100, days: int = 30) -> List[UserInteraction]:
        """Get recent interactions for a user."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        cursor = (
            self.db[settings.interactions_collection]
            .find({"user_id": user_id, "timestamp": {"$gte": cutoff}})
            .sort("timestamp", -1)
            .limit(limit)
        )

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
        for collection_name in [settings.repos_collection]:
            cursor = self.db[collection_name].find(query_filter).skip(skip).limit(limit)
            repos = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                repos.append(doc)
            if repos:
                return repos
        return []

    async def get_repositories_by_repo_ids(self, repo_ids: List[str]) -> Dict[str, Any]:
        """Fetch full repo metadata from repositories by repo_id / _id.

        Returns a mapping of repo_id → document for any IDs found.
        """
        if not repo_ids:
            return {}

        query_filter = {
            "$or": [
                {"_id": {"$in": repo_ids}},
                {"repo_id": {"$in": repo_ids}},
            ]
        }

        docs: List[Dict[str, Any]] = []
        try:
            cursor = self.db[settings.repos_collection].find(query_filter).limit(len(repo_ids))
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                docs.append(doc)
        except Exception as exc:
            logger.warning("repositories lookup failed: %s", exc)
            return {}

        result_map: Dict[str, Any] = {}
        for d in docs:
            key = d.get("repo_id") or d.get("_id") or d.get("nameWithOwner", "")
            if key:
                result_map[key] = d
        return result_map

    # ===== Vector Search (Qdrant) =====

    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 100,
        filter_conditions: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar repositories using vector similarity.

        Runs the synchronous QdrantClient call on the thread-pool executor
        to keep the event loop unblocked.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.qdrant_client.search(
                collection_name=settings.qdrant_repos_collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_conditions,
                with_payload=True,
            ),
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
        await self.redis_client.setex(f"reco:{key}", ttl, json.dumps(value, default=str))

    # ===== Metrics =====

    async def save_metrics(self, metrics: EvaluationMetrics):
        """Save evaluation metrics."""
        await self.db["evaluation_metrics"].insert_one(metrics.model_dump())

    async def get_latest_metrics(self, variant: str) -> Optional[EvaluationMetrics]:
        """Get latest metrics for a variant."""
        doc = await self.db["evaluation_metrics"].find_one({"variant": variant}, sort=[("timestamp", -1)])
        if doc:
            doc.pop("_id", None)
            return EvaluationMetrics(**doc)
        return None

    # ===== A/B Testing =====

    async def get_active_ab_test(self) -> Optional[ABTestConfig]:
        """Get currently active A/B test."""
        now = datetime.now(timezone.utc)
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
        """Get the active model for a given type and variant."""
        doc = await self.db[settings.models_collection].find_one(
            {"model_type": model_type, "variant": variant, "is_active": True},
            sort=[("trained_at", -1)],
        )
        if doc:
            doc.pop("_id", None)
            return ModelMetadata(**doc)
        return None

    async def get_model_by_id(self, model_id: str) -> Optional[ModelMetadata]:
        """Get a model by its ID."""
        doc = await self.db[settings.models_collection].find_one({"model_id": model_id})
        if doc:
            doc.pop("_id", None)
            return ModelMetadata(**doc)
        return None

    async def deactivate_models(self, model_type: str, variant: str) -> None:
        """Archive all currently active models of a given type/variant."""
        await self.db[settings.models_collection].update_many(
            {"model_type": model_type, "variant": variant, "is_active": True},
            {"$set": {"is_active": False, "status": "archived"}},
        )

    async def activate_model(self, model_id: str) -> None:
        """Promote a model to active status."""
        await self.db[settings.models_collection].update_one(
            {"model_id": model_id},
            {"$set": {"is_active": True, "status": "active"}},
        )

    async def list_models_query(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """List models with optional type/status filtering."""
        query: Dict[str, Any] = {}
        if model_type:
            query["model_type"] = model_type
        if status:
            query["status"] = status

        cursor = self.db[settings.models_collection].find(query).sort("trained_at", -1)
        models = []
        async for doc in cursor:
            doc.pop("_id", None)
            models.append(ModelMetadata(**doc))
        return models


# Global database manager instance
db_manager = DatabaseManager()
