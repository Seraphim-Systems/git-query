"""Database clients for the recommendation system."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import redis.asyncio as redis
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from .config import settings
from .models import ABTestConfig, EvaluationMetrics, ModelMetadata, UserInteraction, UserPreferences
from src.db.config import db_clients


# ---------------------------------------------------------------------------
# Gateway-mode helpers — duck-type the Qdrant client using gateway REST calls
# ---------------------------------------------------------------------------

class _Hit:
    """Minimal stand-in for a Qdrant ScoredPoint returned by QdrantClient.search()."""
    def __init__(self, id_: str, score: float, payload: dict):
        self.id = id_
        self.score = score
        self.payload = payload or {}


class _GatewayQdrantClient:
    """Routes QdrantClient.search() calls through the gateway REST API.

    Assigned to ``DatabaseManager.qdrant_client`` when ``USE_GATEWAY=true``
    so that ``vector_search()`` works without any other code changes.
    """

    def __init__(self, base_url: str, session: Any):
        self._base = base_url.rstrip("/")
        self._session = session

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter=None,
        with_payload: bool = True,
        **_,
    ) -> List[_Hit]:
        body: Dict[str, Any] = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": with_payload,
        }
        resp = self._session.post(
            f"{self._base}/api/qdrant/collections/{collection_name}/search",
            json=body,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data if isinstance(data, list) else data.get("results", data.get("hits", []))
        return [_Hit(h.get("id"), h.get("score", 0.0), h.get("payload", {})) for h in hits]

    # Stubs for the few lifecycle calls made during startup checks
    def get_collection(self, name: str):
        return None

    def get_collections(self):
        return None


class DatabaseManager:
    """Manages connections to MongoDB, Qdrant, and Redis.

    Supports two connection modes:
    - **Native mode** (default): connects directly to Qdrant, MongoDB, and Redis
      using their respective client libraries.
    - **Gateway mode** (``USE_GATEWAY=true``): routes all database calls through
      the project's REST gateway at ``API_BASE_URL``.  This is useful when the
      native database ports are not publicly reachable but the gateway is.
    """

    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.qdrant_client: Optional[Any] = None  # QdrantClient or _GatewayQdrantClient
        self.redis_client: Optional[redis.Redis] = None
        self.db = None
        self._gateway_mode: bool = False
        self._gw_session: Optional[Any] = None  # requests.Session
        self._gw_base: str = ""

    # ------------------------------------------------------------------
    # Gateway helpers
    # ------------------------------------------------------------------

    def _gw_post(self, path: str, body: dict) -> dict:
        resp = self._gw_session.post(f"{self._gw_base}{path}", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _gw_put(self, path: str, body: dict) -> dict:
        resp = self._gw_session.put(f"{self._gw_base}{path}", json=body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _gw_get(self, path: str) -> dict:
        resp = self._gw_session.get(f"{self._gw_base}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    async def _run_sync(self, fn):
        """Run a synchronous gateway call on the thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        """Initialize database connections.

        In gateway mode, a ``requests.Session`` is created and pointed at
        ``API_BASE_URL``.  Native client initialization and index setup are
        skipped — the production databases already have the required indexes.
        """
        if settings.use_gateway:
            import requests
            self._gateway_mode = True
            self._gw_base = settings.api_base_url.rstrip("/")
            api_key = settings.apikey_qdrant or settings.qdrant_api_key or ""
            self._gw_session = requests.Session()
            self._gw_session.headers.update({
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            })
            self.qdrant_client = _GatewayQdrantClient(self._gw_base, self._gw_session)
            return  # Skip native client init and _ensure_collections

        # ---- Native mode ----
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
        """Ensure MongoDB collections and Qdrant collections exist."""
        # MongoDB indexes
        await self.db[settings.interactions_collection].create_index(
            [("user_id", 1), ("timestamp", -1)]
        )
        await self.db[settings.interactions_collection].create_index([("repo_id", 1)])
        await self.db[settings.user_prefs_collection].create_index([("user_id", 1)], unique=True)

        # Repository text index for performant keyword search — created on both
        # repos_collection (the clean/normalised store) and raw_repos_collection
        # (the pipeline source) so keyword search works whichever is active.
        for collection_name in {settings.repos_collection, settings.raw_repos_collection}:
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
        if self._gateway_mode:
            if self._gw_session:
                self._gw_session.close()
            return
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()

    # ===== User Interactions =====

    async def log_interaction(self, interaction: UserInteraction) -> str:
        """Log a user interaction."""
        if self._gateway_mode:
            doc = json.loads(json.dumps(interaction.model_dump(), default=str))
            await self._run_sync(lambda: self._gw_post(
                f"/api/mongodb/collections/{settings.interactions_collection}/bulk",
                {"documents": [doc], "upsert": False},
            ))
            return "gateway-logged"
        result = await self.db[settings.interactions_collection].insert_one(
            interaction.model_dump()
        )
        return str(result.inserted_id)

    async def get_user_interactions(
        self, user_id: str, limit: int = 100, days: int = 30
    ) -> List[UserInteraction]:
        """Get recent interactions for a user."""
        if self._gateway_mode:
            # Timestamp-range queries over the gateway are unreliable because
            # datetime objects must be serialised to strings for JSON transport
            # and MongoDB won't compare BSON dates against strings.
            # Personalization degrades gracefully when this returns [].
            logger.warning("get_user_interactions not supported in gateway mode; skipping")
            return []
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
        if self._gateway_mode:
            result = await self._run_sync(lambda: self._gw_post(
                f"/api/mongodb/collections/{settings.user_prefs_collection}/query",
                {"filter": {"user_id": user_id}, "limit": 1},
            ))
            docs = result.get("documents", [])
            if docs:
                docs[0].pop("_id", None)
                return UserPreferences(**docs[0])
            return None
        doc = await self.db[settings.user_prefs_collection].find_one({"user_id": user_id})
        if doc:
            doc.pop("_id", None)
            return UserPreferences(**doc)
        return None

    async def update_user_preferences(self, preferences: UserPreferences):
        """Update user preferences."""
        if self._gateway_mode:
            doc = json.loads(json.dumps(preferences.model_dump(), default=str))
            await self._run_sync(lambda: self._gw_post(
                f"/api/mongodb/collections/{settings.user_prefs_collection}/bulk",
                {"documents": [doc], "upsert": True},
            ))
            return
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
        if self._gateway_mode:
            try:
                result = await self._run_sync(lambda: self._gw_post(
                    f"/api/mongodb/collections/{settings.repos_collection}/query",
                    {"filter": query_filter, "limit": limit, "skip": skip},
                ))
                return result.get("documents", [])
            except Exception as exc:
                logger.warning("Gateway MongoDB search failed (returning empty): %s", exc)
                return []
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

    async def get_repositories_by_repo_ids(
        self, repo_ids: List[str]
    ) -> Dict[str, Any]:
        """Fetch full repo metadata from raw_repositories by owner/name IDs.

        Returns a mapping of repo_id → document for any IDs found.
        Falls back gracefully on gateway errors.
        """
        if not repo_ids:
            return {}

        # Build OR filter: each repo_id is "owner/name", so split and match.
        conditions = []
        for rid in repo_ids:
            parts = rid.split("/", 1)
            if len(parts) == 2:
                conditions.append({"owner": parts[0], "name": parts[1]})
            else:
                conditions.append({"name": rid})

        query_filter = {"$or": conditions} if len(conditions) > 1 else conditions[0]

        docs: List[Dict[str, Any]] = []
        try:
            if self._gateway_mode:
                result = await self._run_sync(lambda: self._gw_post(
                    f"/api/mongodb/collections/{settings.raw_repos_collection}/query",
                    {"filter": query_filter, "limit": len(repo_ids)},
                ))
                docs = result.get("documents", [])
            else:
                cursor = self.db[settings.raw_repos_collection].find(query_filter).limit(len(repo_ids))
                async for doc in cursor:
                    doc.pop("_id", None)
                    docs.append(doc)
        except Exception as exc:
            logger.warning("raw_repositories lookup failed: %s", exc)
            return {}

        return {
            f"{d.get('owner', '')}/{d.get('name', '')}": d
            for d in docs
            if d.get("owner") and d.get("name")
        }

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
            with_payload=True,
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
        if self._gateway_mode or not settings.enable_cache:
            return None
        value = await self.redis_client.get(f"reco:{key}")
        if value:
            return json.loads(value)
        return None

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if self._gateway_mode or not settings.enable_cache:
            return
        ttl = ttl or settings.cache_ttl_seconds
        await self.redis_client.setex(
            f"reco:{key}", ttl, json.dumps(value, default=str)
        )

    # ===== Metrics =====

    async def save_metrics(self, metrics: EvaluationMetrics):
        """Save evaluation metrics."""
        if self._gateway_mode:
            logger.warning("save_metrics not supported in gateway mode; skipping")
            return
        await self.db["evaluation_metrics"].insert_one(metrics.model_dump())

    async def get_latest_metrics(self, variant: str) -> Optional[EvaluationMetrics]:
        """Get latest metrics for a variant."""
        if self._gateway_mode:
            return None
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
        if self._gateway_mode:
            return None
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
        if self._gateway_mode:
            logger.warning("save_model_metadata not supported in gateway mode; skipping")
            return
        await self.db[settings.models_collection].insert_one(metadata.model_dump())

    async def get_active_model(self, model_type: str, variant: str) -> Optional[ModelMetadata]:
        """Get active model for a variant."""
        if self._gateway_mode:
            return None
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

