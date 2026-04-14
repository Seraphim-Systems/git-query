"""User service for managing user data and preferences."""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
from pydantic import BaseModel
import logging


logger = logging.getLogger(__name__)


class UserPreferences(BaseModel):
    """User preferences for recommendations."""

    languages: list[str] = []
    topics: list[str] = []
    frameworks: list[str] = []
    min_stars: int = 0
    max_age_days: int = 365
    exclude_archived: bool = True
    exclude_forks: bool = True
    license_types: list[str] = []
    company_blacklist: list[str] = []


class UserService:
    """Service for managing user data and preferences."""

    def __init__(
        self, mongodb: AsyncIOMotorDatabase, redis: Redis, cache_ttl: int = 3600
    ):
        """
        Initialize user service.

        Args:
            mongodb: MongoDB database connection
            redis: Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 1 hour)
        """
        self.db = mongodb
        self.redis = redis
        self.cache_ttl = cache_ttl
        self._action_map = {
            "click": "click",
            "view": "view",
            "dismiss": "dismiss",
            "thumbs_up": "thumbs_up",
            "thumbs_down": "thumbs_down",
            "save": "save",
            "star": "save",
            "bookmark": "save",
            "clone": "click",
            "fork": "click",
            "open": "click",
        }
        self._value_map = {
            "view": 0.1,
            "click": 1.0,
            "save": 3.0,
            "thumbs_up": 5.0,
            "dismiss": -2.0,
            "thumbs_down": -5.0,
        }

    def _to_recommender_interaction_type(self, action: str) -> str:
        """Map gateway feedback actions to recommender interaction types."""
        return self._action_map.get((action or "").lower(), "view")

    def _to_interaction_value(self, action: str) -> float:
        """Return the numeric signal weight for an action."""
        interaction_type = self._to_recommender_interaction_type(action)
        return self._value_map.get(interaction_type, 0.0)

    @staticmethod
    def _normalize_chat_message(message: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a chat message payload from the frontend."""
        if not isinstance(message, dict):
            return {
                "content": "",
                "role": "assistant",
                "timestamp": int(datetime.utcnow().timestamp() * 1000),
            }

        timestamp = message.get("timestamp")
        if not isinstance(timestamp, int):
            timestamp = int(datetime.utcnow().timestamp() * 1000)

        role = message.get("role")
        if role not in {"user", "assistant"}:
            role = "assistant"

        return {
            "content": str(message.get("content", "")),
            "role": role,
            "timestamp": timestamp,
        }

    def _normalize_chat_session(
        self, session: Dict[str, Any], fallback_idx: int
    ) -> Dict[str, Any]:
        """Normalize a chat session payload from the frontend."""
        session_id = str(session.get("id") or f"chat-{uuid4()}")
        title = str(session.get("title") or "New Session")
        timestamp = session.get("timestamp")
        if not isinstance(timestamp, int):
            timestamp = int(datetime.utcnow().timestamp() * 1000)

        raw_messages = session.get("messages") or []
        if not isinstance(raw_messages, list):
            raw_messages = []
        messages = [self._normalize_chat_message(m) for m in raw_messages]

        return {
            "id": session_id,
            "title": title,
            "timestamp": timestamp,
            "messages": messages,
            "sort_order": int(session.get("sort_order", fallback_idx)),
        }

    @staticmethod
    def _normalize_saved_repo(
        repo: Dict[str, Any], fallback_idx: int
    ) -> Dict[str, Any]:
        """Normalize a saved repo payload from the frontend."""
        return {
            "id": str(repo.get("id") or f"repo-{uuid4()}"),
            "name": str(repo.get("name") or "Unknown"),
            "owner": str(repo.get("owner") or "Unknown"),
            "description": str(repo.get("description") or "No description available"),
            "stars": int(repo.get("stars") or 0),
            "forks": int(repo.get("forks") or 0),
            "language": str(repo.get("language") or "Unknown"),
            "url": str(repo.get("url") or ""),
            "sort_order": int(repo.get("sort_order", fallback_idx)),
        }

    def _normalize_folder_item(
        self, item: Dict[str, Any], fallback_idx: int
    ) -> Dict[str, Any]:
        """Normalize a folder item payload from the frontend."""
        if not isinstance(item, dict):
            return {
                "id": f"item-{fallback_idx}-{uuid4()}",
                "name": "Untitled",
            }

        normalized = dict(item)
        normalized["id"] = str(normalized.get("id") or f"item-{fallback_idx}-{uuid4()}")

        if "title" in normalized:
            normalized["title"] = str(normalized.get("title") or "")
        if "name" in normalized:
            normalized["name"] = str(normalized.get("name") or "")
        if "url" in normalized:
            normalized["url"] = str(normalized.get("url") or "")

        raw_messages = normalized.get("messages")
        if isinstance(raw_messages, list):
            normalized["messages"] = [
                self._normalize_chat_message(msg) for msg in raw_messages
            ]

        return normalized

    def _normalize_folder(
        self, folder: Dict[str, Any], fallback_idx: int
    ) -> Dict[str, Any]:
        """Normalize a folder payload from the frontend."""
        raw_items = folder.get("items") or []
        if not isinstance(raw_items, list):
            raw_items = []

        return {
            "id": str(folder.get("id") or f"folder-{uuid4()}"),
            "name": str(folder.get("name") or "New Folder"),
            "items": [
                self._normalize_folder_item(item, idx)
                for idx, item in enumerate(raw_items)
            ],
            "expanded": bool(folder.get("expanded", False)),
            "sort_order": int(folder.get("sort_order", fallback_idx)),
        }

    async def get_user_chats(
        self, user_id: str, limit: int = 100
    ) -> list[Dict[str, Any]]:
        """Return persisted chat sessions for a user."""
        collection = self.db.get_collection("user_chats")
        cursor = (
            collection.find(
                {"user_id": user_id},
                {
                    "_id": 0,
                    "user_id": 0,
                },
            )
            .sort("sort_order", 1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)
        return [
            {
                "id": d.get("chat_id"),
                "title": d.get("title") or "New Session",
                "timestamp": d.get("timestamp")
                or int(datetime.utcnow().timestamp() * 1000),
                "messages": d.get("messages") or [],
            }
            for d in docs
            if d.get("chat_id")
        ]

    async def replace_user_chats(
        self, user_id: str, chats: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Replace all persisted chat sessions for a user."""
        collection = self.db.get_collection("user_chats")
        normalized = [
            self._normalize_chat_session(chat, idx) for idx, chat in enumerate(chats)
        ]

        await collection.delete_many({"user_id": user_id})
        if normalized:
            now = datetime.utcnow()
            await collection.insert_many(
                [
                    {
                        "user_id": user_id,
                        "chat_id": chat["id"],
                        "title": chat["title"],
                        "timestamp": chat["timestamp"],
                        "messages": chat["messages"],
                        "sort_order": chat["sort_order"],
                        "updated_at": now,
                    }
                    for chat in normalized
                ]
            )

        for chat in normalized:
            chat.pop("sort_order", None)
        return normalized

    async def get_saved_repos(
        self, user_id: str, limit: int = 200
    ) -> list[Dict[str, Any]]:
        """Return saved repositories for a user."""
        collection = self.db.get_collection("user_saved_repos")
        cursor = (
            collection.find(
                {"user_id": user_id},
                {
                    "_id": 0,
                    "user_id": 0,
                },
            )
            .sort("sort_order", 1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)
        return [
            {
                "id": d.get("repo_id"),
                "name": d.get("name") or "Unknown",
                "owner": d.get("owner") or "Unknown",
                "description": d.get("description") or "No description available",
                "stars": d.get("stars") or 0,
                "forks": d.get("forks") or 0,
                "language": d.get("language") or "Unknown",
                "url": d.get("url") or "",
            }
            for d in docs
            if d.get("repo_id")
        ]

    async def replace_saved_repos(
        self, user_id: str, repos: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Replace all saved repositories for a user."""
        collection = self.db.get_collection("user_saved_repos")
        normalized = [
            self._normalize_saved_repo(repo, idx) for idx, repo in enumerate(repos)
        ]

        await collection.delete_many({"user_id": user_id})
        if normalized:
            now = datetime.utcnow()
            await collection.insert_many(
                [
                    {
                        "user_id": user_id,
                        "repo_id": repo["id"],
                        "name": repo["name"],
                        "owner": repo["owner"],
                        "description": repo["description"],
                        "stars": repo["stars"],
                        "forks": repo["forks"],
                        "language": repo["language"],
                        "url": repo["url"],
                        "sort_order": repo["sort_order"],
                        "updated_at": now,
                    }
                    for repo in normalized
                ]
            )

        for repo in normalized:
            repo.pop("sort_order", None)
        return normalized

    async def get_user_folders(
        self, user_id: str, limit: int = 200
    ) -> list[Dict[str, Any]]:
        """Return saved folders for a user."""
        collection = self.db.get_collection("user_folders")
        cursor = (
            collection.find(
                {"user_id": user_id},
                {
                    "_id": 0,
                    "user_id": 0,
                },
            )
            .sort("sort_order", 1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)
        return [
            {
                "id": d.get("folder_id"),
                "name": d.get("name") or "New Folder",
                "items": d.get("items") or [],
                "expanded": bool(d.get("expanded", False)),
            }
            for d in docs
            if d.get("folder_id")
        ]

    async def replace_user_folders(
        self, user_id: str, folders: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        """Replace all saved folders for a user."""
        collection = self.db.get_collection("user_folders")
        normalized = [
            self._normalize_folder(folder, idx) for idx, folder in enumerate(folders)
        ]

        await collection.delete_many({"user_id": user_id})
        if normalized:
            now = datetime.utcnow()
            await collection.insert_many(
                [
                    {
                        "user_id": user_id,
                        "folder_id": folder["id"],
                        "name": folder["name"],
                        "items": folder["items"],
                        "expanded": folder["expanded"],
                        "sort_order": folder["sort_order"],
                        "updated_at": now,
                    }
                    for folder in normalized
                ]
            )

        for folder in normalized:
            folder.pop("sort_order", None)
        return normalized

    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences with Redis cache.

        Args:
            user_id: User identifier

        Returns:
            User preferences (defaults if not found)
        """
        # Check cache first
        cache_key = f"user_prefs:{user_id}"
        cached = await self.redis.get(cache_key)

        if cached:
            return UserPreferences.model_validate_json(cached)

        # Fetch from MongoDB (gateway user profile doc)
        user = await self.db.users.find_one({"user_id": user_id})

        if not user or "preferences" not in user:
            # Fallback to recommender preferences if available.
            reco_prefs = await self.db.user_preferences.find_one({"user_id": user_id})
            if reco_prefs:
                languages = list((reco_prefs.get("language_preferences") or {}).keys())
                topics = list((reco_prefs.get("topic_preferences") or {}).keys())
                prefs = UserPreferences(
                    languages=languages,
                    topics=topics,
                )
            else:
                # Return default preferences
                prefs = UserPreferences()
        else:
            prefs = UserPreferences(**user["preferences"])

        # Cache for specified TTL
        await self.redis.setex(cache_key, self.cache_ttl, prefs.model_dump_json())

        return prefs

    async def update_preferences(
        self, user_id: str, preferences: Dict[str, Any]
    ) -> UserPreferences:
        """
        Update user preferences.

        Args:
            user_id: User identifier
            preferences: New preferences

        Returns:
            Updated preferences
        """
        # Update in MongoDB
        await self.db.users.update_one(
            {"user_id": user_id},
            {"$set": {"preferences": preferences, "updated_at": datetime.utcnow()}},
            upsert=True,
        )

        # Keep recommender's preference store in sync so personalization reads
        # a consistent profile source.
        lang_scores = {
            language: 1.0 for language in (preferences.get("languages") or [])
        }
        topic_scores = {topic: 1.0 for topic in (preferences.get("topics") or [])}
        existing = await self.db.user_preferences.find_one({"user_id": user_id})
        total_interactions = int((existing or {}).get("total_interactions", 0))

        await self.db.user_preferences.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "language_preferences": lang_scores,
                    "topic_preferences": topic_scores,
                    "last_updated": datetime.utcnow(),
                },
                "$setOnInsert": {"total_interactions": total_interactions},
            },
            upsert=True,
        )

        # Invalidate cache
        await self.redis.delete(f"user_prefs:{user_id}")

        return UserPreferences(**preferences)

    async def record_interaction(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        query: str = "",
        variant: str = "hybrid",
        position_in_results: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record user interaction for collaborative filtering.

        Args:
            user_id: User identifier
            repo_id: Repository identifier
            action: Action type (star, view, clone, fork, click, dismiss)
            metadata: Additional metadata
        """
        now = datetime.utcnow()
        interaction = {
            "repo_id": repo_id,
            "action": action,
            "timestamp": now,
            "metadata": metadata or {},
        }

        # Canonical event log for recommender training/personalization.
        recommender_event = {
            "interaction_id": str(uuid4()),
            "user_id": user_id,
            "query": query,
            "repo_id": repo_id,
            "interaction_type": self._to_recommender_interaction_type(action),
            "interaction_value": self._to_interaction_value(action),
            "position_in_results": position_in_results,
            "variant": variant,
            "timestamp": now,
            "metadata": metadata or {},
        }

        await self.db.user_interactions.insert_one(recommender_event)

        # Keep legacy embedded interaction history for backward compatibility.
        await self.db.users.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "interaction_history": {
                        "$each": [interaction],
                        "$slice": -1000,  # Keep last 1000 interactions
                    }
                },
                "$set": {"last_interaction": datetime.utcnow()},
            },
            upsert=True,
        )

        # Keep recommender preference counters warm for users that only touch
        # the gateway feedback endpoint.
        try:
            await self.db.user_preferences.update_one(
                {"user_id": user_id},
                {
                    "$inc": {"total_interactions": 1},
                    "$set": {"last_updated": now},
                    "$setOnInsert": {
                        "user_id": user_id,
                        "language_preferences": {},
                        "topic_preferences": {},
                    },
                },
                upsert=True,
            )
        except Exception as exc:
            logger.warning("Failed to update user_preferences counters: %s", exc)

        # Store in Redis for real-time access (last 100 interactions)
        await self.redis.lpush(
            f"user_interactions:{user_id}",
            f"{repo_id}:{action}:{now.isoformat()}",
        )
        await self.redis.ltrim(f"user_interactions:{user_id}", 0, 99)

    async def get_interaction_history(
        self, user_id: str, limit: int = 100
    ) -> list[Dict[str, Any]]:
        """
        Get user interaction history.

        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return

        Returns:
            List of interactions
        """
        # Primary source: canonical recommender interaction collection.
        cursor = (
            self.db.user_interactions.find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(limit)
        )
        interactions = await cursor.to_list(length=limit)
        if interactions:
            for item in interactions:
                item.pop("_id", None)
            return interactions

        # Backward-compatible fallback.
        user = await self.db.users.find_one(
            {"user_id": user_id}, {"interaction_history": {"$slice": -limit}}
        )

        if not user or "interaction_history" not in user:
            return []

        return user["interaction_history"]

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user document.

        Args:
            user_id: User identifier

        Returns:
            User document or None
        """
        return await self.db.users.find_one({"user_id": user_id})

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user document by email."""
        return await self.db.users.find_one({"email": email})

    async def create_user(
        self,
        email: str,
        username: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            user_id: Optional user identifier; if omitted a UUIDv4 is generated
            email: User email
            username: Username
            **kwargs: Additional user data

        Returns:
            Created user document
        """
        resolved_user_id = user_id or str(uuid4())
        user_doc = {
            "user_id": resolved_user_id,
            "email": email,
            "username": username,
            "created_at": datetime.utcnow(),
            "preferences": UserPreferences().model_dump(),
            "interaction_history": [],
            **kwargs,
        }

        await self.db.users.insert_one(user_doc)
        return user_doc
