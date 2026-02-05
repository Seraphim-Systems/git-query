"""User service for managing user data and preferences."""

from typing import Optional, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
from pydantic import BaseModel


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

        # Fetch from MongoDB
        user = await self.db.users.find_one({"user_id": user_id})

        if not user or "preferences" not in user:
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

        # Invalidate cache
        await self.redis.delete(f"user_prefs:{user_id}")

        return UserPreferences(**preferences)

    async def record_interaction(
        self,
        user_id: str,
        repo_id: str,
        action: str,
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
        interaction = {
            "repo_id": repo_id,
            "action": action,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {},
        }

        # Store in MongoDB
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

        # Store in Redis for real-time access (last 100 interactions)
        await self.redis.lpush(
            f"user_interactions:{user_id}",
            f"{repo_id}:{action}:{datetime.utcnow().isoformat()}",
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

    async def create_user(
        self, user_id: str, email: str, username: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            user_id: User identifier
            email: User email
            username: Username
            **kwargs: Additional user data

        Returns:
            Created user document
        """
        user_doc = {
            "user_id": user_id,
            "email": email,
            "username": username,
            "created_at": datetime.utcnow(),
            "preferences": UserPreferences().model_dump(),
            "interaction_history": [],
            **kwargs,
        }

        await self.db.users.insert_one(user_doc)
        return user_doc
