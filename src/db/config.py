import os
from typing import Optional
from dataclasses import dataclass

from src.shared.config import settings


@dataclass
class DatabaseConfig:
    """Small holder that maps shared settings into a config shape used by
    legacy callers in the codebase.
    """

    mongodb_url: str
    mongodb_db: str
    qdrant_url: Optional[str]
    qdrant_api_key: Optional[str]

    @classmethod
    def from_settings(cls):
        # Prefer shared settings, otherwise fall back to env vars
        mongodb_url = getattr(settings, "mongodb_url", None) or os.getenv("MONGODB_URL")
        mongodb_db = getattr(settings, "mongodb_db", "gitquery")

        qdrant_url = os.getenv(
            "QDRANT_URL",
            f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_HTTP_PORT', '6333')}",
        )

        qdrant_api_key = getattr(settings, "qdrant_api_key", None) or os.getenv("QDRANT_API_KEY")

        return cls(
            mongodb_url=mongodb_url,
            mongodb_db=mongodb_db,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )


class DatabaseClients:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.config = DatabaseConfig.from_settings()
            self._mongodb_client = self._qdrant_client = None
            self._initialized = True

    @property
    def mongodb(self):
        if self._mongodb_client is None:
            from pymongo import MongoClient

            self._mongodb_client = MongoClient(self.config.mongodb_url)
        return self._mongodb_client

    @property
    def qdrant(self):
        if self._qdrant_client is None and self.config.qdrant_url:
            from qdrant_client import QdrantClient

            self._qdrant_client = QdrantClient(url=self.config.qdrant_url, api_key=self.config.qdrant_api_key)
        return self._qdrant_client

    @property
    def redis(self):
        """Return the runtime Redis client if available.

        The canonical Redis client is managed by `src.db.clients`. Some
        callsites access `db_clients.redis` directly; this property delegates
        to that module to avoid duplication and to preserve a single
        initialization path.
        """
        try:
            # import here to avoid circular import at module import time
            from src.db.clients import get_redis_client

            return get_redis_client()
        except Exception:
            return None

    def close_all(self):
        if self._mongodb_client:
            self._mongodb_client.close()
        if self._qdrant_client:
            self._qdrant_client.close()
        # Attempt to close redis if present on the runtime clients
        try:
            from src.db.clients import get_redis_client

            rc = get_redis_client()
            if rc is not None and hasattr(rc, "close"):
                rc.close()
        except Exception:
            pass


db_clients = DatabaseClients()


def get_mongodb_db():
    return db_clients.mongodb[db_clients.config.mongodb_db]


def get_qdrant_client():
    return db_clients.qdrant
