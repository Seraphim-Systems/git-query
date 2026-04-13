"""
Database manager convenience wrapper.

This module exposes a thin `DatabaseManager` that delegates to the canonical
database configuration and client singletons in `src.db.config` and
`src.db.clients` so storage code can import a consistent API without
duplicating environment loading.
"""

from typing import Optional

from src.db.config import db_clients


class DatabaseManager:
    """Central database manager for all Git-Query services.

    This manager intentionally avoids re-reading environment variables and
    instead delegates to the shared `db_clients` singleton (which itself is
    configured from shared settings). This keeps configuration in one place
    and avoids duplication.
    """

    def __init__(self):
        self._clients = db_clients
        self.config = getattr(db_clients, "config", None)

    def get_mongodb(self):
        """Get MongoDB client instance (pymongo.MongoClient)."""
        return self._clients.mongodb

    def get_qdrant(self):
        """Get Qdrant client instance or None."""
        return self._clients.qdrant

    def get_redis(self):
        """Get Redis client from `src.db.clients.get_redis_client()`."""
        return self._clients.redis

    def close_all(self):
        """Close underlying clients managed by `db_clients`."""
        try:
            self._clients.close_all()
        except Exception:
            # Some deployments may manage lifecycle separately; ignore errors.
            pass


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_mongodb_db(db_name: Optional[str] = None):
    if db_name:
        return db_manager.get_mongodb()[db_name]
    if db_manager.config and getattr(db_manager.config, "mongodb_db", None):
        return db_manager.get_mongodb()[db_manager.config.mongodb_db]
    # Fallback: return the default database handle
    return db_manager.get_mongodb()


def get_qdrant_client():
    return db_manager.get_qdrant()


def get_redis_client():
    return db_manager.get_redis()


def get_postgres_conn():
    """Postgres support not configured in this manager.

    If you require Postgres, wire it into `src.db.config` and expose it from
    the `db_clients` singleton. For now this raises to make missing usage
    explicit.
    """
    raise NotImplementedError("Postgres is not configured in db_clients")


def return_postgres_conn(conn):
    raise NotImplementedError("Postgres is not configured in db_clients")
