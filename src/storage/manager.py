"""
Database clients for Git-Query application.
Provides unified access to all database services.
"""

from typing import Optional

# Import database configuration
from src.storage.db_config import DatabaseConfig, db_clients


class DatabaseManager:
    """Central database manager for all Git-Query services."""

    def __init__(self):
        """Initialize database connections."""
        self.config = DatabaseConfig.from_env()
        self._clients = db_clients

    def get_mongodb(self):
        """Get MongoDB client."""
        return self._clients.mongodb

    def get_cosmos(self):
        """Get Cosmos DB client."""
        return self._clients.cosmos

    def get_qdrant(self):
        """Get Qdrant client."""
        return self._clients.qdrant

    def get_redis(self):
        """Get Redis client."""
        return self._clients.redis

    def close_all(self):
        """Close all database connections."""
        self._clients.close_all()


# Global database manager instance
db_manager = DatabaseManager()


# Convenience functions
def get_mongodb_db(db_name: Optional[str] = None):
    """
    Get MongoDB database instance.

    Args:
        db_name: Database name (defaults to configured database)

    Returns:
        MongoDB database instance
    """
    if db_name:
        return db_manager.get_mongodb()[db_name]
    return db_manager.get_mongodb()[db_manager.config.mongodb_db]


def get_cosmos_db(db_name: Optional[str] = None):
    """
    Get Cosmos DB database instance.

    Args:
        db_name: Database name (defaults to configured database)

    Returns:
        Cosmos DB database instance
    """
    if db_name:
        return db_manager.get_cosmos()[db_name]
    return db_manager.get_cosmos()[db_manager.config.cosmos_db_name]


def get_qdrant_client():
    """
    Get Qdrant client.

    Returns:
        Qdrant client instance
    """
    return db_manager.get_qdrant()


def get_redis_client():
    """
    Get Redis client.

    Returns:
        Redis client instance
    """
    return db_manager.get_redis()


def get_postgres_conn():
    """
    Get PostgreSQL connection from pool.

    Returns:
        PostgreSQL connection
    """
    return db_manager.get_postgres().getconn()


def return_postgres_conn(conn):
    """
    Return PostgreSQL connection to pool.

    Args:
        conn: PostgreSQL connection to return
    """
    db_manager.get_postgres().putconn(conn)
