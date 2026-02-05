"""
Database Configuration Module
Centralized configuration for all database connections in Git-Query chatbot
"""
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Configuration for all databases used in the application"""
    
    # MongoDB (General Database)
    mongodb_url: str
    mongodb_host: str
    mongodb_port: int
    mongodb_user: str
    mongodb_password: str
    mongodb_db: str
    
    # Azure Cosmos DB (Large-scale data)
    cosmos_db_url: str
    cosmos_db_key: Optional[str] = None
    cosmos_db_name: str = "gitquery_cosmos"
    
    # Qdrant (Vector Database)
    qdrant_url: str
    qdrant_host: str
    qdrant_port: int
    qdrant_api_key: Optional[str] = None
    
    # Redis (Cache Database)
    redis_url: str
    redis_host: str
    redis_port: int
    redis_password: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables"""
        
        # MongoDB
        mongodb_host = os.getenv("MONGO_HOST", "localhost")
        mongodb_port = int(os.getenv("MONGO_PORT", "27017"))
        mongodb_user = os.getenv("MONGO_USER", "admin")
        mongodb_password = os.getenv("MONGO_PASSWORD", "mongopass")
        mongodb_db = os.getenv("MONGO_DB", "gitquery")
        mongodb_url = os.getenv(
            "MONGODB_URL",
            f"mongodb://{mongodb_user}:{mongodb_password}@{mongodb_host}:{mongodb_port}/{mongodb_db}?authSource=admin"
        )
        
        # Cosmos DB
        cosmos_db_url = os.getenv("COSMOS_DB_URL", "https://localhost:8081")
        cosmos_db_key = os.getenv("COSMOS_DB_KEY")
        cosmos_db_name = os.getenv("COSMOS_DB_NAME", "gitquery_cosmos")
        
        # Qdrant
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_HTTP_PORT", "6333"))
        qdrant_url = os.getenv("QDRANT_URL", f"http://{qdrant_host}:{qdrant_port}")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Redis
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        redis_url = os.getenv("REDIS_URL", f"redis://{redis_host}:{redis_port}")
        
        return cls(
            mongodb_url=mongodb_url,
            mongodb_host=mongodb_host,
            mongodb_port=mongodb_port,
            mongodb_user=mongodb_user,
            mongodb_password=mongodb_password,
            mongodb_db=mongodb_db,
            cosmos_db_url=cosmos_db_url,
            cosmos_db_key=cosmos_db_key,
            cosmos_db_name=cosmos_db_name,
            qdrant_url=qdrant_url,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=qdrant_api_key,
            redis_url=redis_url,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password
        )


class DatabaseClients:
    """Singleton class for managing database client connections"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = DatabaseConfig.from_env()
            self._mongodb_client = None
            self._cosmos_client = None
            self._qdrant_client = None
            self._redis_client = None
            self._initialized = True
    
    @property
    def mongodb(self):
        """Get MongoDB client (lazy initialization)"""
        if self._mongodb_client is None:
            from pymongo import MongoClient
            self._mongodb_client = MongoClient(self.config.mongodb_url)
        return self._mongodb_client
    
    @property
    def cosmos(self):
        """Get Azure Cosmos DB client (lazy initialization)"""
        if self._cosmos_client is None:
            from pymongo import MongoClient
            # Cosmos DB uses MongoDB API
            self._cosmos_client = MongoClient(
                self.config.cosmos_db_url,
                password=self.config.cosmos_db_key,
                ssl=True,
                tls=True,
                tlsAllowInvalidCertificates=True  # For emulator only
            )
        return self._cosmos_client
    
    @property
    def qdrant(self):
        """Get Qdrant client (lazy initialization)"""
        if self._qdrant_client is None:
            from qdrant_client import QdrantClient
            self._qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key
            )
        return self._qdrant_client
    
    @property
    def redis(self):
        """Get Redis client (lazy initialization)"""
        if self._redis_client is None:
            import redis
            self._redis_client = redis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                decode_responses=True
            )
        return self._redis_client
    
    def close_all(self):
        """Close all database connections"""
        if self._mongodb_client:
            self._mongodb_client.close()
        if self._cosmos_client:
            self._cosmos_client.close()
        if self._qdrant_client:
            self._qdrant_client.close()
        if self._redis_client:
            self._redis_client.close()


# Global instance
db_clients = DatabaseClients()


# Convenience functions
def get_mongodb_db():
    """Get MongoDB database instance"""
    return db_clients.mongodb[db_clients.config.mongodb_db]


def get_cosmos_db():
    """Get Cosmos DB database instance"""
    return db_clients.cosmos[db_clients.config.cosmos_db_name]


def get_qdrant_client():
    """Get Qdrant client instance"""
    return db_clients.qdrant


def get_redis_client():
    """Get Redis client instance"""
    return db_clients.redis
