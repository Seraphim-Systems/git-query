"""Shared database utilities"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from redis.asyncio import Redis
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class DatabaseHelper:
    """Helper class for database operations"""
    
    @staticmethod
    async def ensure_indexes(db: AsyncIOMotorDatabase):
        """Create indexes for processing collections"""
        try:
            await db.repositories.create_index("processing_status")
            await db.repositories.create_index("scraped_at")
            await db.repositories.create_index("repo_id", unique=True)
            await db.repositories.create_index("full_name")
            await db.repositories.create_index("language")
            await db.repositories.create_index([("stars", -1)])
            await db.repositories.create_index([("search_text", "text")], language_override="none")
            
            logger.info("✓ Database indexes created")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    @staticmethod
    def ensure_qdrant_collection(
        client: QdrantClient, 
        collection_name: str,
        vector_size: int = 384
    ):
        """Ensure Qdrant collection exists"""
        from qdrant_client.models import Distance, VectorParams
        
        try:
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if not exists:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error with Qdrant collection: {e}")