"""Main data processor - orchestrates the cleaning pipeline"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis
from qdrant_client import QdrantClient

from processing.config import settings
from processing.pipelines.ingestion import DataIngestion
from processing.pipelines.preparation import run_preparation_batch
from processing.pipelines.transformation import DataTransformer
from processing.pipelines.vectorization import DataVectorizer

logger = logging.getLogger(__name__)

class DataProcessor:
    """Orchestrates the data processing pipeline"""
    
    def __init__(self):
        self.settings = settings
        
        # Database clients
        self.mongo_client = None
        self.redis_client = None
        self.qdrant_client = None
        
        # Pipeline components
        self.ingestion = None
        self.transformer = None
        self.vectorizer = None
        
        self.running = False
        
    async def initialize(self):
        """Initialize database connections and pipeline components"""
        logger.info("Initializing Data Processor...")
        
        # MongoDB connection
        self.mongo_client = AsyncIOMotorClient(settings.mongodb_url)
        self.db = self.mongo_client[settings.mongodb_db]
        logger.info("✓ MongoDB connected")
        
        # Redis connection
        self.redis_client = Redis.from_url(
            settings.redis_url, 
            decode_responses=True
        )
        await self.redis_client.ping()
        logger.info("✓ Redis connected")
        
        # Qdrant connection
        self.qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        logger.info("✓ Qdrant connected")
        
        # Ensure database collections and indexes exist
        from processing.database import DatabaseHelper
        
        # Create MongoDB indexes
        await DatabaseHelper.ensure_indexes(self.db)
        logger.info("✓ MongoDB indexes verified")
        
        # Create Qdrant collection
        DatabaseHelper.ensure_qdrant_collection(
            self.qdrant_client,
            settings.vector_collection,
            vector_size=384  # all-MiniLM-L6-v2 embedding size
        )
        logger.info("✓ Qdrant collection verified")
        
        # Initialize pipeline components
        self.ingestion = DataIngestion(self.db)
        self.transformer = DataTransformer()
        self.vectorizer = DataVectorizer(self.qdrant_client)
        
        logger.info("Data Processor initialized successfully")
        
    async def process_batch(self) -> Dict[str, int]:
        """Process a single batch of records"""
        stats = {
            "fetched": 0,
            "cleaned": 0,
            "saved": 0,
            "vectorized": 0,
            "errors": 0
        }
        
        try:
            # 1-3. Fetch raw data, clean it, and persist cleaned records.
            prep_result = await run_preparation_batch(
                ingestion=self.ingestion,
                transformer=self.transformer,
                limit=settings.batch_size,
                mark_processed=True,
            )
            stats.update(prep_result["stats"])
            cleaned_records = prep_result["cleaned_records"]

            if stats["fetched"] == 0:
                logger.debug("No unprocessed records found")
                return stats
            
            logger.info(
                "Prepared batch: fetched=%s cleaned=%s saved=%s errors=%s",
                stats["fetched"],
                stats["cleaned"],
                stats["saved"],
                stats["errors"],
            )
            
            # 4. Generate embeddings and index
            for record in cleaned_records:
                try:
                    await self.vectorizer.vectorize_and_index(record)
                    stats["vectorized"] += 1
                except Exception as e:
                    logger.error(f"Error vectorizing record: {e}")
                    stats["errors"] += 1
            
            logger.info(f"Batch complete: {stats}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            stats["errors"] += 1
        
        return stats
    
    async def mark_as_failed(self, record: Dict[str, Any]):
        """Mark a record as failed processing"""
        try:
            await self.db[settings.source_collection].update_one(
                {"_id": record["_id"]},
                {
                    "$set": {
                        "processing_status": "failed",
                        "processing_error": str(record.get("error", "Unknown")),
                        "failed_at": datetime.utcnow()
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to mark record as failed: {e}")
    
    async def run(self):
        """Main processing loop"""
        await self.initialize()
        self.running = True
        
        logger.info("Starting processing loop...")
        
        while self.running:
            try:
                stats = await self.process_batch()
                
                # Store stats in Redis
                await self.redis_client.hset(
                    "processing:stats",
                    mapping={
                        "last_run": datetime.utcnow().isoformat(),
                        **{k: str(v) for k, v in stats.items()}
                    }
                )
                
                # Wait before next batch
                if stats["fetched"] == 0:
                    # No records found, wait longer
                    await asyncio.sleep(settings.processing_interval)
                else:
                    # Records processed, check for more immediately
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(settings.processing_interval)
    
    async def cleanup(self):
        """Clean up connections"""
        logger.info("Cleaning up...")
        
        if self.mongo_client:
            self.mongo_client.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.qdrant_client:
            self.qdrant_client.close()
        
        logger.info("Cleanup complete")