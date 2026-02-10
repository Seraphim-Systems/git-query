"""Data ingestion pipeline - fetches raw data from MongoDB"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from processing.config import settings

logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles fetching and saving data to/from MongoDB"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.source_collection = db[settings.source_collection]
        self.dest_collection = db[settings.dest_collection]
    
    async def fetch_unprocessed(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch unprocessed records from source collection
        
        Args:
            limit: Maximum number of records to fetch
            
        Returns:
            List of unprocessed records
        """
        query = {
            "$or": [
                {"processing_status": {"$exists": False}},
                {"processing_status": "pending"}
            ]
        }
        
        cursor = self.source_collection.find(query).limit(limit)
        records = await cursor.to_list(length=limit)
        
        logger.info(f"Fetched {len(records)} unprocessed records")
        return records
    
    async def save_cleaned(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Save cleaned records to destination collection
        
        Args:
            records: List of cleaned records
            
        Returns:
            List of inserted IDs
        """
        if not records:
            return []
        
        # Add metadata
        for record in records:
            record["cleaned_at"] = datetime.utcnow()
            record["source"] = "processing_pipeline"
        
        result = await self.dest_collection.insert_many(records)
        
        logger.info(f"Saved {len(result.inserted_ids)} cleaned records")
        return [str(id) for id in result.inserted_ids]
    
    async def mark_as_processed(self, record_ids: List):
        """
        Mark records as processed in source collection
        
        Args:
            record_ids: List of record IDs to mark
        """
        if not record_ids:
            return
        
        result = await self.source_collection.update_many(
            {"_id": {"$in": record_ids}},
            {
                "$set": {
                    "processing_status": "completed",
                    "processed_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Marked {result.modified_count} records as processed")
    
    async def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        pipeline = [
            {
                "$group": {
                    "_id": "$processing_status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        results = await self.source_collection.aggregate(pipeline).to_list(None)
        
        stats = {
            "total": 0,
            "pending": 0,
            "completed": 0,
            "failed": 0
        }
        
        for result in results:
            status = result["_id"] or "pending"
            stats[status] = result["count"]
            stats["total"] += result["count"]
        
        return stats