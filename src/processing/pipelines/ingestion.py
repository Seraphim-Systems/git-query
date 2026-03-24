"""Data ingestion pipeline - fetches raw data from MongoDB"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

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
        
        # Upsert by stable identifier so reruns are safe.
        now = datetime.utcnow()
        operations = []
        saved_ids: List[str] = []

        for record in records:
            repo_id = record.get("repo_id")
            full_name = record.get("full_name")
            name = record.get("name")

            if repo_id:
                stable_id = str(repo_id)
            elif full_name:
                stable_id = str(full_name)
            elif name:
                stable_id = str(name)
            else:
                logger.debug("Skipping cleaned record without identifier")
                continue

            record["_id"] = stable_id
            record["cleaned_at"] = now
            record["source"] = "processing_pipeline"

            operations.append(
                UpdateOne(
                    {"_id": stable_id},
                    {
                        "$set": record,
                        "$setOnInsert": {"created_at_pipeline": now},
                    },
                    upsert=True,
                )
            )
            saved_ids.append(stable_id)

        if not operations:
            return []

        result = await self.dest_collection.bulk_write(operations, ordered=False)

        logger.info(
            "Upserted cleaned records: matched=%s modified=%s upserted=%s",
            result.matched_count,
            result.modified_count,
            result.upserted_count,
        )
        return saved_ids
    
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