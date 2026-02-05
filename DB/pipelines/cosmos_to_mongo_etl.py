"""
Cosmos DB to MongoDB ETL Pipeline
Transforms raw repository data from Cosmos DB into cleaned MongoDB format
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseManager
from pymongo import UpdateOne

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CosmosToMongoETL:
    """ETL pipeline for transforming Cosmos data to MongoDB."""
    
    def __init__(self):
        """Initialize ETL with database connections."""
        self.db_manager = DatabaseManager()
        
        # Cosmos DB (source)
        self.cosmos_db = self.db_manager.get_cosmos()["gitquery_cosmos"]
        self.cosmos_collection = self.cosmos_db["repository_activity"]
        
        # MongoDB (destination)
        self.mongo_db = self.db_manager.get_mongodb()["gitquery"]
        self.mongo_collection = self.mongo_db["recommendations"]
    
    def clean_repository_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean and transform raw Cosmos data for MongoDB."""
        try:
            # Validation
            if not raw_data.get("name") or not raw_data.get("owner"):
                logger.warning(f"Invalid repository data: missing name or owner")
                return None
            
            # Transform to clean format
            cleaned = {
                "_id": f"{raw_data['owner']}/{raw_data['name']}",
                "repository": {
                    "owner": raw_data["owner"],
                    "name": raw_data["name"],
                    "fullName": raw_data.get("nameWithOwner", f"{raw_data['owner']}/{raw_data['name']}"),
                    "description": raw_data.get("description", "").strip(),
                    "url": f"https://github.com/{raw_data['owner']}/{raw_data['name']}"
                },
                "metrics": {
                    "stars": max(0, raw_data.get("stars", 0)),
                    "forks": max(0, raw_data.get("forks", 0)),
                    "watchers": max(0, raw_data.get("watchers", 0)),
                    "openIssues": max(0, raw_data.get("issues", 0)),
                    "pullRequests": max(0, raw_data.get("pullRequests", 0)),
                    "diskSizeMB": round(raw_data.get("diskUsageKb", 0) / 1024, 2)
                },
                "metadata": {
                    "primaryLanguage": raw_data.get("primaryLanguage", "Unknown"),
                    "languages": raw_data.get("languages", []),
                    "languageCount": raw_data.get("languageCount", 0),
                    "topics": raw_data.get("topics", []),
                    "topicCount": raw_data.get("topicCount", 0),
                    "license": raw_data.get("license", "No License"),
                    "codeOfConduct": raw_data.get("codeOfConduct", "None")
                },
                "status": {
                    "isArchived": raw_data.get("isArchived", False),
                    "isFork": raw_data.get("isFork", False),
                    "forkingAllowed": raw_data.get("forkingAllowed", True),
                    "parent": raw_data.get("parent")
                },
                "dates": {
                    "createdAt": raw_data.get("createdAt"),
                    "lastPushedAt": raw_data.get("pushedAt"),
                    "lastScrapedAt": raw_data.get("scrapedAt", datetime.utcnow().isoformat())
                },
                "processedAt": datetime.utcnow().isoformat(),
                "isActive": not raw_data.get("isArchived", False),
                "quality_score": self.calculate_quality_score(raw_data)
            }
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning repository data: {e}")
            return None
    
    def calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate repository quality score (0-100)."""
        try:
            score = 0.0
            
            # Stars contribution (max 40 points)
            stars = data.get("stars", 0)
            if stars > 10000:
                score += 40
            elif stars > 1000:
                score += 30
            elif stars > 100:
                score += 20
            elif stars > 10:
                score += 10
            
            # Activity contribution (max 20 points)
            if data.get("pushedAt"):
                from dateutil import parser
                last_push = parser.parse(data["pushedAt"])
                days_since_push = (datetime.utcnow() - last_push.replace(tzinfo=None)).days
                if days_since_push < 7:
                    score += 20
                elif days_since_push < 30:
                    score += 15
                elif days_since_push < 90:
                    score += 10
                elif days_since_push < 180:
                    score += 5
            
            # Documentation (max 15 points)
            if data.get("description"):
                score += 10
            if data.get("license"):
                score += 5
            
            # Community (max 15 points)
            forks = data.get("forks", 0)
            if forks > 100:
                score += 10
            elif forks > 10:
                score += 5
            
            if data.get("codeOfConduct") and data["codeOfConduct"] != "None":
                score += 5
            
            # Technical diversity (max 10 points)
            lang_count = data.get("languageCount", 0)
            topic_count = data.get("topicCount", 0)
            if lang_count > 3:
                score += 5
            if topic_count > 5:
                score += 5
            
            return round(min(score, 100), 2)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def extract_from_cosmos(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """Extract raw data from Cosmos DB."""
        try:
            # Get unprocessed or recently updated repositories
            cursor = self.cosmos_collection.find({}).limit(batch_size)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error extracting from Cosmos: {e}")
            return []
    
    def load_to_mongo(self, cleaned_data: List[Dict[str, Any]]) -> int:
        """Load cleaned data into MongoDB."""
        if not cleaned_data:
            return 0
        
        try:
            # Bulk upsert to MongoDB
            operations = [
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": doc},
                    upsert=True
                )
                for doc in cleaned_data
            ]
            
            result = self.mongo_collection.bulk_write(operations)
            return result.upserted_count + result.modified_count
            
        except Exception as e:
            logger.error(f"Error loading to MongoDB: {e}")
            return 0
    
    def run(self, batch_size: int = 100):
        """Run the ETL pipeline."""
        logger.info("Starting Cosmos to MongoDB ETL pipeline")
        start_time = datetime.now()
        
        # Extract
        raw_data = self.extract_from_cosmos(batch_size=batch_size)
        logger.info(f"Extracted {len(raw_data)} records from Cosmos DB")
        
        # Transform
        cleaned_data = []
        for raw_doc in raw_data:
            cleaned = self.clean_repository_data(raw_doc)
            if cleaned:
                cleaned_data.append(cleaned)
        
        logger.info(f"Transformed {len(cleaned_data)} records")
        
        # Load
        loaded = self.load_to_mongo(cleaned_data)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"ETL completed: {loaded} records loaded in {elapsed:.2f}s")
        
        return loaded


def main():
    """Main entry point for ETL."""
    etl = CosmosToMongoETL()
    etl.run(batch_size=100)


if __name__ == "__main__":
    main()
