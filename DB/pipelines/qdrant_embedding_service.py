"""
Qdrant Vector Embedding Service
Generates OpenAI embeddings for MongoDB repository data and stores in Qdrant
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import DatabaseManager
import openai
from qdrant_client.models import PointStruct, Distance, VectorParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantEmbeddingService:
    """Generates and manages vector embeddings for repository recommendations."""
    
    def __init__(self):
        """Initialize service with database connections."""
        self.db_manager = DatabaseManager()
        
        # MongoDB (source)
        self.mongo_db = self.db_manager.get_mongodb()["gitquery"]
        self.mongo_collection = self.mongo_db["recommendations"]
        
        # Qdrant (vector storage)
        self.qdrant_client = self.db_manager.get_qdrant()
        self.collection_name = "repository_embeddings"
        
        # OpenAI setup
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dim = 1536  # OpenAI embedding dimension
    
    def ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct configuration."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def generate_embedding_text(self, repo_data: Dict[str, Any]) -> str:
        """Generate text representation of repository for embedding."""
        parts = []
        
        # Repository name and owner
        if repo_data.get("repository", {}).get("fullName"):
            parts.append(f"Repository: {repo_data['repository']['fullName']}")
        
        # Description
        description = repo_data.get("repository", {}).get("description", "")
        if description:
            parts.append(f"Description: {description}")
        
        # Primary language
        primary_lang = repo_data.get("metadata", {}).get("primaryLanguage", "")
        if primary_lang and primary_lang != "Unknown":
            parts.append(f"Primary Language: {primary_lang}")
        
        # Languages
        languages = repo_data.get("metadata", {}).get("languages", [])
        if languages:
            parts.append(f"Languages: {', '.join(languages)}")
        
        # Topics
        topics = repo_data.get("metadata", {}).get("topics", [])
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")
        
        # License
        license_info = repo_data.get("metadata", {}).get("license", "")
        if license_info and license_info != "No License":
            parts.append(f"License: {license_info}")
        
        # Metrics (for context)
        metrics = repo_data.get("metrics", {})
        stars = metrics.get("stars", 0)
        if stars > 0:
            parts.append(f"Stars: {stars}")
        
        return " | ".join(parts)
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector using OpenAI."""
        try:
            response = openai.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def fetch_unembedded_repositories(self, batch_size: int = 50) -> List[Dict[str, Any]]:
        """Fetch repositories from MongoDB that need embeddings."""
        try:
            # Get repositories without embeddings or updated recently
            cursor = self.mongo_collection.find({
                "isActive": True
            }).limit(batch_size)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching repositories: {e}")
            return []
    
    def store_embedding(self, repo_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Store embedding in Qdrant."""
        try:
            point = PointStruct(
                id=hash(repo_id) & 0x7FFFFFFF,  # Convert to positive int
                vector=embedding,
                payload={
                    "repo_id": repo_id,
                    "owner": metadata.get("owner", ""),
                    "name": metadata.get("name", ""),
                    "full_name": metadata.get("fullName", ""),
                    "primary_language": metadata.get("primaryLanguage", ""),
                    "languages": metadata.get("languages", []),
                    "topics": metadata.get("topics", []),
                    "stars": metadata.get("stars", 0),
                    "quality_score": metadata.get("quality_score", 0.0),
                    "embedded_at": datetime.utcnow().isoformat()
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
        except Exception as e:
            logger.error(f"Error storing embedding for {repo_id}: {e}")
            return False
    
    def update_mongo_with_embedding_status(self, repo_id: str, success: bool):
        """Update MongoDB document with embedding status."""
        try:
            self.mongo_collection.update_one(
                {"_id": repo_id},
                {
                    "$set": {
                        "hasEmbedding": success,
                        "lastEmbeddedAt": datetime.utcnow().isoformat()
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error updating MongoDB status for {repo_id}: {e}")
    
    def run(self, batch_size: int = 50):
        """Run the embedding generation pipeline."""
        logger.info("Starting Qdrant embedding service")
        start_time = datetime.now()
        
        # Ensure collection exists
        self.ensure_collection_exists()
        
        # Fetch repositories
        repositories = self.fetch_unembedded_repositories(batch_size=batch_size)
        logger.info(f"Processing {len(repositories)} repositories")
        
        success_count = 0
        for repo in repositories:
            try:
                repo_id = repo["_id"]
                
                # Generate embedding text
                text = self.generate_embedding_text(repo)
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                if not embedding:
                    continue
                
                # Store in Qdrant
                metadata = {
                    "owner": repo.get("repository", {}).get("owner", ""),
                    "name": repo.get("repository", {}).get("name", ""),
                    "fullName": repo.get("repository", {}).get("fullName", ""),
                    "primaryLanguage": repo.get("metadata", {}).get("primaryLanguage", ""),
                    "languages": repo.get("metadata", {}).get("languages", []),
                    "topics": repo.get("metadata", {}).get("topics", []),
                    "stars": repo.get("metrics", {}).get("stars", 0),
                    "quality_score": repo.get("quality_score", 0.0)
                }
                
                stored = self.store_embedding(repo_id, embedding, metadata)
                
                # Update MongoDB
                self.update_mongo_with_embedding_status(repo_id, stored)
                
                if stored:
                    success_count += 1
                    logger.info(f"Generated embedding for {repo_id}")
                
            except Exception as e:
                logger.error(f"Error processing repository: {e}")
                continue
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Embedding service completed: {success_count}/{len(repositories)} embeddings generated in {elapsed:.2f}s")
        
        return success_count


def main():
    """Main entry point for embedding service."""
    service = QdrantEmbeddingService()
    service.run(batch_size=50)


if __name__ == "__main__":
    main()
