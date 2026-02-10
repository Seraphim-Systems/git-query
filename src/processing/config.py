"""Processing service configuration"""

from pydantic_settings import BaseSettings
from typing import Optional

class ProcessingSettings(BaseSettings):
    # Service
    log_level: str = "INFO"
    batch_size: int = 100
    processing_interval: int = 60  # seconds
    
    # MongoDB (source)
    mongodb_url: str
    mongodb_db: str = "gitquery"
    source_collection: str = "raw_repositories"
    
    # MongoDB (destination - cleaned data)
    dest_collection: str = "repositories"
    
    # Qdrant (vector storage)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    vector_collection: str = "repository_embeddings"
    
    # Redis (coordination)
    redis_url: str
    
    # Processing options
    min_description_length: int = 10
    max_description_length: int = 5000
    required_fields: list[str] = [
        "repo_id", "full_name", "description", "language"
    ]
    
    class Config:
        env_file = ".env"

settings = ProcessingSettings()