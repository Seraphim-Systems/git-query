"""Configuration for the recommendation system."""

import logging
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

_config_logger = logging.getLogger(__name__)

class RecommenderSettings(BaseSettings):
    """Settings for the recommender service."""

    # Server settings
    recommender_host: str = "0.0.0.0"
    recommender_port: int = 8095
    log_level: str = "INFO"

    # Database connections
    mongodb_url: str = "mongodb://localhost:27017/gitquery?authSource=admin"
    redis_url: str = "redis://localhost:6379"
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_url: str = ""  # computed from qdrant_host + qdrant_http_port if not set directly
    qdrant_api_key: Optional[str] = None

    @model_validator(mode="after")
    def _build_qdrant_url(self) -> "RecommenderSettings":
        if not self.qdrant_url:
            self.qdrant_url = f"http://{self.qdrant_host}:{self.qdrant_http_port}"
        return self

    # API Keys
    embedding_api_key: Optional[str] = None

    # Model paths
    model_path: str = "./models"
    checkpoint_path: str = "./models/checkpoints"
    eval_path: str = "./models/eval"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval settings
    hybrid_search_top_k: int = 100
    rerank_top_k: int = 20
    final_top_k: int = 10
    rrf_k: int = 60  # Reciprocal Rank Fusion constant; smaller = sharper rank differences

    # Embeddings
    embedding_dimension: int = 384
    batch_size: int = 32
    inference_batch_size: int = 32

    # Personalization
    enable_personalization: bool = True
    personalization_weight: float = 0.15  # How much to boost based on user prefs
    min_interactions_for_personalization: int = 5

    # Caching
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_cache: bool = True

    # A/B Testing
    ab_test_enabled: bool = True
    default_variant: str = "hybrid"

    # Collections
    repos_collection: str = "repositories"
    raw_repos_collection: str = "raw_repositories"
    interactions_collection: str = "user_interactions"
    user_prefs_collection: str = "user_preferences"
    models_collection: str = "ml_models"
    ab_tests_collection: str = "ab_tests"

    # Qdrant collections
    qdrant_repos_collection: str = "repositories_embeddings"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = RecommenderSettings()

if not settings.embedding_api_key:
    _config_logger.warning(
        "EMBEDDING_API_KEY not set. Embedding functionality may be limited."
    )
