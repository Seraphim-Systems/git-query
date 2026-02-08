"""Gateway configuration."""

from pydantic_settings import BaseSettings
from typing import List


class GatewaySettings(BaseSettings):
    """API Gateway settings."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 86400  # 24 hours
    cache_ttl: int = 3600  # 1 hour

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "gitquery"

    # Backend services
    mcp_server_url: str = "http://mcp-server:8001"
    recommender_url: str = "http://recommender-api:8002"

    # Security
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400  # 24 hours

    # API Keys (per-service authentication)
    mongodb_api_key: str = "mongodb-dev-key-change-in-prod"
    redis_api_key: str = "redis-dev-key-change-in-prod"
    qdrant_api_key: str = "qdrant-dev-key-change-in-prod"
    mcp_api_key: str = "mcp-dev-key-change-in-prod"

    # CORS
    allowed_origins: str = "*"

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        env_prefix = "GATEWAY_"


settings = GatewaySettings()
