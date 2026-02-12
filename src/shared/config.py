from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 80

    # Redis
    redis_url: str = "redis://localhost:6379"
    session_ttl: int = 86400
    cache_ttl: int = 3600

    # MongoDB
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db: str = "gitquery"

    # Backend services
    mcp_server_url: str = "http://mcp-server:8001"
    recommender_url: str = "http://recommender-api:8002"

    # Security
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400

    # API Keys
    mongodb_api_key: Optional[str] = None
    redis_api_key: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    mcp_api_key: Optional[str] = None

    # CORS
    allowed_origins: str = "*"

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        env_prefix = ""


settings = Settings()
