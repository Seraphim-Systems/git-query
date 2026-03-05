from pydantic_settings import BaseSettings
from pydantic import Field
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
    mcp_server_url: str = "http://mcp-server:8090"
    recommender_url: str = "http://recommender:8095"

    # Security
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400

    # API Keys
    # API keys - read from new environment variable names (preferred)
    mongodb_api_key: Optional[str] = Field(None, env="APIKEY_MONGODB")
    redis_api_key: Optional[str] = Field(None, env="APIKEY_REDIS")
    qdrant_api_key: Optional[str] = Field(None, env="APIKEY_QDRANT")
    mcp_api_key: Optional[str] = Field(None, env="APIKEY_MCP")
    # Cosmos support removed; no emulator API key

    # CORS - when using credentials, must specify exact origins (not "*")
    # For development, allow localhost on common ports
    allowed_origins: str = "http://localhost:8080,http://localhost:80,http://localhost:3000,http://127.0.0.1:8080"

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
        extra = "ignore"


settings = Settings()
