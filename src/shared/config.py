import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field
from typing import List, Optional

# Resolve the docker .env relative to this file so it is found regardless of
# the working directory (important for local dev where there is no root .env).
_DOCKER_ENV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../infrastructure/docker/.env")
)


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
    web_url: str = "http://web:8080"

    # Security
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400

    # API Keys
    # API keys - read from new environment variable names (preferred)
    mongodb_api_key: Optional[str] = Field(None, validation_alias="APIKEY_MONGODB")
    redis_api_key: Optional[str] = Field(None, validation_alias="APIKEY_REDIS")
    qdrant_api_key: Optional[str] = Field(None, validation_alias="APIKEY_QDRANT")
    mcp_api_key: Optional[str] = Field(None, validation_alias="APIKEY_MCP")
    # Cosmos support removed; no emulator API key

    # CORS - when using credentials, must specify exact origins (not "*")
    # For development, allow localhost on common ports
    allowed_origins: str = (
        "http://localhost:8080,http://localhost:80,http://localhost:3000,http://127.0.0.1:8080"
    )

    # Admin seed user - created on gateway startup if not already present.
    # Set WEB_ADMIN_EMAIL (and the other two) to enable seeding.
    web_admin_email: Optional[str] = Field(None, validation_alias="WEB_ADMIN_EMAIL")
    web_admin_password: Optional[str] = Field(
        None, validation_alias="WEB_ADMIN_PASSWORD"
    )
    web_admin_username: str = Field("admin", validation_alias="WEB_ADMIN_USERNAME")

    # Rate limiting
    # Reads from gateway-scoped env secrets; falls back to legacy names.
    # Set requests to 0 to disable (uncap) rate limiting.
    rate_limit_requests: int = Field(
        0,
        validation_alias=AliasChoices(
            "GATEWAY_RATE_LIMIT_REQUESTS",
            "RATE_LIMIT_REQUESTS",
        ),
    )
    rate_limit_window: int = Field(
        60,
        validation_alias=AliasChoices(
            "GATEWAY_RATE_LIMIT_WINDOW_SECONDS",
            "RATE_LIMIT_WINDOW",
        ),
    )

    # Logging
    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    model_config = SettingsConfigDict(
        # Check root .env first (Docker passes vars via environment, so the
        # path not existing is fine), then fall back to infrastructure/docker/.env
        # so local dev picks up credentials without needing a root-level .env.
        env_file=(".env", _DOCKER_ENV),
        env_prefix="",
        extra="ignore",
    )


settings = Settings()
