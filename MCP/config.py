"""Configuration settings for MCP server."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MCP Server settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Server settings
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000
    log_level: str = "INFO"

    # API Keys
    openai_api_key: str = ""

    # Database
    database_url: str = ""

    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:8080"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


settings = Settings()

