"""Configuration settings for ChatbotClient."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ChatbotClient settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MCP Server
    mcp_server_url: str = "http://localhost:8000"

    # OpenAI
    openai_api_key: str = ""
    model_name: str = "gpt-4"

    # Logging
    log_level: str = "INFO"


settings = Settings()
