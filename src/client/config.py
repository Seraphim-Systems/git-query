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
    mcp_server_url: str = "http://localhost:8090"

    # OpenAI - reads OPENAI_API_KEY or APP_OPENAI_API_KEY from env
    openai_api_key: str = ""
    app_openai_api_key: str = ""
    model_name: str = "gpt-4o"

    # Databases
    mongodb_url: str = ""
    redis_url: str = ""

    # Logging
    log_level: str = "INFO"

    @property
    def resolved_openai_api_key(self) -> str:
        """Return whichever OpenAI key is set."""
        return self.openai_api_key or self.app_openai_api_key


settings = Settings()
