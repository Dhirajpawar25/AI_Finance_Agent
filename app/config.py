"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object. Values come from .env or environment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "AI Financial Agent"
    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: str = "./data"

    # Telegram
    telegram_bot_token: str = ""
    allowed_telegram_ids: str = ""  # comma-separated
    use_webhook: bool = False
    webhook_path: str = "/webhook/telegram"

    # AI providers
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Database
    database_url: str = "sqlite:///./financial_agent.db"

    # Scheduling
    bot_timezone: str = "Asia/Kolkata"
    default_briefing_time: str = "08:30"
    alert_check_interval_minutes: int = 5
    enable_scheduler: bool = True

    # Financial data
    finnhub_api_key: str = ""
    alpha_vantage_api_key: str = ""
    sec_user_agent: str = "FinancialAgent demo@example.com"

    # Google integrations
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/integrations/google/callback"
    google_credentials_file: str = "credentials.json"

    @property
    def allowed_user_ids(self) -> set[int]:
        """Parsed set of allowed Telegram user ids."""
        raw = self.allowed_telegram_ids.strip()
        if not raw:
            return set()  # empty means allow everyone (demo mode)
        return {int(part.strip()) for part in raw.split(",") if part.strip()}

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_any_ai(self) -> bool:
        return self.has_openai or self.has_anthropic or self.has_gemini


@lru_cache
def get_settings() -> Settings:
    return Settings()