from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración tipada del microservicio EchoFlow."""

    app_name: str = "EchoFlow Backend API"
    app_version: str = "0.1.2"
    app_env: str = "development"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5000",
        "http://localhost",
        "https://ingles-web-2b277.web.app",
        "https://ingles-web-2b277.firebaseapp.com",
        "*",
    ]

    # Google Gemini API (Free Tier)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # YouTube Transcript Extraction, Cookies, Proxy & Fallback
    youtube_cookies_content: Optional[str] = None
    youtube_cookies_file_path: Optional[str] = None
    youtube_proxy_url: Optional[str] = None
    enable_mock_fallback: bool = True
    allow_mock_fallback: bool = True

    # Guardarraíles pedagógicos de segmentación de subtítulos
    segment_min_words: int = 4
    segment_target_max_words: int = 18
    segment_hard_max_words: int = 26
    segment_max_duration_seconds: float = 12.0
    segment_acoustic_gap_seconds: float = 1.15
    segment_micro_merge_gap_seconds: float = 0.45

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def has_gemini_key(self) -> bool:
        return bool(
            self.gemini_api_key
            and self.gemini_api_key != "your_gemini_api_key_here"
        )

    @property
    def has_youtube_cookies(self) -> bool:
        if self.youtube_cookies_content and self.youtube_cookies_content.strip():
            return True
        if (
            self.youtube_cookies_file_path
            and Path(self.youtube_cookies_file_path).is_file()
        ):
            return True
        for candidate in (
            Path("cookies.txt"),
            Path("backend/cookies.txt"),
            Path(__file__).resolve().parent.parent.parent / "cookies.txt",
        ):
            if candidate.is_file():
                return True
        return False

    @property
    def has_youtube_proxy(self) -> bool:
        return bool(self.youtube_proxy_url and self.youtube_proxy_url.strip())

    @property
    def is_fallback_enabled(self) -> bool:
        return self.enable_mock_fallback and self.allow_mock_fallback


@lru_cache
def get_settings() -> Settings:
    return Settings()
