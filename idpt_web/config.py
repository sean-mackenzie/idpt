"""Configuration settings for IDPT Web."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Base paths
    base_dir: Path = Path(__file__).parent.parent
    storage_dir: Path = base_dir / "storage"

    # Storage subdirectories
    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / "results"

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "idpt_web.db"

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path}"

    # API settings
    api_v1_prefix: str = "/api/v1"

    # CORS settings
    cors_origins: list[str] = ["*"]

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    class Config:
        env_prefix = "IDPT_WEB_"


settings = Settings()
