"""Centralised configuration for the Sg-ai application.

Loads settings from environment variables (with ``.env`` file support)
so that **no paths or magic numbers are hardcoded** in the codebase.

Usage::

    from config import settings
    print(settings.MODEL_PATH)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the AI/ directory (next to this file)
_ENV_PATH: Path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------------------
# Resolved base directories
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent          # AI/
ML_DIR: Path = BASE_DIR.parent / "Ml"                    # Ml/


class _Settings:
    """Application settings resolved from environment variables.

    Attributes:
        MODEL_PATH: Absolute path to the trained ``.pkl`` model file.
        DATABASE_URL: SQLAlchemy database URL string.
        MAX_INPUT_LENGTH: Maximum allowed characters for tweet input.
        HOST: Server bind host.
        PORT: Server bind port.
    """

    def __init__(self) -> None:
        # Model path — resolve relative paths against AI/ directory
        _raw_model: str = os.getenv("MODEL_PATH", "../Ml/model_pipeline_LOGISTIK.pkl")
        _model_path = Path(_raw_model)
        if not _model_path.is_absolute():
            _model_path = (BASE_DIR / _model_path).resolve()
        self.MODEL_PATH: Path = _model_path

        # Database
        _raw_db: str = os.getenv("DATABASE_URL", "sqlite:///predictions.db")
        # For relative SQLite paths, resolve against AI/
        if _raw_db.startswith("sqlite:///") and not Path(_raw_db.replace("sqlite:///", "")).is_absolute():
            _db_file = (BASE_DIR / _raw_db.replace("sqlite:///", "")).resolve()
            _raw_db = f"sqlite:///{_db_file}"
        self.DATABASE_URL: str = _raw_db

        # Input validation
        self.MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "1000"))

        # Server
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))


settings = _Settings()
