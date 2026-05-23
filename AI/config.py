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
_ENV_PATH: Path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)
BASE_DIR: Path = Path(__file__).resolve().parent          
ML_DIR: Path = BASE_DIR.parent / "Ml"                    


class _Settings:
    def __init__(self) -> None:
        self.MODEL_REGISTRY: dict[str, Path] = {}

        _registry_raw: dict[str, str] = {
            "logistic_regression": os.getenv(
                "MODEL_PATH_LOGISTIC",
                "../Ml/model_pipeline.pkl(logistic)",
            ),
            "linear_svm": os.getenv(
                "MODEL_PATH_LINEAR_SVM",
                "../Ml/model_pipeline.pkl",
            ),
        }

        for name, raw_path in _registry_raw.items():
            p = Path(raw_path)
            if not p.is_absolute():
                p = (BASE_DIR / p).resolve()
            self.MODEL_REGISTRY[name] = p

        self.DEFAULT_MODEL_NAME: str = os.getenv(
            "DEFAULT_MODEL_NAME", "logistic_regression"
        )

        # Legacy single model path (backward compatibility)
        _raw_model: str = os.getenv("MODEL_PATH", "../Ml/model_pipeline.pkl(logistic)")
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
        self.MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "10000"))

        # Server
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))


settings = _Settings()
