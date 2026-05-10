"""Sg-ai Inference API — FastAPI backend for tweet sentiment prediction.

This module exposes a REST API for performing NLP inference using a
pre-trained scikit-learn pipeline.  Predictions are automatically logged
to an SQLite database.

Usage::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from config import settings, ML_DIR, BASE_DIR
from database import (
    get_db,
    get_prediction_stats,
    get_recent_predictions,
    init_db,
    log_prediction,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger: logging.Logger = logging.getLogger("sg-ai")

# ---------------------------------------------------------------------------
# Ensure Ml/ is on sys.path so processing.py can be imported
# ---------------------------------------------------------------------------
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------
_model_pipeline: Optional[Any] = None
_preprocessor: Optional[Any] = None


# ---------------------------------------------------------------------------
# Request / Response schemas (Pydantic)
# ---------------------------------------------------------------------------
class TweetRequest(BaseModel):
    """Schema for the prediction request body.

    Attributes:
        tweet: Raw tweet text to classify (1 to MAX_INPUT_LENGTH characters).
    """

    tweet: str

    @field_validator("tweet")
    @classmethod
    def validate_tweet(cls, v: str) -> str:
        """Validate that the tweet is a non-empty string within length limit.

        Args:
            v: The raw tweet value.

        Returns:
            The stripped tweet string.

        Raises:
            ValueError: If the tweet is empty or exceeds the configured max length.
        """
        if not isinstance(v, str):
            raise ValueError("tweet must be a string")
        v = v.strip()
        if len(v) == 0:
            raise ValueError("tweet must not be empty")
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(
                f"tweet exceeds maximum length of {settings.MAX_INPUT_LENGTH} "
                f"characters (got {len(v)})"
            )
        return v


class PredictionResponse(BaseModel):
    """Schema for the prediction response body.

    Attributes:
        label: Predicted class label.
        confidence_score: Probability score for the predicted class.
        timestamp: ISO-8601 UTC timestamp of the prediction.
    """

    label: str
    confidence_score: float
    timestamp: str


class HealthResponse(BaseModel):
    """Schema for the health-check response.

    Attributes:
        status: Service health status string.
        model_loaded: Whether the model is loaded and ready.
        version: Application version string.
    """

    status: str
    model_loaded: bool
    version: str = "1.0.0"


class HistoryItem(BaseModel):
    """A single prediction record returned by ``/history``.

    Attributes:
        id: Record primary key.
        tweet_text: Original tweet.
        label: Predicted label.
        confidence: Confidence score.
        created_at: ISO timestamp of creation.
    """

    id: int
    tweet_text: str
    label: str
    confidence: float
    created_at: Optional[str] = None


class StatsResponse(BaseModel):
    total: int
    label_distribution: Dict[str, int]


class ErrorResponse(BaseModel):
    error: str
    detail: str

# Application lifespan — load model once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_pipeline, _preprocessor
    # 1. Initialise database
    try:
        init_db()
        logger.info("Database initialised successfully")
    except Exception:
        logger.exception("Failed to initialise database")
        raise

    # 2. Load model
    try:
        model_path: Path = settings.MODEL_PATH
        if not model_path.exists():
            logger.error("Model file not found at %s", model_path)
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please train and save the model first."
            )
        _model_pipeline = joblib.load(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    except FileNotFoundError:
        raise
    except Exception:
        logger.exception("Failed to load model")
        raise

    # 3. Instantiate preprocessor
    try:
        from processing import Preprocessing  # type: ignore[import-untyped]

        _preprocessor = Preprocessing(str(ML_DIR))
        logger.info("Preprocessor initialised successfully")
    except Exception:
        logger.exception("Failed to initialise preprocessor")
        raise

    yield 
    logger.info("Application shutting down")


# FastAPI application
app = FastAPI(
    title="Sg-ai -- Tweet Sentiment Predictor",
    description=(
        "NLP inference API for classifying tweets using a pre-trained "
        "Logistic Regression pipeline (TF-IDF + SMOTE)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Endpoints — Health
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return the health status of the API.

    Returns:
        A ``HealthResponse`` indicating whether the service and model are up.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_model_pipeline is not None,
    )


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend single-page application.

    Returns:
        The ``index.html`` file from the static directory, or redirects
        to ``/health`` if no frontend is deployed.
    """
    index_file = _STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    # Fallback — return health check JSON
    return HealthResponse(
        status="ok",
        model_loaded=_model_pipeline is not None,
    )


# ---------------------------------------------------------------------------
# Endpoints — Inference
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Inference"],
)
async def predict(request: TweetRequest) -> PredictionResponse:
    """Perform tweet sentiment prediction.

    Accepts a tweet string, preprocesses it, runs the ML pipeline,
    and returns the predicted label with a confidence score.
    The result is also logged to the SQLite database.

    Args:
        request: A ``TweetRequest`` containing the tweet text.

    Returns:
        A ``PredictionResponse`` with label, confidence_score, and timestamp.

    Raises:
        HTTPException: 503 if the model is not loaded.
        HTTPException: 500 on preprocessing / inference / database errors.
    """
    logger.info("Prediction request received (length=%d)", len(request.tweet))

    # Guard: model must be loaded
    if _model_pipeline is None or _preprocessor is None:
        logger.error("Prediction attempted but model is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later.",
        )

    # 1. Preprocess
    try:
        cleaned_text: str = _preprocessor.preprocess_full(request.tweet)
    except Exception as exc:
        logger.exception("Preprocessing failed for tweet")
        raise HTTPException(
            status_code=500,
            detail=f"Preprocessing error: {str(exc)}",
        ) from exc

    # 2. Predict
    try:
        prediction = _model_pipeline.predict([cleaned_text])
        label: str = str(prediction[0])
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(exc)}",
        ) from exc

    # 3. Confidence score
    try:
        if hasattr(_model_pipeline, "predict_proba"):
            probabilities = _model_pipeline.predict_proba([cleaned_text])
            confidence: float = float(np.max(probabilities))
        elif hasattr(_model_pipeline, "decision_function"):
            decision = _model_pipeline.decision_function([cleaned_text])
            confidence = float(np.max(np.abs(decision)))
        else:
            confidence = 0.0
    except Exception:
        logger.warning("Could not compute confidence score, defaulting to 0.0")
        confidence = 0.0

    # 4. Timestamp
    timestamp: str = datetime.now(timezone.utc).isoformat()

    # 5. Log to database
    try:
        db = get_db()
        try:
            log_prediction(
                db=db,
                tweet_text=request.tweet,
                label=label,
                confidence=confidence,
            )
        finally:
            db.close()
    except Exception:
        logger.exception("Database logging failed (prediction still returned)")

    logger.info("Prediction successful: label=%s confidence=%.4f", label, confidence)

    return PredictionResponse(
        label=label,
        confidence_score=round(confidence, 4),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Endpoints — History & Stats (Dashboard support)
# ---------------------------------------------------------------------------
@app.get("/history", response_model=List[HistoryItem], tags=["Dashboard"])
async def get_history(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records"),
) -> List[HistoryItem]:
    """Return the most recent predictions from the database.

    Args:
        limit: Maximum number of records to return (1-500, default 50).

    Returns:
        List of ``HistoryItem`` objects, newest first.
    """
    logger.info("History requested (limit=%d)", limit)
    try:
        db = get_db()
        try:
            records = get_recent_predictions(db, limit=limit)
            return [
                HistoryItem(
                    id=r.id,
                    tweet_text=r.tweet_text,
                    label=r.label,
                    confidence=round(r.confidence, 4),
                    created_at=r.created_at.isoformat() if r.created_at else None,
                )
                for r in records
            ]
        finally:
            db.close()
    except Exception as exc:
        logger.exception("Failed to fetch history")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(exc)}"
        ) from exc


@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
async def get_stats() -> StatsResponse:
    """Return aggregate prediction statistics.

    Returns:
        A ``StatsResponse`` with total count and label distribution.
    """
    logger.info("Stats requested")
    try:
        db = get_db()
        try:
            stats = get_prediction_stats(db)
            return StatsResponse(**stats)
        finally:
            db.close()
    except Exception as exc:
        logger.exception("Failed to fetch stats")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(exc)}"
        ) from exc


# ---------------------------------------------------------------------------
# Serve frontend static files (MUST be after all route definitions)
# ---------------------------------------------------------------------------
_STATIC_DIR: Path = BASE_DIR / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
