"""
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import settings, ML_DIR, BASE_DIR
from database import (
    get_db,
    get_prediction_stats,
    get_recent_predictions,
    init_db,
    log_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger: logging.Logger = logging.getLogger("sg-ai")
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
_model_pipelines: Dict[str, Any] = {}
_preprocessor: Optional[Any] = None


class ConnectionManager:
    """Manages active WebSocket connections for the real-time dashboard."""

    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(
            "WebSocket client connected. Active connections: %d",
            len(self._connections),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            len(self._connections),
        )

    async def broadcast(self, data: dict) -> None:
        """Send JSON data to all connected clients."""
        payload = json.dumps(data, default=str)
        stale: List[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(ws)


ws_manager = ConnectionManager()

# Request / Response schemas (Pydantic)
class TweetRequest(BaseModel):
    tweet: str
    model_name: str = settings.DEFAULT_MODEL_NAME

    @field_validator("tweet")
    @classmethod
    def validate_tweet(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("tweet must be a string")
        v = v.strip()
        if len(v) == 0:
            raise ValueError("tweet must not be empty")
        if v.isdigit():
            raise ValueError(
                "Input tidak boleh hanya berisi angka. "
                "Masukkan teks tweet yang valid."
            )
        if len(v) > settings.MAX_INPUT_LENGTH:
            raise ValueError(
                f"tweet exceeds maximum length of {settings.MAX_INPUT_LENGTH} "
                f"characters (got {len(v)})"
            )
        return v


class PredictionResponse(BaseModel):
    label: str
    confidence_score: float
    model_name: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"


class HistoryItem(BaseModel):
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


# Application lifespan -- load all models at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_pipelines, _preprocessor

    # 1. Initialise database
    try:
        init_db()
        logger.info("Database initialised successfully")
    except Exception:
        logger.exception("Failed to initialise database")
        raise

    # 2. Load all models from registry
    for model_name, model_path in settings.MODEL_REGISTRY.items():
        try:
            if not model_path.exists():
                logger.warning(
                    "Model file not found for '%s' at %s -- skipping",
                    model_name,
                    model_path,
                )
                continue
            _model_pipelines[model_name] = joblib.load(model_path)
            logger.info(
                "Model '%s' loaded successfully from %s", model_name, model_path
            )
        except Exception:
            logger.exception("Failed to load model '%s'", model_name)

    if not _model_pipelines:
        raise RuntimeError(
            "No models could be loaded. Check MODEL_REGISTRY paths in config."
        )

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
        "NLP inference API for classifying tweets using pre-trained "
        "scikit-learn pipelines (TF-IDF + SMOTE). Supports multiple models."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints -- Health
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Return the health status of the API and all loaded models."""
    models_status: Dict[str, bool] = {
        name: name in _model_pipelines
        for name in settings.MODEL_REGISTRY
    }
    return HealthResponse(
        status="ok",
        models_loaded=models_status,
    )

# Endpoints -- Inference
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
)
async def predict(request: TweetRequest) -> PredictionResponse:
    logger.info(
        "Prediction request received (length=%d, model=%s)",
        len(request.tweet),
        request.model_name,
    )

    # Guard: requested model must exist
    if request.model_name not in _model_pipelines:
        available = list(_model_pipelines.keys())
        logger.error(
            "Model '%s' not found. Available: %s", request.model_name, available
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model_name}' not found. "
                f"Available models: {available}"
            ),
        )

    pipeline = _model_pipelines[request.model_name]

    # Guard: preprocessor must be loaded
    if _preprocessor is None:
        logger.error("Prediction attempted but preprocessor is not loaded")
        raise HTTPException(
            status_code=503,
            detail="Preprocessor is not loaded. Please try again later.",
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
        prediction = pipeline.predict([cleaned_text])
        label: str = str(prediction[0])
    except Exception as exc:
        logger.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(exc)}",
        ) from exc

    # 3. Confidence score
    try:
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba([cleaned_text])
            confidence: float = float(np.max(probabilities))
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function([cleaned_text])
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

    logger.info(
        "Prediction successful: model=%s label=%s confidence=%.4f",
        request.model_name,
        label,
        confidence,
    )

    # 6. Broadcast to WebSocket clients
    try:
        db = get_db()
        try:
            stats = get_prediction_stats(db)
            recent = get_recent_predictions(db, limit=10)
            history_data = [
                {
                    "id": r.id,
                    "tweet_text": r.tweet_text,
                    "label": r.label,
                    "confidence": round(r.confidence, 4),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recent
            ]
        finally:
            db.close()

        await ws_manager.broadcast({
            "type": "update",
            "stats": stats,
            "history": history_data,
        })
    except Exception:
        logger.warning("Failed to broadcast WebSocket update")

    return PredictionResponse(
        label=label,
        confidence_score=round(confidence, 4),
        model_name=request.model_name,
        timestamp=timestamp,
    )

# Endpoints -- WebSocket Dashboard
@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    """Real-time dashboard WebSocket endpoint.

    On connect, sends current stats and recent history.
    Stays open to receive broadcast updates after each new prediction.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial data on connection
        db = get_db()
        try:
            stats = get_prediction_stats(db)
            recent = get_recent_predictions(db, limit=10)
            history_data = [
                {
                    "id": r.id,
                    "tweet_text": r.tweet_text,
                    "label": r.label,
                    "confidence": round(r.confidence, 4),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in recent
            ]
        finally:
            db.close()

        await websocket.send_text(
            json.dumps(
                {"type": "init", "stats": stats, "history": history_data},
                default=str,
            )
        )

        # Keep connection alive -- wait for client messages or disconnect
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# Endpoints -- History & Stats (REST fallback)
@app.get("/history", response_model=List[HistoryItem], tags=["Dashboard"])
async def get_history(
    limit: int = Query(default=50, ge=1, le=500, description="Number of records"),
) -> List[HistoryItem]:
    """Return the most recent predictions from the database."""
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
    """Return aggregate prediction statistics."""
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
