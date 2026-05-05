"""Database layer for Sg-ai prediction logging.

This module provides SQLAlchemy-based database operations for logging
inference predictions to a SQLite database.  Designed for portability
across Windows, macOS, and Linux environments.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import settings

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database configuration — driven by .env / config.py
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Model
# ---------------------------------------------------------------------------
class Prediction(Base):
    """SQLAlchemy ORM model for the ``predictions`` table.

    Attributes:
        id: Auto-incremented primary key.
        tweet_text: The raw tweet text submitted for prediction.
        label: Predicted sentiment / category label.
        confidence: Confidence score (probability) of the prediction.
        created_at: UTC timestamp when the record was created.
    """

    __tablename__ = "predictions"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    tweet_text: str = Column(String(1000), nullable=False)
    label: str = Column(String(100), nullable=False)
    confidence: float = Column(Float, nullable=False)
    created_at: datetime = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, label='{self.label}', "
            f"confidence={self.confidence:.4f})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the record to a plain dictionary.

        Returns:
            Dictionary with all column values.
        """
        return {
            "id": self.id,
            "tweet_text": self.tweet_text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Create all tables if they do not already exist.

    Should be called once at application startup.
    """
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


def get_db() -> Session:
    """Provide a transactional database session.

    Returns:
        A SQLAlchemy ``Session`` instance.  Caller is responsible for
        closing it.
    """
    return SessionLocal()


def log_prediction(
    db: Session,
    tweet_text: str,
    label: str,
    confidence: float,
) -> Prediction:
    """Log a successful prediction to the database.

    Args:
        db: Active SQLAlchemy session.
        tweet_text: Original tweet text from the request.
        label: Predicted label returned by the model.
        confidence: Model confidence score (0–1).

    Returns:
        The newly created ``Prediction`` record.

    Raises:
        Exception: Re-raises any database error after logging.
    """
    try:
        record = Prediction(
            tweet_text=tweet_text,
            label=label,
            confidence=confidence,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info("Prediction logged: id=%s label=%s", record.id, record.label)
        return record
    except Exception:
        db.rollback()
        logger.exception("Failed to log prediction to database")
        raise


def get_recent_predictions(db: Session, limit: int = 50) -> List[Prediction]:
    """Fetch the most recent predictions.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of records to return (default 50).

    Returns:
        List of ``Prediction`` objects ordered newest-first.
    """
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )


def get_prediction_stats(db: Session) -> Dict[str, Any]:
    """Compute aggregate statistics over all predictions.

    Args:
        db: Active SQLAlchemy session.

    Returns:
        Dictionary with:
            - ``total``: total prediction count.
            - ``label_distribution``: ``{label: count}`` mapping.
    """
    total: int = db.query(func.count(Prediction.id)).scalar() or 0
    rows = (
        db.query(Prediction.label, func.count(Prediction.id))
        .group_by(Prediction.label)
        .all()
    )
    distribution: Dict[str, int] = {label: count for label, count in rows}
    return {"total": total, "label_distribution": distribution}
