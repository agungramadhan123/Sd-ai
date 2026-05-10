#database untuk meyambungkan ke web api
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config import settings

logger: logging.Logger = logging.getLogger(__name__)

# penyambungan database
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ORM Model
class Prediction(Base):
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
        return {
            "id": self.id,
            "tweet_text": self.tweet_text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


def get_db() -> Session:
    return SessionLocal()


def log_prediction(
    db: Session,
    tweet_text: str,
    label: str,
    confidence: float,
) -> Prediction:
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
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .all()
    )


def get_prediction_stats(db: Session) -> Dict[str, Any]:
    total: int = db.query(func.count(Prediction.id)).scalar() or 0
    rows = (
        db.query(Prediction.label, func.count(Prediction.id))
        .group_by(Prediction.label)
        .all()
    )
    distribution: Dict[str, int] = {label: count for label, count in rows}
    return {"total": total, "label_distribution": distribution}
