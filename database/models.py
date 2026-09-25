from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func

from database.database import Base


# =========================================================
# USER TABLE
# =========================================================

class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(
        String(100),
        nullable=False
    )

    email = Column(
        String(150),
        unique=True,
        nullable=False,
        index=True
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


# =========================================================
# FARM RECORD TABLE
# =========================================================

class FarmRecord(Base):

    __tablename__ = "farm_records"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True
    )

    N = Column(Float, nullable=False)
    P = Column(Float, nullable=False)
    K = Column(Float, nullable=False)

    temperature = Column(
        Float,
        nullable=False
    )

    humidity = Column(
        Float,
        nullable=False
    )

    ph = Column(
        Float,
        nullable=False
    )

    rainfall = Column(
        Float,
        nullable=False
    )

    area = Column(
        Float,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


# =========================================================
# PREDICTION TABLE
# =========================================================

class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    farm_record_id = Column(
        Integer,
        ForeignKey("farm_records.id"),
        nullable=False
    )

    recommended_crop = Column(
        String(100),
        nullable=False
    )

    predicted_yield = Column(
        Float,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )


# =========================================================
# CROP RECOMMENDATION TABLE
# =========================================================

class CropRecommendation(Base):

    __tablename__ = "crop_recommendations"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    prediction_id = Column(
        Integer,
        ForeignKey("predictions.id"),
        nullable=False
    )

    crop = Column(
        String(100),
        nullable=False
    )

    distance = Column(
        Float,
        nullable=False
    )

    rank = Column(
        Integer,
        nullable=False
    )