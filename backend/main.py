from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import get_db
from database.models import (
    FarmRecord,
    Prediction,
    CropRecommendation
)

from backend.nsaf_service import recommend_crop
from backend.yield_service import predict_yield


# =========================================================
# FastAPI APPLICATION
# =========================================================

app = FastAPI(
    title="AgriSense AI",
    description="AgriSense AI - NSAF Crop Recommendation and Adaptive Yield Prediction",
    version="1.0.0"
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# INPUT SCHEMAS
# =========================================================

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class YieldInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    area: float


class FarmInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    area: float


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AgriSense AI API is running",

        "services": [
            "NSAF Crop Recommendation",
            "Adaptive Yield Prediction",
            "Farm Analysis",
            "PostgreSQL Database"
        ]
    }


# =========================================================
# CROP RECOMMENDATION
# =========================================================

@app.post("/recommend")
def crop_recommendation(data: CropInput):

    input_data = data.model_dump()

    result = recommend_crop(input_data)

    return result


# =========================================================
# YIELD PREDICTION
# =========================================================

@app.post("/predict-yield")
def yield_prediction(data: YieldInput):

    input_data = data.model_dump()

    result = predict_yield(input_data)

    return result


# =========================================================
# COMPLETE FARM ANALYSIS
# =========================================================

@app.post("/analyze")
def analyze_farm(
    data: FarmInput,
    db: Session = Depends(get_db)
):

    input_data = data.model_dump()


    # =====================================================
    # 1. SAVE FARM INPUT
    # =====================================================

    farm_record = FarmRecord(
        N=data.N,
        P=data.P,
        K=data.K,
        temperature=data.temperature,
        humidity=data.humidity,
        ph=data.ph,
        rainfall=data.rainfall,
        area=data.area
    )

    db.add(farm_record)
    db.commit()
    db.refresh(farm_record)


    # =====================================================
    # 2. NSAF CROP RECOMMENDATION
    # =====================================================

    crop_result = recommend_crop(
        input_data
    )


    # =====================================================
    # 3. ADAPTIVE YIELD PREDICTION
    # =====================================================

    yield_result = predict_yield(
        input_data
    )


    # =====================================================
    # 4. SAVE MAIN PREDICTION
    # =====================================================

    prediction = Prediction(
        farm_record_id=farm_record.id,
        recommended_crop=crop_result["recommended_crop"],
        predicted_yield=yield_result["predicted_yield"]
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)


    # =====================================================
    # 5. SAVE TOP CROP RECOMMENDATIONS
    # =====================================================

    for rank, item in enumerate(
        crop_result["top_predictions"],
        start=1
    ):

        recommendation = CropRecommendation(
            prediction_id=prediction.id,
            crop=item["crop"],
            distance=item["distance"],
            rank=rank
        )

        db.add(recommendation)

    db.commit()


    # =====================================================
    # 6. RETURN COMPLETE ANALYSIS
    # =====================================================

    return {

        "farm_record_id": farm_record.id,

        "prediction_id": prediction.id,

        "crop_recommendation": crop_result,

        "yield_prediction": yield_result,

        "database_status": "saved"
    }

    # =========================================================
# DATABASE HISTORY
# =========================================================

@app.get("/history")
def get_history(db: Session = Depends(get_db)):

    records = (
        db.query(FarmRecord, Prediction)
        .join(
            Prediction,
            Prediction.farm_record_id == FarmRecord.id
        )
        .order_by(FarmRecord.created_at.desc())
        .limit(20)
        .all()
    )

    history = []

    for farm_record, prediction in records:

        recommendations = (
            db.query(CropRecommendation)
            .filter(
                CropRecommendation.prediction_id == prediction.id
            )
            .order_by(CropRecommendation.rank.asc())
            .all()
        )

        history.append({
            "farm_record_id": farm_record.id,

            "created_at": farm_record.created_at,

            "inputs": {
                "N": farm_record.N,
                "P": farm_record.P,
                "K": farm_record.K,
                "temperature": farm_record.temperature,
                "humidity": farm_record.humidity,
                "ph": farm_record.ph,
                "rainfall": farm_record.rainfall,
                "area": farm_record.area
            },

            "recommended_crop": prediction.recommended_crop,

            "predicted_yield": prediction.predicted_yield,

            "top_predictions": [
                {
                    "crop": item.crop,
                    "distance": item.distance,
                    "rank": item.rank
                }
                for item in recommendations
            ]
        })

    return {
        "count": len(history),
        "history": history
    }