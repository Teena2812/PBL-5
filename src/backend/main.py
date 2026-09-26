"""
Single central FastAPI backend (per the locked architecture -- one
backend for the whole system, not one per hospital). Serves:
  - precomputed dashboard data (Phases 2-6 results) -- src/backend/data_routes.py
  - live single-patient prediction + explanation -- src/backend/inference.py,
    inference only, no retraining ever

Run:
    venv\\Scripts\\uvicorn src.backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.backend import inference
from src.backend.data_routes import router as data_router
from src.backend.live_training import router as live_training_router
from src.backend.schemas import PatientInput, PredictionResponse

app = FastAPI(
    title="Personalized FL Heart Disease Risk Prediction API",
    description=(
        "Research prototype backend. Serves precomputed Phase 2-6 results "
        "and a live single-patient prediction endpoint (inference + "
        "single-instance SHAP explanation only -- models are never "
        "retrained here). Not a clinical decision-making tool."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Vite (5173) and Create React App (3000) dev server defaults.
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(data_router)
app.include_router(live_training_router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "live_prediction_available": inference.TORCH_AVAILABLE,
        "live_prediction_note": (
            None if inference.TORCH_AVAILABLE else
            "torch/shap not installed in this environment -- dashboard data "
            "endpoints work normally, but POST /api/predict will return 503."
        ),
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict(patient: PatientInput) -> dict:
    try:
        return inference.predict_and_explain(
            patient.model_dump(exclude={"hospital_id"}), patient.hospital_id
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
