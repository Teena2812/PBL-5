"""
Pydantic request/response models for the FastAPI backend.

PatientInput's field bounds are generous clinical sanity ranges (catches
garbage input like age=-5 or trestbps=9999), not the tight bounds of the
training data -- the categorical fields (cp/restecg/slope/thal) are
further validated against exactly what the model was trained on (see
src/data/load_dataset.py:transform_new_patient), since a value outside
that set has no corresponding one-hot column and can't be encoded at all.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0/1)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (0/1)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia result (3=normal, 6=fixed defect, 7=reversible defect)")
    hospital_id: str = Field(..., description="Which hospital's personalized model to use, e.g. 'hospital_1'")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 6,
                "hospital_id": "hospital_1",
            }
        }
    }


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class PredictionResponse(BaseModel):
    hospital_id: str
    predicted_probability: float = Field(..., description="Model's predicted probability of disease (0-1)")
    predicted_label: int = Field(..., description="0 = no disease, 1 = disease, at the 0.5 threshold")
    risk_level: str = Field(..., description="LOW / MODERATE / HIGH, bucketed from predicted_probability")
    model_test_accuracy: float = Field(..., description="This hospital's personalized model's own test accuracy, for context")
    base_value: float = Field(..., description="SHAP base value (average model output over the background sample)")
    explanation: list[FeatureContribution] = Field(..., description="Top features driving this prediction, ranked by |SHAP value|")
    disclaimer: str = (
        "Research prototype, not a clinical decision-making tool. Trained on a "
        "small, simulated non-IID partition of the public UCI Heart Disease "
        "dataset -- not validated for real patient care."
    )
