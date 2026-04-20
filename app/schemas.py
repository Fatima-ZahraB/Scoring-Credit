from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    SK_ID_CURR: int

    @field_validator("SK_ID_CURR")
    @classmethod
    def validate_id(cls, v):
        if v <= 0:
            raise ValueError("SK_ID_CURR must be a positive integer")
        return v


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    probability_default: float
    risk_level: str
    recommendation: str
    inference_time_ms: float
    timestamp: str

    @classmethod
    def from_proba(cls, sk_id: int, proba: float, inference_time_ms: float):
        if proba < 0.3:
            risk_level = "LOW"
            recommendation = "Credit likely to be approved"
        elif proba < 0.6:
            risk_level = "MEDIUM"
            recommendation = "Manual review recommended"
        else:
            risk_level = "HIGH"
            recommendation = "Credit likely to be refused"

        return cls(
            SK_ID_CURR=sk_id,
            probability_default=round(proba, 6),
            risk_level=risk_level,
            recommendation=recommendation,
            inference_time_ms=round(inference_time_ms, 3),
            timestamp=datetime.now().isoformat(),
        )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool
    total_clients: int
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    SK_ID_CURR: Optional[int] = None
