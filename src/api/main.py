"""FastAPI service that returns a default-risk score for a loan application.

Run locally:
    uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build import add_derived_features

LOW_RISK_MAX = 0.15
MEDIUM_RISK_MAX = 0.35


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))


@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m src.models.train` first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_model_name() -> str:
    meta = model_path().with_name("metadata.json")
    return json.loads(meta.read_text())["best_model"] if meta.exists() else "unknown"


class LoanApplication(BaseModel):
    loan_amnt: float = Field(gt=0, description="Requested loan amount in USD")
    term_months: Literal[36, 60]
    int_rate: float = Field(ge=0, description="Interest rate in percent, e.g. 13.5")
    installment: float = Field(gt=0, description="Monthly payment in USD")
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    emp_length_years: Optional[int] = Field(default=None, ge=0, le=10)
    home_ownership: str = "RENT"
    annual_inc: float = Field(ge=0)
    verification_status: str = "Verified"
    purpose: str = "debt_consolidation"
    addr_state: str = Field(min_length=2, max_length=2)
    application_type: str = "Individual"
    dti: Optional[float] = None
    delinq_2yrs: float = 0
    fico_range_low: float = Field(ge=300, le=850)
    fico_range_high: float = Field(ge=300, le=850)
    inq_last_6mths: float = 0
    open_acc: float = 0
    pub_rec: float = 0
    revol_bal: float = 0
    revol_util: Optional[float] = None
    total_acc: float = 0
    mort_acc: Optional[float] = None
    pub_rec_bankruptcies: Optional[float] = None
    credit_history_years: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "loan_amnt": 15000,
                    "term_months": 36,
                    "int_rate": 12.5,
                    "installment": 502.0,
                    "grade": "B",
                    "emp_length_years": 5,
                    "home_ownership": "RENT",
                    "annual_inc": 65000,
                    "verification_status": "Verified",
                    "purpose": "debt_consolidation",
                    "addr_state": "MA",
                    "application_type": "Individual",
                    "dti": 18.2,
                    "delinq_2yrs": 0,
                    "fico_range_low": 700,
                    "fico_range_high": 704,
                    "inq_last_6mths": 1,
                    "open_acc": 10,
                    "pub_rec": 0,
                    "revol_bal": 12000,
                    "revol_util": 45.0,
                    "total_acc": 22,
                    "mort_acc": 0,
                    "pub_rec_bankruptcies": 0,
                    "credit_history_years": 9.5,
                }
            ]
        }
    }


class RiskScore(BaseModel):
    default_probability: float
    risk_band: Literal["low", "medium", "high"]
    model: str


def risk_band(probability: float) -> str:
    if probability < LOW_RISK_MAX:
        return "low"
    if probability < MEDIUM_RISK_MAX:
        return "medium"
    return "high"


app = FastAPI(
    title="Loan Default Risk API",
    description="Scores a loan application using a model trained on finished Lending Club loans.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": model_path().exists()}


@app.post("/predict", response_model=RiskScore)
def predict(application: LoanApplication) -> RiskScore:
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = add_derived_features(pd.DataFrame([application.model_dump()]))
    probability = float(model.predict_proba(features)[0, 1])
    return RiskScore(
        default_probability=round(probability, 4),
        risk_band=risk_band(probability),
        model=get_model_name(),
    )
