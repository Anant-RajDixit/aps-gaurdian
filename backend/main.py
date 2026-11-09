from __future__ import annotations

import json
import os
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.predictor import APSPredictor, InputValidationError, ModelArtifactError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "scania_aps_xgboost_final.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "100000"))
MONITORING_PATH = Path(os.getenv(
    "MONITORING_PATH",
    str(Path(__file__).resolve().parent / "data" / "model_monitoring.json")
)).expanduser()


class SinglePredictionRequest(BaseModel):
    record_id: str | None = None
    features: dict[str, Any] = Field(..., description="Raw sensor features using the APS dataset column names.")


class ExplanationRequest(SinglePredictionRequest):
    top_n: int = Field(default=10, ge=1, le=25)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.predictor = APSPredictor(MODEL_PATH)
    except ModelArtifactError as exc:
        # Raise a clear startup error rather than serving predictions without a model.
        raise RuntimeError(str(exc)) from exc
    yield
    app.state.predictor = None


app = FastAPI(
    title="Scania APS Failure Prediction API",
    description=(
        "Backend API for batch and single-truck APS failure scoring. "
        "The API loads the saved preprocessing + XGBoost pipeline and applies "
        "the OOF-selected cost-sensitive threshold. Scores are uncalibrated."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> APSPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")
    return predictor


async def read_csv_upload(file: UploadFile, required_features: list[str] | None = None) -> pd.DataFrame:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Please upload a .csv file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
        )

    na_values = ["na", "NA", "Na", "N/A", "n/a", ""]
    try:
        # First try a normal CSV. The original UCI APS files contain 20
        # descriptive lines before the header, so fall back to skiprows=20
        # when the normal parse does not resemble the saved training schema.
        frame = pd.read_csv(
            io.BytesIO(content),
            na_values=na_values,
            low_memory=False,
        )
        if required_features:
            normal_missing = sum(col not in frame.columns for col in required_features)
            if normal_missing > max(2, int(0.05 * len(required_features))):
                try:
                    uci_frame = pd.read_csv(
                        io.BytesIO(content),
                        skiprows=20,
                        na_values=na_values,
                        low_memory=False,
                    )
                    uci_missing = sum(col not in uci_frame.columns for col in required_features)
                    if uci_missing < normal_missing:
                        frame = uci_frame
                except Exception:
                    # Keep the normal parse; schema validation will report missing columns.
                    pass
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc

    if len(frame) > MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"CSV contains {len(frame):,} rows; maximum allowed is {MAX_UPLOAD_ROWS:,}."
        )
    return frame


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    return {
        "application": "Scania APS Failure Prediction API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health", tags=["System"])
def health(request: Request) -> dict[str, Any]:
    predictor = getattr(request.app.state, "predictor", None)
    return {
        "status": "healthy" if predictor is not None else "model_not_loaded",
        "model_loaded": predictor is not None,
    }


@app.get("/model-info", tags=["Model"])
def model_info(request: Request) -> dict[str, Any]:
    return get_predictor(request).model_info()


@app.post("/validate", tags=["Data"])
async def validate_csv(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    predictor = get_predictor(request)
    frame = await read_csv_upload(file, predictor.required_features)
    try:
        _, _, quality = predictor.validate_frame(frame)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"valid": True, "data_quality": quality}


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    predictor = get_predictor(request)
    frame = await read_csv_upload(file, predictor.required_features)
    try:
        return predictor.score_frame(frame)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/single", tags=["Prediction"])
def predict_single(payload: SinglePredictionRequest, request: Request) -> dict[str, Any]:
    predictor = get_predictor(request)
    frame = predictor.validate_single(payload.features)
    try:
        output = predictor.score_frame(frame)
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = output["results"][0]
    result["record_id"] = payload.record_id or result["record_id"]
    return {
        "model_name": output["model_name"],
        "score_type": output["score_type"],
        "threshold": output["threshold"],
        "result": result,
        "data_quality": output["data_quality"],
    }


@app.post("/explain", tags=["Explainability"])
def explain_single(payload: ExplanationRequest, request: Request) -> dict[str, Any]:
    predictor = get_predictor(request)
    try:
        return predictor.explain_single(
            payload.features,
            record_id=payload.record_id,
            top_n=payload.top_n,
        )
    except InputValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc


@app.get("/model-metrics", tags=["Monitoring"])
def model_metrics() -> JSONResponse:
    """Return historical holdout metrics and optional exported monitoring curves."""
    metrics_path = Path(__file__).resolve().parent / "data" / "model_metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(status_code=404, detail="Historical model metrics are not configured.")

    with metrics_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if MONITORING_PATH.is_file():
        with MONITORING_PATH.open("r", encoding="utf-8") as file:
            monitoring = json.load(file)
        data["curves"] = monitoring.get("curves", {})
        data["monitoring_curves_available"] = bool(data["curves"])
        data["monitoring_curves_note"] = monitoring.get("note", "")
    else:
        data["monitoring_curves_available"] = False
        data["monitoring_curves_note"] = (
            "Run backend/export_monitoring.py from the analysis notebook to export "
            "holdout PR, calibration and cost-versus-threshold curves."
        )

    return JSONResponse(content=data)
