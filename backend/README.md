# Scania APS Failure Prediction — FastAPI backend

This backend loads the **already-fitted** Cloudpickle artifact created by the notebook and exposes prediction, validation, explainability and model-monitoring endpoints. It does not retrain the model.

## 1. Expected project layout

Copy this `backend/` folder into the root of the existing project, next to the `model/` folder:

```text
Truck_Health/
├── backend/
│   ├── main.py
│   ├── predictor.py
│   ├── export_monitoring.py
│   ├── requirements.txt
│   └── data/
│       └── model_metrics.json
├── model/
│   └── scania_aps_xgboost_final.pkl
└── ...
```

The artifact must be the same file saved by the notebook and must contain a dictionary with `pipeline`, `threshold`, `feature_names`, `false_positive_cost` and `false_negative_cost` entries.

## 2. Install dependencies

From the project root:

```bash
python -m pip install -r backend/requirements.txt
```

Use the same or compatible versions of Python, scikit-learn, XGBoost and Cloudpickle as the environment in which the artifact was saved. Only load model pickle files from trusted sources.

## 3. Start the API

From the project root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Model information: http://127.0.0.1:8000/model-info

The API loads the artifact once when it starts. If the artifact is missing, startup fails with a clear message.

## 4. Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API/model readiness |
| GET | `/model-info` | Algorithm, threshold, feature counts and costs |
| POST | `/validate` | Validate a CSV and return data-quality checks |
| POST | `/predict/batch` | Score every row in an uploaded CSV |
| POST | `/predict/single` | Score one vehicle supplied as JSON |
| POST | `/explain` | Return local SHAP contributions for one vehicle |
| GET | `/model-metrics` | Return saved holdout metrics and monitoring curves, if exported |

### Batch CSV upload

Use Swagger `/docs`, select `POST /predict/batch`, and upload a CSV.

The model expects all 169 original sensor columns used during training. The optional UCI target column `class` is ignored. Optional identifiers (`vehicle_id`, `truck_id`, `record_id`, or `id`) are carried into the output but are not model features. Other unexpected columns are ignored and reported in the data-quality summary.

The endpoint returns:

- batch summary (total vehicles, inspection recommendations, inspection rate)
- data-quality summary (missing cells, invalid numeric values, extra columns)
- row-level results sorted by descending APS failure score

Predicted scores are **uncalibrated model scores**. The saved OOF threshold is applied to produce `INSPECT` or `NO_INSPECTION` decisions.

### Single vehicle

`POST /predict/single` accepts JSON of this form. Include every required raw sensor feature in `features`:

```json
{
  "record_id": "Truck-001",
  "features": {
    "aa_000": 10000,
    "ab_000": 0
  }
}
```

The example above is intentionally incomplete and will return HTTP 422 until all required sensor columns are supplied. Use `/model-info` for counts and the notebook's `feat_cols` to confirm the full input schema.

### Local SHAP explanation

`POST /explain` uses TreeSHAP on one raw input row and returns the top absolute contributions. Positive contributions push the raw model output towards APS failure; negative contributions push it away. These are not causal effects or percentage-point changes in probability.

### Model monitoring metrics

`GET /model-metrics` returns the holdout metric values recorded in the notebook. The API does not infer live model performance from unlabeled batch uploads.

To enable PR, calibration and cost-versus-threshold plots in the monitoring page, run the following in the notebook **after** the final holdout evaluation, when `y_hold`, `test_probs`, `FINAL_THRESHOLD`, `C_FP` and `C_FN` are defined:

```python
from backend.export_monitoring import export_monitoring_data

monitoring_path = export_monitoring_data(
    y_true=y_hold,
    probabilities=test_probs,
    threshold=FINAL_THRESHOLD,
    false_positive_cost=C_FP,
    false_negative_cost=C_FN,
)
print(monitoring_path)
```

The generated curves are historical holdout diagnostics only. Do not use them to re-optimise the threshold or select another model after looking at the official holdout set. The selected XGBoost model was not calibrated; its calibration curve is a diagnostic, not a claim of calibrated probabilities.

## 5. Streamlit connection

The frontend should call the API at `http://127.0.0.1:8000` using `requests`:

- Batch page: POST the uploaded CSV to `/predict/batch`.
- Truck details: POST a row to `/predict/single` and `/explain`.
- Overview: GET `/model-info` and `/model-metrics`.
- Monitoring: GET `/model-metrics`; use batch response `data_quality` for upload-specific quality checks.

For local development, allow `http://localhost:8501` in `CORS_ORIGINS`. Change the environment variable when the Streamlit host differs.

## 6. Important implementation notes

- The backend does not retrain or refit the saved model.
- The original preprocessing pipeline is used exactly as saved.
- The target column `class` is never passed to the model.
- Missing sensor measurements are handled by the model's saved median imputer; missing **columns** are rejected because the trained feature schema must be preserved.
- Input upload defaults are limited to 50 MB and 100,000 rows; configure via environment variables if required.
- The UCI dataset has no vehicle identifier column, so a synthetic row identifier is generated when none is supplied.
- This is a local development backend. Add authentication, request logging, rate limiting and HTTPS before exposing it publicly.
