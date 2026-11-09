# Truck Health — Scania APS Failure Prediction

This project provides a FastAPI prediction API and a Streamlit dashboard for prioritising truck inspections using the saved Scania APS XGBoost model. The raw training and test datasets are intentionally excluded from version control; the fitted model required for inference is included in `model/`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend\requirements.txt -r frontend\requirements.txt
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, start the dashboard:

```powershell
.\.venv\Scripts\python -m streamlit run frontend\app.py
```

- API documentation: http://127.0.0.1:8000/docs
- API health check: http://127.0.0.1:8000/health
- Dashboard: http://127.0.0.1:8501

## Deploy on Render

`render.yaml` is a Render Blueprint at the repository root. In Render, create a **New Blueprint** and use `render.yaml` as the Blueprint path. It creates two web services:

1. `truck-health-api`: the public FastAPI service, with `/health` as its health check.
2. `truck-health-frontend`: the public Streamlit dashboard. It receives the API service's private host and port automatically through `API_HOSTPORT`.

Both services use Python 3.13.1 and deploy automatically on commits. After the Blueprint deploys, use the `truck-health-frontend` URL for the dashboard and append `/docs` to the `truck-health-api` URL for Swagger.

See [backend/README.md](backend/README.md) for API endpoints and input expectations.
