# AlloyTower AVM — Inference Application

Web-based property valuation tool built on top of the Phase 1 AVM models.
FastAPI backend + Streamlit frontend, designed for non-technical stakeholder
demos and analyst exploration.

## What this is

A two-service app that lets users type in a property's details, get a market
value estimate, see what factors drove the prediction, and browse comparable
properties from the dataset.

The backend is a thin REST API around the trained `model_A.pkl` and
`model_B.pkl` bundles produced by `train_avm.py`. The frontend is a
three-page Streamlit app that renders results in a stakeholder-friendly way.

```
┌──────────────────────────────┐
│  Streamlit (port 8501)       │  ← what users see
│  - Home: Get a valuation     │
│  - Market Explorer           │
│  - About this Model          │
└────────────┬─────────────────┘
             │  HTTP/JSON
             ↓
┌──────────────────────────────┐
│  FastAPI (port 8000)         │  ← model serving
│  /predict                    │
│  /comparables                │
│  /metadata                   │
│  /health                     │
└──────────────────────────────┘
             │
             ↓
       artifacts/
       ├─ model_A.pkl
       ├─ model_B.pkl
       └─ alloy_clean.csv
```

## Prerequisites

1. The `train_avm.py` pipeline must have already produced its outputs in
   `../model/models/`. Specifically you need `model_A.pkl`, `model_B.pkl`, and
   `alloy_clean.csv`.
2. The `similar_properties.py` module from the project root must be on the
   Python path (it sits one directory up from `api/`).

## Setup

```bash
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The app expects to find the artifacts at `model/models/` relative to wherever
you start the services from. If your artifacts live elsewhere, either symlink
them or run the services from your project root.

## Running

### 1. Start the FastAPI backend

In one terminal, from the project root (where `model/` lives):

```bash
uvicorn api.main:app --port 8000 --reload
```

You should see:

```
[HH:MM:SS] INFO | avm_api | Loading AVM models...
[HH:MM:SS] INFO | avm_api | Loaded Model A (MAPE 8.5%) and Model B (MAPE 44.9%)
[HH:MM:SS] INFO | avm_api | Loading comparables index from artifacts/alloy_clean.csv...
[HH:MM:SS] INFO | avm_api | Indexed 1,956 properties for similarity lookup
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Quick API check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

### 2. Start the Streamlit frontend

In a second terminal, from the same project root:

```bash
streamlit run app/Home.py
```

The browser opens automatically at `http://localhost:8501`. If it doesn't,
open that URL manually.

### 3. Use the app

The sidebar has a "Try a sample property" dropdown — pick one and click
"Load sample" to populate the form, then click "Estimate value".

## Configuration

The frontend reads the API URL from the `AVM_API_URL` environment variable
(default: `http://localhost:8000`). Useful when the backend is on a different
host:

```bash
AVM_API_URL=http://10.0.1.5:8000 streamlit run app/Home.py
```

## Project structure

```
avm_app/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app + endpoints
│   └── predictor.py         # Model wrapper (pipeline + SHAP)
├── app/
│   ├── lib.py               # Shared helpers (API client, colors, components)
│   ├── Home.py              # Page 1 — valuation form + result
│   └── pages/
│       ├── 2_Market_Explorer.py    # Page 2 — filter & explore the dataset
│       └── 3_About_Model.py        # Page 3 — model card & limitations
├── .streamlit/
│   └── config.toml          # Theme + telemetry settings
├── requirements.txt
└── README.md
```

## Endpoints reference

| Method | Path             | Body          | Returns                          |
|--------|------------------|---------------|----------------------------------|
| GET    | /health          |               | Liveness + which models loaded   |
| GET    | /metadata        |               | Model labels, MAPE, training info|
| POST   | /predict         | PropertyInput | Prediction + SHAP contributions  |
| POST   | /predict/batch   | up to 100 props| List of predictions             |
| POST   | /comparables     | query + n     | Top-N similar properties         |

OpenAPI docs are auto-generated at `http://localhost:8000/docs`.

## Notes for stakeholder demos

- The "Prototype — Phase 1" banner is shown on every page on purpose. Don't
  remove it — it's the visual reminder that predictions are based on a small
  training set and shouldn't be used for actual business decisions yet.
- The "About this Model" page is the credibility page. If a stakeholder asks
  "how accurate is this really?", point them there. It's deliberately honest
  about the dataset's limitations.
- For investor or external presentations, screenshot the Home page after
  loading a sample property — the SF condo example reliably produces a
  high-confidence Model A prediction with a clean comparables grid.

## Common issues

**`API service unavailable` on the home page.** The FastAPI backend isn't
running, or it's running on a different port. Check `curl http://localhost:8000/health`.

**`Model bundles missing` when starting the API.** Run `python train_avm.py`
first to produce `model/models/model_A.pkl` and `model/models/model_B.pkl`.

**`Cannot import similar_properties`.** The module needs to be on the Python
path. Make sure you're running uvicorn from a directory where Python can find
`similar_properties.py` (typically the project root).

**Predictions seem to take a long time.** SHAP value computation adds ~0.5–1s
per prediction. This is expected and used for the "What drove this estimate"
explanation.
