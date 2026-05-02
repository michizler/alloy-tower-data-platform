# AlloyTower Centralized Data Platform — Prototype

**Property valuation and market intelligence for the AlloyTower real estate platform.**

A 3-week internship project covering data validation, automated valuation modelling (AVM), comparables search, Power BI integration, and a stakeholder-facing inference application.

> **Author:** Bright Uzosike — Data Scientist
> **Status:** Phase 1 prototype — see [Known limitations](#known-limitations) before drawing business conclusions.
> **Reflections** Loom Video - see [Reflection Video](https://www.loom.com/share/4158cfcfc12c41a4b3bffc0ce2a10020)

---

## Live demo

| Service               | URL                                              | What it is                                                                             |
| --------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| **Web app**           | <https://alloy-avm.streamlit.app/>               | Streamlit frontend — try a property valuation, browse comparables, explore the dataset |
| **API backend**       | <https://alloytower-avm-api.onrender.com>        | FastAPI service — model serving + similarity search                                    |
| **API health check**  | <https://alloytower-avm-api.onrender.com/health> | Readiness probe                                                                        |
| **API documentation** | <https://alloytower-avm-api.onrender.com/docs>   | Auto-generated Swagger UI                                                              |

### ⚠️ First-load delay (this is normal)

The backend runs on Render's free tier, which **spins the service down after 15 minutes of inactivity**. The first request after that delay takes **30 to 50 seconds** while the container wakes up and reloads the model.

If you open the app and see one of these:

- "API service unavailable. Start the backend with: `uvicorn api.main:app --port 8000`"
- A spinner that hangs for more than 30 seconds on the first prediction
- A 502 / 503 error from the API

**It's not broken — the service is waking up.** Wait 30-50 seconds and reload the Streamlit page. The app will work normally for the rest of the session, then go to sleep again after 15 minutes of inactivity.

For demos, hit the health-check URL above ~60 seconds before opening the app — that warms the service so the demo is snappy.

---

## What's in this repository

```
alloy-tower-data-platform/
├── api/                          FastAPI backend (deployed to Render)
│   ├── main.py                   API endpoints + CORS + startup hooks
│   └── predictor.py              Model wrapper — feature pipeline + SHAP
├── app/                          Streamlit frontend (deployed to Streamlit Cloud)
│   ├── Home.py                   Page 1 — valuation form + result + comparables
│   └── lib.py                    Shared UI helpers + API client
├── pages/                        Streamlit auto-discovered pages
│   ├── 2_Market_Explorer.py      Filter & explore the dataset
│   └── 3_About_Model.py          Model card + honest limitations
├── model/models/                 Trained models + cleaned dataset (committed)
│   ├── model_A.pkl               Assessment-aware AVM
│   ├── model_B.pkl               Fundamentals-only AVM
│   ├── alloy_clean.csv           Cleaned dataset (1,956 rows)
│   └── target_encoder.pkl        Fitted target encoder for state/city/county
├── source_data/                  Raw input
│   └── alloy_data.csv            2,090 rows × 28 columns
├── eda/                          Exploratory analysis notebook
│   └── eda.ipynb
├── train_avm.py                  End-to-end training pipeline (raw → models)
├── similar_properties.py         Content-based property similarity
├── build_powerbi_export.py       Power BI CSV export with derived columns
├── run_app.py                    Local Streamlit launcher
├── Dockerfile                    Render deployment config
├── render.yaml                   Render service definition
├── requirements.txt
└── README.md                     ← you are here
```

---

## How to use the app

### From the live demo

Open <https://alloy-avm.streamlit.app/> and (assuming the backend is awake):

1. **Quick fill** in the sidebar — pick a sample property and click **Load sample**
2. Click **Estimate value** on the form
3. See the headline valuation, confidence range, top features that drove the prediction, and 5 comparable properties from the dataset

The two other pages are useful for analysts:

- **Market Explorer** — filter the 1,956-row dataset by state, type, price, year, size; see summary stats and a scatter of price vs. living area
- **About this model** — model cards for both AVM variants, honest limitations of the dataset, Phase 2 roadmap

### From the API directly

```bash
curl -X POST https://alloytower-avm-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "CA",
    "city": "San Francisco",
    "county": "San Francisco",
    "property_type": "Condo",
    "sqft": 1200,
    "lot_size_sqft": 0,
    "bedrooms": 2,
    "bathrooms": 2.0,
    "year_built": 2010,
    "owner_occupied": true,
    "assessed_value": 1100000
  }'
```

Returns the prediction, confidence band, and SHAP-based feature contributions.

Full API docs at <https://alloytower-avm-api.onrender.com/docs> (interactive Swagger UI).

---

## Models

Two model variants are trained, tracked in MLflow, and served by the API. The right one is chosen automatically based on whether the user provides an assessed value.

| Model                     | Use case                                                  | MAPE (5-fold CV) | R²          |
| ------------------------- | --------------------------------------------------------- | ---------------- | ----------- |
| **A — Assessment-aware**  | Property has a recent county assessment                   | **8.5% ± 1.0%**  | 0.97 ± 0.01 |
| **B — Fundamentals-only** | New build, unassessed property, or assessment unavailable | **44.9% ± 2.8%** | 0.55 ± 0.04 |

### Why the gap

Model A relies heavily on `assessed_value`, which correlates 0.99 with `last_sale_price` in this dataset. Most of Model A's accuracy comes from that relationship rather than learned valuation insight.

Model B excludes assessment data and is the more honest test of what the dataset's other features can predict. The 45% MAPE reflects a property of _this dataset_ — physical attributes (sqft, bedrooms, bathrooms) carry weak signal here — not a limitation of the AVM approach. State and city dominate predictions.

See [Known limitations](#known-limitations) below and the [Data Quality Findings Report](#deliverables) for the full picture.

---

## Running locally

### Prerequisites

- Python 3.11+
- Git
- Repo cloned to your machine

### One-time setup

```bash
git clone https://github.com/<your-username>/alloy-tower-data-platform.git
cd alloy-tower-data-platform
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train the models (if `model/models/*.pkl` aren't present)

```bash
python train_avm.py --input source_data/alloy_data.csv --output-dir model/models
```

Takes 1-2 minutes. Produces both model bundles, the cleaned CSV, and MLflow runs.

### Run the backend

In one terminal:

```bash
uvicorn api.main:app --port 8000 --reload
```

Backend lives at `http://localhost:8000`. Health check: `http://localhost:8000/health`.

### Run the frontend

In a second terminal:

```bash
streamlit run run_app.py
```

Frontend opens at `http://localhost:8501`.

If you started the backend on a different port:

```bash
AVM_API_URL=http://localhost:9000 streamlit run run_app.py
```

---

## Deliverables

This project produced the following artefacts. Most are tracked in the repo; some are documents that live separately.

| Artefact                       | Type     | What it is                                                           |
| ------------------------------ | -------- | -------------------------------------------------------------------- |
| `train_avm.py`                 | Pipeline | Raw CSV → cleaned data → 2 trained models, with MLflow tracking      |
| `similar_properties.py`        | Module   | Content-based property similarity (FR-009 stretch deliverable)       |
| `export_for_powerbi.py`      | Pipeline | Adds `assessment_flag` + `median_price_per_sqft_group` for Power BI  |
| `api/` + `app/` + `Dockerfile` | App      | Live inference application (this repo, deployed)                     |
| Project Scope & Proposal       | Document | Phase 1 scoping document for stakeholders                            |
| Data Quality Findings Report   | Document | 11 detailed findings + decisions (severity-ranked)                   |
| PowerBi Dashboard      | Document | Dashboard showcasing insights from dataset |


---

## Known limitations

These are documented in detail in the Data Quality Findings Report. Listed here in summary so anyone using the app or the model knows what they are.

- **`latitude`, `longitude`, and `zip_code` are randomly generated** in the source dataset (only 3% of coordinates match the stated state). All three columns are excluded from modelling and analytics.
- **Sale dates do not reflect real market trends.** Real US prices appreciated ~40% over 2016-2026; this dataset shows ~10% with no trend, and 2022 (a real-market peak) is the lowest year. The AVM is built as a **cross-sectional** model and does not use date-based features.
- **Physical attributes carry weaker signal than typical AVMs.** In real US data, sqft correlates 0.6-0.8 with sale price; in this dataset, 0.22. Bedrooms and bathrooms correlate near zero. As a result, **location features dominate predictions**.
- **Phase 1 dataset is small (1,956 rows after cleaning).** Production AVMs typically train on 10,000+ records. Predictions in tail cases (very small studios, very large estates, underrepresented states) carry more uncertainty.

These are dataset properties, not modelling errors. Phase 2 priorities are: real geographic identifiers, a real listings feed with active-market data, a substantially larger training set, and a real time-series component. See the **About this model** page in the live app for the full Phase 2 roadmap.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  https://alloy-avm.streamlit.app        │
│  Streamlit frontend (Streamlit Cloud)   │
│  - Property valuation page              │
│  - Market explorer                      │
│  - Model card + limitations             │
└──────────────────┬──────────────────────┘
                   │
                   │  HTTPS / JSON
                   │  (AVM_API_URL secret)
                   ↓
┌─────────────────────────────────────────┐
│  https://alloytower-avm-api.onrender.com│
│  FastAPI backend (Render free tier)     │
│  - /predict      single valuation       │
│  - /predict/batch                       │
│  - /comparables  similar properties     │
│  - /metadata     model info             │
│  - /health       liveness check         │
└──────────────────┬──────────────────────┘
                   │
                   ↓
            model/models/*.pkl
            (committed to repo)
```

**Frontend hosting:** Streamlit Community Cloud (free, auto-deploys on push to `main`).

**Backend hosting:** Render free tier (free, auto-deploys on push to `main`, sleeps after 15 min idle).

**Total cost:** $0/month.

---

## Tech stack

- **Modelling:** LightGBM (gradient-boosted trees), scikit-learn, category-encoders for target encoding
- **Explainability:** SHAP (TreeExplainer for per-prediction feature contributions)
- **Tracking:** MLflow with SQLite backend
- **Backend:** FastAPI + uvicorn + Pydantic v2
- **Frontend:** Streamlit + Altair
- **Containerisation:** Docker (slim Python 3.11 base)

---

## Updating the deployed app

Both services auto-deploy on push to `master`:

- **Frontend** redeploys via Streamlit Community Cloud (~1 min)
- **Backend** redeploys via Render (~5 min — fresh wheel install for ML deps)

To update the model in production:

```bash
python train_avm.py
git add model/models/model_A.pkl model/models/model_B.pkl model/models/alloy_clean.csv
git commit -m "Retrain models on refreshed dataset"
git push origin master
```

Render rebuilds the backend image with the new bundles. Streamlit needs no code change because it talks to the API rather than loading models directly.

---

## Repository conventions

- **Branching:** `master` is deployment-tracking; feature work in branches.
- **Commits:** descriptive messages preferred; the trained model bundles are committed (1-4 MB total) so deployments are reproducible from the repo alone.
- **Secrets:** the only secret is the API URL on Streamlit Cloud (`AVM_API_URL`); everything else is public.

---

## Contact

[Bright Uzosike](mailto:michizler@gmail.com) — Data Scientist · AlloyTower Phase 1

For technical questions about the modelling, see the Data Quality Findings Report and the in-code docstrings. For project-level questions, see the Phase 1 Scope document.
