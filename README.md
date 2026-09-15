# Financial Crime Detection Engine

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_Sentinel_ZA-22c7b8?style=for-the-badge)](https://financial-crime-detection-six.vercel.app/)

**[Launch the live financial-crime command centre →](https://financial-crime-detection-six.vercel.app/)**

> **Phase 1 MVP + interactive command centre** — An end-to-end fraud/AML detection pipeline built in Python, from synthetic data generation through anomaly detection to live South African fraud simulation and plain-language risk explanations.

---

## Project Overview

This project demonstrates a complete financial crime detection pipeline suitable for a data analytics or financial-crime portfolio. It generates realistic synthetic banking data, engineers behavioural features, detects anomalies using an unsupervised Isolation Forest model, and presents the results in a polished Dash command centre.

**Key capabilities:**
- Synthetic data generation with 4 injected fraud/AML typologies
- SQLite-based data storage with proper schema design
- 10 per-account behavioural features targeting known fraud signals
- Unsupervised anomaly detection (no labelled data required)
- 0–100 risk scoring with Low/Medium/High triage bands
- Rule-based plain-language explanations for compliance analysts
- Four-view Dash command centre for monitoring and investigation
- Automatic SA Live Fraud Lab with Start, Pause, and Reset controls
- Linked South African hotspot map with clearly separated historical and simulated layers

---

## Architecture Summary

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Data Generation │────▶│   SQLite DB  │────▶│ Feature Engineer │
│  (faker + numpy) │     │  (schema.sql)│     │  (10 features)   │
└─────────────────┘     └──────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Explanations  │◀────│  Risk Scorer │◀────│ Isolation Forest │
│  (rule-based)   │     │  (0–100 band)│     │  (unsupervised)  │
└─────────────────┘     └──────────────┘     └──────────────────┘
```

**Pipeline flow:**
1. `src/data_generation.py` → Generates ~50K transactions with ~3% suspicious patterns
2. `src/db.py` → Loads data into `data/financial_crime.db`
3. `src/features.py` → Computes 10 per-account behavioural features
4. `src/model.py` → Trains Isolation Forest for unsupervised anomaly detection
5. `src/scoring.py` → Converts raw scores to 0–100 risk scores with bands
6. `src/explain.py` → Generates plain-language explanations for high-risk accounts

---

## Repository Structure

```
financial-crime-detection/
├── README.md                  ← You are here
├── requirements.txt           ← Pinned Python dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_generation.py     ← Synthetic data with fraud patterns
│   ├── db.py                  ← SQLite utilities
│   ├── features.py            ← 10 per-account features
│   ├── model.py               ← Isolation Forest training & scoring
│   ├── scoring.py             ← Risk score (0–100) + bands
│   ├── explain.py             ← Plain-language explanations
│   ├── dashboard_data.py      ← Shared dashboard analytics loader
│   ├── hotspots.py            ← Curated South African hotspot context
│   └── simulator.py           ← Automatic live-event simulation and scoring
├── assets/
│   └── command-center.css     ← Dash command-centre design system
├── app.py                     ← Dash application and four dashboard views
├── sql/
│   └── schema.sql             ← Database schema (4 tables)
├── notebooks/
│   ├── 01_eda.ipynb           ← Exploratory data analysis
│   └── 02_results.ipynb       ← Full pipeline results
├── data/
│   └── financial_crime.db     ← Included synthetic demo dataset
├── reports/
│   └── *.png / *.csv          ← Output charts & tables
└── tests/                     ← Analytics, dashboard, hotspot and simulator tests
```

---

## How to Run

### 1. Setup

```bash
# Clone the repository
git clone <repo-url>
cd financial-crime-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install runtime dependencies
pip install -r requirements.txt

# Include notebook, data-generation, and test tooling for development
pip install -r requirements-dev.txt
```

### 2. Generate Data

```bash
python -m src.data_generation
```

This creates `data/financial_crime.db` with ~500 customers, ~700 accounts, ~50 merchants, and ~50,000 transactions (~3% flagged as suspicious).

### 3. Run EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Or run non-interactively:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```

### 4. Run Results Notebook

```bash
jupyter notebook notebooks/02_results.ipynb
```

Or run non-interactively:
```bash
jupyter nbconvert --to notebook --execute notebooks/02_results.ipynb
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Launch the Command Centre

```bash
python app.py
```

Open `http://127.0.0.1:8050/` if it does not open automatically.

The application contains four views:

1. **Executive Overview** — network metrics, transaction velocity, exposure, and risk distribution.
2. **High-Risk Alerts** — prioritised alert queue with explanations and linked investigations.
3. **Account Deep Dive** — account-level behavioural profile, timeline, and transaction log.
4. **SA Live Fraud Lab** — one-click automatic transaction simulation, live scoring, alerts, feed, and a South African hotspot map.

The simulator runs entirely in the browser session and does not write generated events back to the SQLite database. Click **Start Simulation** to begin; no transaction entry is required.

### Deploy to Vercel

The repository exports the underlying Flask WSGI application for Vercel while retaining Dash for local development. The included database contains synthetic demonstration data only.

```bash
npx vercel
npx vercel --prod
```

Vercel uses the root `requirements.txt` for the production runtime and `.python-version` for Python 3.12. Development-only notebook and test packages are kept in `requirements-dev.txt` to reduce the serverless bundle size.

### South African hotspot context

The map combines two visually distinct layers:

- **Historical context:** province-level banking fraud distribution curated from the [SABRIC 2024 Crime Statistics Report](https://www.sabric.co.za/wp-content/uploads/2025/09/CRIME-STATISTICS-REPORT-2024.pdf) and selected commercial-crime station context from [SAPS January–March 2025 statistics](https://www.saps.gov.za/services/downloads/2024/2024-2025_Q4_crime_stats.pdf).
- **Simulated activity:** automatically generated events produced by this project and scored with the existing analytics pipeline plus transparent event-level rules.

Historical geography provides contextual awareness only. It does not prove that a person, transaction, city, or province is fraudulent and is not used on its own to determine individual risk.

---

## Key Findings from EDA

| Finding | Chart |
|---------|-------|
| Transaction volume is uniform over 12 months; injected spikes are clearly visible at account level | `daily_monthly_volume.png` |
| Suspicious transactions have ~4× higher cross-border ratio than normal ones | `cross_border_share.png` |
| Clear clustering of amounts at R9,000–R9,999 reveals structuring patterns | `near_threshold_amounts.png` |
| Dormant accounts show unexpected activity bursts from reactivation injection | `dormant_account_analysis.png` |
| Long right tail in per-account velocity identifies volume-spike accounts | `transaction_velocity.png` |

---

## Model Choice & Limitations

### Why Isolation Forest?

1. **Unsupervised** — No labelled fraud data is available in production settings; Isolation Forest doesn't require labels.
2. **Efficient** — Scales well on tabular data with our ~700-account feature matrix.
3. **Interpretable** — Feature importances can be approximated via permutation importance.
4. **Contamination-aware** — The `contamination` parameter lets us hint at the expected anomaly rate (~3–5%).

### Limitations

- **Single model** — No ensemble or model comparison (Random Forest, XGBoost, Autoencoders).
- **Demonstration stream** — Live events are simulated in-session; this is not connected to a bank payment rail or production message broker.
- **Synthetic data** — Patterns are deliberately injected; real-world fraud is more subtle and evolving.
- **Contextual hotspot data** — Public reports provide aggregated geographic context, not a live official fraud-event feed.
- **No concept drift** — Model doesn't adapt to changing patterns over time.
- **Simple explanations** — Rule-based z-score thresholds, not SHAP/LIME-level feature attributions.

---

## Future Work

The following are **not built** in this Phase 1 MVP but represent production-ready next steps:

| Category | Enhancement |
|----------|-------------|
| **Graph/Network Analysis** | Build transaction networks to detect money laundering rings and mule chains |
| **Production Streaming** | Connect the simulator interface to an authorised Kafka/Flink or payment-event feed |
| **Containerization** | Docker + docker-compose for reproducible deployment |
| **CI/CD** | GitHub Actions pipeline for automated testing and deployment |
| **Production Infrastructure** | Managed database, authentication, audit logging, and scalable model serving |
| **MLOps & Monitoring** | Model versioning, A/B testing, concept drift detection (Evidently AI) |
| **Multi-Model Comparison** | Benchmark Isolation Forest vs. Autoencoders, XGBoost, LOF |
| **Advanced Explainability** | SHAP values, LIME, or Anchor explanations for model decisions |
| **Feature Store** | Centralized feature management (Feast, Tecton) |
| **Alert Management** | Case management UI for analysts to review and disposition alerts |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11+ | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | Isolation Forest model |
| matplotlib / seaborn | Visualization |
| faker | Synthetic data generation |
| SQLite | Lightweight database (no server needed) |
| pytest | Unit testing |
| Jupyter | Interactive analysis notebooks |
| Dash / Dash Bootstrap Components | Multi-view command-centre application |
| Dash AG Grid | Investigation and live-event data grids |
| Plotly | Interactive charts and South African risk map |

---

## License

This is a portfolio/educational project. Feel free to fork and adapt.
