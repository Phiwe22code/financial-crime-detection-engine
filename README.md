# Financial Crime Detection Engine

> **Phase 1 MVP** — An end-to-end fraud/AML detection pipeline built in Python, from synthetic data generation through anomaly detection to plain-language risk explanations.

---

## Project Overview

This project demonstrates a complete financial crime detection pipeline suitable for a junior data analyst portfolio. It generates realistic synthetic banking data, engineers behavioural features, detects anomalies using an unsupervised Isolation Forest model, and produces human-readable explanations for high-risk accounts.

**Key capabilities:**
- Synthetic data generation with 4 injected fraud/AML typologies
- SQLite-based data storage with proper schema design
- 10 per-account behavioural features targeting known fraud signals
- Unsupervised anomaly detection (no labelled data required)
- 0–100 risk scoring with Low/Medium/High triage bands
- Rule-based plain-language explanations for compliance analysts

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
│   └── explain.py             ← Plain-language explanations
├── app.py                     ← Streamlit Dashboard application
├── sql/
│   └── schema.sql             ← Database schema (4 tables)
├── notebooks/
│   ├── 01_eda.ipynb           ← Exploratory data analysis
│   └── 02_results.ipynb       ← Full pipeline results
├── data/
│   └── financial_crime.db     ← Generated (gitignored)
├── reports/
│   └── *.png / *.csv          ← Output charts & tables
└── tests/
    └── test_features.py       ← Unit tests for features
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

# Install dependencies
pip install -r requirements.txt
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

### 6. Launch Dashboard (New!)

```bash
streamlit run app.py
```
This will open an interactive Command Center in your browser where you can view high-level metrics, the alert queue, and perform a deep dive into flagged accounts.

---

## Key Findings from EDA

| Finding | Chart |
|---------|-------|
| Transaction volume is uniform over 12 months; injected spikes are clearly visible at account level | `daily_monthly_volume.png` |
| Suspicious transactions have ~4× higher cross-border ratio than normal ones | `cross_border_share.png` |
| Clear clustering of amounts at $9,000–$9,999 reveals structuring patterns | `near_threshold_amounts.png` |
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
- **Static batch** — Processes all data at once; no real-time or streaming capability.
- **Synthetic data** — Patterns are deliberately injected; real-world fraud is more subtle and evolving.
- **No concept drift** — Model doesn't adapt to changing patterns over time.
- **Simple explanations** — Rule-based z-score thresholds, not SHAP/LIME-level feature attributions.

---

## Future Work

The following are **not built** in this Phase 1 MVP but represent production-ready next steps:

| Category | Enhancement |
|----------|-------------|
| **Graph/Network Analysis** | Build transaction networks to detect money laundering rings and mule chains |
| **Real-Time Streaming** | Kafka/Flink pipeline for live transaction scoring |
| **Containerization** | Docker + docker-compose for reproducible deployment |
| **CI/CD** | GitHub Actions pipeline for automated testing and deployment |
| **Cloud Deployment** | AWS/GCP/Azure deployment with managed ML services |
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
| Streamlit | Interactive dashboard UI |
| Plotly | Interactive dashboard charts |

---

## License

This is a portfolio/educational project. Feel free to fork and adapt.
