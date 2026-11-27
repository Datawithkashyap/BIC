[[https://github.com/Datawithkashyap/BIC/actions/workflows/ci.yml/badge.svg]](https://github.com/Datawithkashyap/BIC/actions/workflows/ci.yml)

# BIC — AI-Powered Business Intelligence Copilot

A local-first and cloud-deployed Business Intelligence Copilot & Data Analyst Workbench built with **Python, Streamlit, FastAPI, DuckDB, Plotly, Seaborn, Scikit-Learn, StatsForecast, Gemini, Groq, and Ollama**.

## Key Capabilities
- **Conversational NL-to-SQL (`Ask BIC`)**: Converts plain-English questions into read-only DuckDB `SELECT`/`WITH` queries with 1-step automatic SQL self-healing (`repair_sql`) and inline SQL editing.
- **Multi-Table `JOIN` & Data Prep**: Upload multiple CSV/XLSX files, apply skew-aware/KNN missing-value imputation and IQR/Z-Score outlier management.
- **Auto-EDA & Hypothesis Testing**: Column profiling, Welch's t-test, One-Way ANOVA, Chi-Square, and Pearson correlation (`p < 0.05`).
- **Visualization Studio**: 8 chart types across **Plotly** (interactive) and **Seaborn/Matplotlib** (statistical).
- **ML Lab & Forecasting**: 5-Fold CV Classification/Regression, live What-If simulator, K-Means + PCA clustering, and `AutoARIMA` forecasting.
- **FastAPI REST Backend (`api.py`)**: `/health`, `/query`, and `/profile` endpoints with structured latency telemetry.

## Quick Start

    pip install -r requirements.txt
    pytest -v
    streamlit run app.py

## Benchmark Results
- Run `python scripts/eval_sql.py` to measure First-Pass & Self-Healing NL-to-SQL accuracy and average query latency.