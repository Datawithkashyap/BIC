import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.sql_engine import load_dataframe, get_schema_summary, run_sql
from core.llm_copilot import generate_sql, repair_sql, summarize_insight
from core.analyst_workbench import profile_dataset
from core.telemetry import log_query_event

app = FastAPI(title="BIC Analytics API", version="1.0.0")

class QueryRequest(BaseModel):
    question: str
    dataset_path: str = "data/sample_sales.csv"

def _df_to_json_records(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict(orient="records")

@app.get("/health")
def health():
    return {"status": "ok", "service": "BIC Analytics API"}

@app.post("/query")
def query_data(req: QueryRequest):
    t0 = time.perf_counter()
    repaired = False
    try:
        df = load_dataframe(req.dataset_path)
        schema = get_schema_summary(df)
        sql = generate_sql(req.question, schema)
        try:
            res = run_sql(df, sql)
        except Exception as err:
            repaired = True
            sql = repair_sql(req.question, schema, sql, str(err))
            res = run_sql(df, sql)
        insight = summarize_insight(req.question, res.head(15).to_string(index=False))
        dt = round(time.perf_counter() - t0, 3)
        log_query_event(req.question, dt, repaired, True)
        return {
            "question": req.question,
            "sql": sql,
            "repaired": repaired,
            "latency_s": dt,
            "rows": _df_to_json_records(res),
            "insight": insight,
        }
    except Exception as exc:
        dt = round(time.perf_counter() - t0, 3)
        log_query_event(req.question, dt, repaired, False, str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/profile")
def profile(dataset_path: str = "data/sample_sales.csv"):
    df = load_dataframe(dataset_path)
    prof = profile_dataset(df)
    return {"rows": len(df), "columns": _df_to_json_records(prof)}