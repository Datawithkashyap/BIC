import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sql_engine import load_dataframe, get_schema_summary, run_sql
from core.llm_copilot import generate_sql, repair_sql
from core.telemetry import log_query_event

def build_questions(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

    qs = ["Count total rows in the dataset"]
    for c in cat_cols[:2]:
        qs.append(f"Count rows grouped by {c}")
        qs.append(f"Show top 5 {c} by row count")
    for n in num_cols[:3]:
        qs.append(f"Show overall sum, average, min, and max of {n}")
        qs.append(f"List top 5 rows with the highest {n}")
        if cat_cols:
            qs.append(f"Show total and average {n} by {cat_cols[0]}")
            qs.append(f"Which {cat_cols[0]} has the highest average {n}?")
    if len(num_cols) >= 2:
        qs.append(f"Compare total {num_cols[0]} and {num_cols[1]}")
    if date_cols and num_cols:
        qs.append(f"Show {num_cols[0]} trend ordered by {date_cols[0]}")
    return qs[:20]

def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_sales.csv"
    df = load_dataframe(dataset_path)
    schema = get_schema_summary(df)
    questions = build_questions(df)
    first_pass, healed, total = 0, 0, len(questions)
    latencies = []

    print(f"Running {total} dynamic benchmark queries on: {dataset_path}\n")
    for q in questions:
        t0 = time.perf_counter()
        repaired_flag = False
        try:
            sql = generate_sql(q, schema)
            try:
                res = run_sql(df, sql)
                first_pass += 1
                healed += 1
            except Exception as err:
                repaired_flag = True
                sql = repair_sql(q, schema, sql, str(err))
                res = run_sql(df, sql)
                healed += 1
            dt = time.perf_counter() - t0
            latencies.append(dt)
            log_query_event(q, dt, repaired_flag, True)
            print(f"[PASS{' (repaired)' if repaired_flag else ''}] {dt:.2f}s | {q} -> {len(res)} rows")
        except Exception as err:
            dt = time.perf_counter() - t0
            latencies.append(dt)
            log_query_event(q, dt, repaired_flag, False, str(err))
            print(f"[FAIL] {dt:.2f}s | {q} -> {err}")

    avg_lat = sum(latencies) / max(len(latencies), 1)
    print(f"\nBenchmark: First-Pass {first_pass}/{total} ({100*first_pass/total:.0f}%), "
          f"With Self-Healing {healed}/{total} ({100*healed/total:.0f}%), Avg Latency {avg_lat:.2f}s")

if __name__ == "__main__":
    main()