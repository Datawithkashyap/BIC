import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sql_engine import load_dataframe, get_schema_summary, run_sql
from core.llm_copilot import generate_sql, repair_sql

GROUND_TRUTH_CASES = [
    ("How many total rows are in data?", lambda res: int(res.iloc[0, 0]) == 10),
    ("What is the overall total revenue?", lambda res: int(round(float(res.iloc[0, 0]))) == 132200),
    ("What is total revenue for region North?", lambda res: int(round(float(res.iloc[0, 0]))) == 42000),
    ("How many orders are in category Electronics?", lambda res: int(res.iloc[0, 0]) == 5),
    ("Which region has the highest total revenue?", lambda res: "north" in str(res.iloc[0]).lower()),
]

def main():
    df = load_dataframe("data/sample_sales.csv")
    schema = get_schema_summary(df)
    passed = 0
    for q, check_fn in GROUND_TRUTH_CASES:
        t0 = time.perf_counter()
        try:
            sql = generate_sql(q, schema)
            try:
                res = run_sql(df, sql)
            except Exception as err:
                sql = repair_sql(q, schema, sql, str(err))
                res = run_sql(df, sql)
            ok = bool(check_fn(res))
            passed += int(ok)
            print(f"[{'PASS' if ok else 'WRONG_ANSWER'}] {time.perf_counter()-t0:.2f}s | {q}")
        except Exception as err:
            print(f"[ERROR] {time.perf_counter()-t0:.2f}s | {q} -> {err}")

    print(f"\nSemantic Accuracy: {passed}/{len(GROUND_TRUTH_CASES)} ({100*passed/len(GROUND_TRUTH_CASES):.0f}%)")

if __name__ == "__main__":
    main()