import csv
import os
from datetime import datetime, timezone
import pandas as pd

LOG_PATH = os.path.join("logs", "query_metrics.csv")

def log_query_event(question: str, latency_s: float, repaired: bool, success: bool, error: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp_utc", "question", "latency_s", "repaired", "success", "error"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            question[:120],
            round(latency_s, 3),
            repaired,
            success,
            error[:150],
        ])

def get_telemetry_summary() -> dict:
    if not os.path.isfile(LOG_PATH):
        return {"queries": 0, "avg_latency_s": 0.0, "repair_rate_pct": 0.0}
    df = pd.read_csv(LOG_PATH)
    if df.empty:
        return {"queries": 0, "avg_latency_s": 0.0, "repair_rate_pct": 0.0}
    return {
        "queries": len(df),
        "avg_latency_s": round(float(df["latency_s"].mean()), 2),
        "repair_rate_pct": round(100.0 * float(df["repaired"].astype(bool).mean()), 1),
    }