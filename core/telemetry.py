import csv
import os
from datetime import datetime, timezone

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