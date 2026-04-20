import csv
import os
from pathlib import Path
from datetime import datetime
from threading import Lock

LOG_DIR  = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "log_predictions.csv"

FIELDNAMES = [
    "timestamp",
    "SK_ID_CURR",
    "probability_default",
    "risk_level",
    "inference_time_ms",
    "status",        # "success" | "error"
    "error_message", # empty on success
]

_lock = Lock()


def _ensure_log_file():
    LOG_DIR.mkdir(exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def log_prediction(
    sk_id: int,
    probability: float = None,
    risk_level: str = None,
    inference_time_ms: float = None,
    status: str = "success",
    error_message: str = "",
):
    _ensure_log_file()
    row = {
        "timestamp":           datetime.now().isoformat(),
        "SK_ID_CURR":          sk_id,
        "probability_default": round(probability, 6) if probability is not None else "",
        "risk_level":          risk_level or "",
        "inference_time_ms":   round(inference_time_ms, 3) if inference_time_ms is not None else "",
        "status":              status,
        "error_message":       error_message,
    }
    with _lock:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)


def read_logs() -> list[dict]:
    """Read all prediction logs as a list of dicts."""
    _ensure_log_file()
    with _lock:
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
