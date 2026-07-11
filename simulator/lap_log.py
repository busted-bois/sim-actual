"""Persist overnight auto-flight lap times to rl/data/."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "rl", "data")
LAPS_JSONL_PATH = os.path.join(_DATA_DIR, "auto_laps.jsonl")
BEST_LAP_TXT_PATH = os.path.join(_DATA_DIR, "best_lap.txt")


def append_lap(attempt: int, lap_s: float, best_lap_s: float) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    iso_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "ts": iso_ts,
        "attempt": int(attempt),
        "lap_s": round(float(lap_s), 2),
        "best_lap_s": round(float(best_lap_s), 2),
    }
    with open(LAPS_JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    with open(BEST_LAP_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"best={best_lap_s:.1f}s attempt={attempt} {iso_ts}\n")
