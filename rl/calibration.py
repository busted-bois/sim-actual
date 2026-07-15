"""Measured-plant calibration written by the flightlab attitude harness.

flightlab/calibration.json is produced by `make attitude-harness` (see
flightlab/run_attitude.py). Consumers fall back to their built-in defaults
when the file is absent, so behavior is identical until a measurement exists.
Keys: hover_thrust, thrust_accel, latency_ms, rate_tau_s, k_att, k_d,
rate_clip, last_stable_gain, measured_utc, source_run (all optional).
"""

from __future__ import annotations

import json
import os

CAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "flightlab",
    "calibration.json",
)


def load_calibration() -> dict:
    """Measured plant values, or {} when the file is absent/unreadable."""
    try:
        with open(CAL_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
