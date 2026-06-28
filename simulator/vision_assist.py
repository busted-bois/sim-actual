"""Vision geometry assist for telemetry fallback (no full EKF)."""

from __future__ import annotations

import math


def blend_telemetry_bearing(
    bearing_error_mavlink: float,
    gate_target: dict | None,
    max_age_s: float = 1.0,
    now: float | None = None,
) -> float:
    """Blend mavlink bearing error with vision bearing when geometry is confident.

    Uses gate_target bearing_rad from GateEstimator (camera frame). Helps telemetry
    mode when mavlink position drifts but recent vision geometry is available.
    """
    if not gate_target:
        return bearing_error_mavlink

    confidence = float(gate_target.get("confidence", 0.0) or 0.0)
    bearing_rad = gate_target.get("bearing_rad")
    if bearing_rad is None or confidence < 0.4:
        return bearing_error_mavlink

    if now is not None:
        cam = gate_target.get("_camera_received_at")
        if cam is not None and now - cam > max_age_s:
            return bearing_error_mavlink

    vision_err = float(bearing_rad)
    w = min(confidence, 0.6) * 0.35
    blended = (1.0 - w) * bearing_error_mavlink + w * vision_err
    return (blended + math.pi) % (2 * math.pi) - math.pi
