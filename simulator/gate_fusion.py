"""Tier 2 drift correction from gate PnP and known gate pose."""

from __future__ import annotations

import math

from simulator.flight_config import PNP_BLEND
from simulator.racing_planner import gate_yaw_from_orientation


def _normalize_angle(error):
    while error > math.pi:
        error -= 2.0 * math.pi
    while error < -math.pi:
        error += 2.0 * math.pi
    return error


def implied_position_from_pnp(state, gate, pnp_result):
    """Estimate NED position from gate map pose and camera-to-gate translation."""
    gx, gy, gz = gate["position_ned"]
    gate_yaw = gate_yaw_from_orientation(gate["orientation_ned"])
    range_m = float(pnp_result["range_m"])
    lateral_m = float(pnp_result["lateral_m"])

    approach_x = math.cos(gate_yaw)
    approach_y = math.sin(gate_yaw)
    left_x = -approach_y
    left_y = approach_x

    # Camera looks toward gate; stand range_m before the gate plane along approach.
    px = gx - approach_x * range_m + left_x * lateral_m
    py = gy - approach_y * range_m + left_y * lateral_m
    pz = gz
    return px, py, pz


def apply_pnp_to_state(state, gate, pnp_result, blend=PNP_BLEND):
    if pnp_result is None:
        return state

    px, py, pz = implied_position_from_pnp(state, gate, pnp_result)
    updated = dict(state)
    alpha = blend
    updated["x"] = (1.0 - alpha) * updated["x"] + alpha * px
    updated["y"] = (1.0 - alpha) * updated["y"] + alpha * py
    updated["z"] = (1.0 - alpha) * updated["z"] + alpha * pz

    yaw_corr = float(pnp_result.get("yaw_correction", 0.0))
    updated["yaw"] = _normalize_angle(updated["yaw"] + alpha * yaw_corr)
    return updated
