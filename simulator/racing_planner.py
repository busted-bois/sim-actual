"""Gate-plane racing line and pure-pursuit targets (Tier 2 TOGT-lite)."""

from __future__ import annotations

import math

from simulator.flight_config import (
    GATE_CORNER_CUT,
    GATE_INNER_M,
    PURE_PURSUIT_LOOKAHEAD_M,
)


def _normalize_xy(dx, dy):
    dist = math.hypot(dx, dy)
    if dist <= 1e-6:
        return 1.0, 0.0, 0.0
    return dx / dist, dy / dist, dist


def gate_yaw_from_orientation(orientation_ned):
    qw, qx, qy, qz = orientation_ned
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def gate_forward_xy(gate):
    if "orientation_ned" in gate:
        yaw = gate_yaw_from_orientation(gate["orientation_ned"])
        return math.cos(yaw), math.sin(yaw)
    return 1.0, 0.0


def _toward_next_xy(gate, next_gate):
    gx, gy, _gz = gate["position_ned"]
    if next_gate is not None:
        nx, ny, _nz = next_gate["position_ned"]
        return _normalize_xy(nx - gx, ny - gy)
    fx, fy = gate_forward_xy(gate)
    return fx, fy, 0.0


def gate_crossing_point(gate, next_gate=None, prev_gate=None):
    """TOGT-lite: cut the inner gate opening toward the next gate."""
    position = gate.get("position_ned")
    if position is None:
        return None
    gx, gy, gz = position
    tx, ty, _ = _toward_next_xy(gate, next_gate)
    lateral_x, lateral_y = -ty, tx

    if prev_gate is not None:
        px, py, _ = prev_gate["position_ned"]
        incoming_x, incoming_y, _ = _normalize_xy(gx - px, gy - py)
        turn = incoming_x * ty - incoming_y * tx
        if abs(turn) > 0.15:
            sign = 1.0 if turn >= 0.0 else -1.0
            half = 0.5 * GATE_INNER_M * GATE_CORNER_CUT
            cx = gx + tx * 0.15 + lateral_x * half * sign
            cy = gy + ty * 0.15 + lateral_y * half * sign
            return cx, cy, gz

    return gx + tx * 0.15, gy + ty * 0.15, gz


def precompute_racing_path(gates):
    """Crossing points for each gate in order."""
    if not gates:
        return []
    points = []
    for index, gate in enumerate(gates):
        prev_gate = gates[index - 1] if index > 0 else None
        next_gate = gates[index + 1] if index + 1 < len(gates) else None
        point = gate_crossing_point(gate, next_gate, prev_gate)
        if point is not None:
            points.append(point)
    return points


def _segment_target(px, py, ax, ay, bx, by, lookahead_m):
    seg_x, seg_y = bx - ax, by - ay
    seg_len = math.hypot(seg_x, seg_y)
    if seg_len <= 1e-6:
        return bx, by

    ux, uy = seg_x / seg_len, seg_y / seg_len
    rel_x, rel_y = px - ax, py - ay
    along = rel_x * ux + rel_y * uy
    target_along = min(seg_len, max(0.0, along + lookahead_m))
    return ax + ux * target_along, ay + uy * target_along


def pure_pursuit_target(px, py, pz, gates, active_index, racing_path=None):
    """Carrot target ahead on the racing line through gate crossing points."""
    if not gates or active_index < 0 or active_index >= len(gates):
        return px, py, pz

    if racing_path is None:
        racing_path = precompute_racing_path(gates)

    gate_point = racing_path[active_index]
    gx, gy, gz = gate_point

    if active_index + 1 < len(gates):
        nx, ny, nz = racing_path[active_index + 1]
        tx, ty = _segment_target(px, py, gx, gy, nx, ny, PURE_PURSUIT_LOOKAHEAD_M)
        tz = gz + 0.35 * (nz - gz)
        return tx, ty, tz

    fx, fy = gate_forward_xy(gates[active_index])
    return gx + fx * PURE_PURSUIT_LOOKAHEAD_M, gy + fy * PURE_PURSUIT_LOOKAHEAD_M, gz


def pursuit_target_from_data(pose, data, racing_path=None):
    gates = data.get("track_gates") or []
    race = data.get("race_status") or {}
    active_index = int(race.get("active_gate_index", 0))
    return pure_pursuit_target(
        pose["x"],
        pose["y"],
        pose["z"],
        gates,
        active_index,
        racing_path=racing_path,
    )
