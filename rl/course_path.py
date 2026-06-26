"""Build spline raceline from gate_map.json."""

from __future__ import annotations

import json

import numpy as np

from rl import raceline
from rl.sim_interface import GATE_MAP_PATH


def load_gates(path: str = GATE_MAP_PATH) -> list[np.ndarray]:
    raw = json.load(open(path))["gates"]
    return [np.asarray(g["pos"], dtype=float) for g in raw]


def build_path(start, gates: list[np.ndarray] | None = None):
    if gates is None:
        gates = load_gates()
    direction = gates[-1] - gates[-2]
    direction = direction / float(np.linalg.norm(direction))
    control_points = [np.asarray(start, float), *gates, gates[-1] + direction * 8.0]
    pts = raceline.catmull_rom(control_points, 18)
    return pts, raceline.arc_lengths(pts)


def build_profile(
    pts,
    cum,
    *,
    v_max: float = 9.0,
    v_min: float = 2.5,
    v_start: float = 1.0,
    a_lat: float = 5.5,
    a_lon: float = 6.0,
    a_brk: float = 5.0,
    climb_max: float = 2.3,
):
    return raceline.speed_profile(
        pts,
        cum,
        a_lat,
        a_lon,
        a_brk,
        v_max,
        v_min,
        v_start,
        climb_max=climb_max,
    )
