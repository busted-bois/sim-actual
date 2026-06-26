from __future__ import annotations

import numpy as np

from rl.fastpath import raceline

# Hardcoded gate centers in world NED (meters). These match rl/data/gate_map.json
# exactly (same sim, same track) -- the descending/climbing 6-gate course.
# True gate centers (from the sim / rl/data/gate_map.json). Used for the
# closest-approach DIAGNOSTICS so miss numbers are always measured against the
# real openings, independent of any aim tuning below.
TRUE_GATES = [
    np.array([-23.30, -0.40, -0.05]),
    np.array([-46.89, -2.50, 5.05]),
    np.array([-74.59, 1.20, 13.65]),
    np.array([-111.49, -5.10, 24.55]),
    np.array([-135.49, -0.80, 25.34]),
    np.array([-159.19, -4.40, 25.95]),
]

# AIM points the raceline is built through. Start equal to the true centers;
# NOTE: nudging a single control point reshapes the Catmull-Rom spline (sharper
# peak -> bigger overshoot), so it is NOT a clean linear knob -- prefer gains.
GATES = [np.array(g, dtype=float) for g in TRUE_GATES]


def build_path(start):
    direction = GATES[-1] - GATES[-2]
    direction = direction / float(np.linalg.norm(direction))
    control_points = [np.asarray(start, float), *GATES, GATES[-1] + direction * 8.0]
    pts = raceline.catmull_rom(control_points, 18)
    return pts, raceline.arc_lengths(pts)


def build_profile(pts, cum, cfg):
    return raceline.speed_profile(
        pts,
        cum,
        cfg.a_lat,
        cfg.a_lon,
        cfg.a_brk,
        cfg.v_max,
        cfg.v_min,
        cfg.v_start,
        climb_max=cfg.climb_max,
    )
