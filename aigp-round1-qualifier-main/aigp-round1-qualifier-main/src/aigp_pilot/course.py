from __future__ import annotations

import numpy as np

from aigp_pilot import raceline

GATES = [
    np.array([-23.30, -0.40, -0.05]),
    np.array([-46.89, -2.50, 5.05]),
    np.array([-74.59, 1.20, 13.65]),
    np.array([-111.49, -5.10, 24.55]),
    np.array([-135.49, -0.80, 25.34]),
    np.array([-159.19, -4.40, 25.95]),
]


# VQ2 has no position telemetry; reset INS here at race GO.
SPAWN = np.array([-8.0, -0.10, -0.05])


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


def gate_arcs(pts, cum) -> list[float]:
    return [raceline.project_arc(pts, cum, g) for g in GATES]
