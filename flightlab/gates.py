"""Resolve first-gate pass altitude (NED z, down-positive) + gate normal."""

from __future__ import annotations

import numpy as np

from rl.fly2_course import detect_climb_course, resolve_gate_map
from rl.spec import quat_to_R
from simulator.vq2_pose import gate_z_ned, spawn_position_ned

GATE_PASS_ZOFF = 0.0
# Max climb above current pose (m). Hard cap — stops ballooning.
MAX_CLIMB_M = 1.0
DEFAULT_CLIMB_M = 1.0

# Normal-vector approach (spec): n_hat = R_gate @ [0,0,1];
# p_approach = p_gate + d_offset * n_hat.
D_OFFSET_M = 5.0  # approach point distance in front of gate
D_THROUGH_M = 2.0  # exit point behind gate plane
# If local +Z is not the through-axis (pure-yaw gate quats keep it vertical),
# fall back to the gate axis best aligned with the drone->gate direction.
MIN_AXIS_ALIGN = 0.5


def first_gate_pass_z(data: dict, spawn_z: float) -> tuple[float, dict]:
    """NED z for gate-0 pass height, taken as a small move from spawn.

    On this course spawn is already ~1 m above gate centre, so we usually
    **hold** rather than climb. Only climb (capped) if the map says the
    opening is above us.
    """
    gm = resolve_gate_map(data)
    if not gm:
        z = spawn_z - DEFAULT_CLIMB_M
        return z, {
            "source": "default_climb",
            "z_hold": z,
            "spawn_z": spawn_z,
            "climb_m": DEFAULT_CLIMB_M,
            "alt_m": DEFAULT_CLIMB_M,
        }

    flipz = detect_climb_course(gm)
    z_gate = gate_z_ned(gm[0]["pos"], flipz)
    z_pass_world = z_gate + GATE_PASS_ZOFF
    z_spawn_approx = float(spawn_position_ned(gm, flipz=flipz)[2])
    # World Δz: negative => pass is above spawn => climb.
    delta = z_pass_world - z_spawn_approx
    if delta < 0:
        climb = min(MAX_CLIMB_M, -delta)
        z_hold = spawn_z - climb
    else:
        # Already at/above pass height — hold pad altitude (don't climb).
        climb = 0.0
        z_hold = spawn_z

    return z_hold, {
        "source": "gate0",
        "flipz": flipz,
        "gate0_z": z_gate,
        "z_pass_world": z_pass_world,
        "z_spawn_approx": z_spawn_approx,
        "delta_z": delta,
        "climb_m": climb,
        "z_hold": z_hold,
        "spawn_z": spawn_z,
        "n_gates": len(gm),
        "alt_m": -z_hold if spawn_z == 0 else climb,
    }


# ---------------------------------------------------------------------------
# Gate normal vector — perpendicular approach
# ---------------------------------------------------------------------------


def gate_pos_ned(gate: dict, flipz: bool) -> np.ndarray:
    """Gate centre in NED with the captured-map z convention normalized."""
    pos = gate["pos"]
    return np.array([float(pos[0]), float(pos[1]), gate_z_ned(pos, flipz)], dtype=float)


def _axis_ned(r: np.ndarray, axis: np.ndarray, flipz: bool) -> np.ndarray:
    v = r @ axis
    if flipz:
        v = np.array([v[0], v[1], -v[2]], dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def gate_normal_ned(gate: dict, drone_pos, flipz: bool) -> np.ndarray:
    """Unit gate normal (NED), sign-flipped to point toward the drone.

    Spec: n_hat = R_gate @ [0,0,1]. On this course the captured quats are
    pure yaw, which keeps local +Z vertical (in the gate plane, not through
    it) — then fall back to the gate axis best aligned with the drone->gate
    line so the approach point still lands in front of the opening.
    """
    r = quat_to_R(np.asarray(gate["quat"], dtype=float))
    p_gate = gate_pos_ned(gate, flipz)
    to_drone = np.asarray(drone_pos, dtype=float) - p_gate
    d = float(np.linalg.norm(to_drone))
    u_drone = to_drone / d if d > 1e-9 else np.array([1.0, 0.0, 0.0])

    n_hat = _axis_ned(r, np.array([0.0, 0.0, 1.0]), flipz)  # spec axis
    if abs(float(n_hat @ u_drone)) < MIN_AXIS_ALIGN:
        candidates = [
            _axis_ned(r, np.array([1.0, 0.0, 0.0]), flipz),
            _axis_ned(r, np.array([0.0, 1.0, 0.0]), flipz),
        ]
        n_hat = max(candidates, key=lambda v: abs(float(v @ u_drone)))

    if float(n_hat @ u_drone) < 0.0:
        n_hat = -n_hat
    return n_hat


def approach_point(
    gate: dict,
    drone_pos,
    flipz: bool,
    d_offset: float = D_OFFSET_M,
) -> tuple[np.ndarray, np.ndarray]:
    """(p_approach, n_hat): p_gate + d_offset * n_hat, normal toward drone."""
    n_hat = gate_normal_ned(gate, drone_pos, flipz)
    return gate_pos_ned(gate, flipz) + d_offset * n_hat, n_hat


def through_point(
    gate: dict,
    drone_pos,
    flipz: bool,
    d_through: float = D_THROUGH_M,
) -> np.ndarray:
    """Exit point d_through metres behind the gate plane (along -n_hat)."""
    p, n_hat = approach_point(gate, drone_pos, flipz, d_offset=-d_through)
    return p


def resolve_course(data: dict) -> tuple[list, bool]:
    """(gate_map, flipz) from the live/captured map; ([], False) if absent."""
    gm = resolve_gate_map(data)
    return gm, (detect_climb_course(gm) if gm else False)


def gate_geometry(gate: dict, drone_pos, flipz: bool) -> dict:
    """Approach geometry for one gate from the drone's current position."""
    p_appr, n_hat = approach_point(gate, drone_pos, flipz)
    return {
        "gate": gate,
        "flipz": flipz,
        "p_gate": gate_pos_ned(gate, flipz),
        "n_hat": n_hat,
        "p_approach": p_appr,
        "p_through": through_point(gate, drone_pos, flipz),
        "half_w": 0.5 * float(gate.get("w", 2.72)),
        "half_h": 0.5 * float(gate.get("h", 2.72)),
    }


def first_gate_geometry(data: dict, drone_pos) -> dict | None:
    """Gate-0 approach geometry from the live/captured map, or None."""
    gm, flipz = resolve_course(data)
    if not gm:
        return None
    return gate_geometry(gm[0], drone_pos, flipz)
