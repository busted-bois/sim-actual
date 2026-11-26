"""Resolve first-gate pass altitude (NED z, down-positive)."""

from __future__ import annotations

from rl.fly2_course import detect_climb_course, resolve_gate_map
from simulator.vq2_pose import gate_z_ned, spawn_position_ned

GATE_PASS_ZOFF = 0.0
# Max climb above current pose (m). Hard cap — stops ballooning.
MAX_CLIMB_M = 1.0
DEFAULT_CLIMB_M = 1.0


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
