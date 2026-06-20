"""Module 6 — 24-D gate-relative observation vector.

Maps the filtered drone state (Module 5) + gate map into the fixed 24-D
observation consumed by the PPO policy. Indices follow spec.OBS_LAYOUT, the
single contract shared by Modules 7 (env) and 8 (policy).

All quantities are expressed in the BODY frame and normalized to ~O(1) so the
3x64 MLP sees well-scaled inputs. Deploy and training build observations the
same way, so the sim-trained policy transfers.

    uv run -m rl.observation --selftest
"""

from __future__ import annotations

import argparse

import numpy as np

from rl import spec

DIST_SCALE = 10.0  # meters -> ~O(1)
V_SCALE = 10.0  # m/s
RATE_SCALE = spec.MAX_PITCH_RATE  # rad/s


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def build_observation(
    p: np.ndarray,
    v_world: np.ndarray,
    q: np.ndarray,
    ang_vel: np.ndarray,
    gate_map: list,
    gate_idx: int,
    last_action: np.ndarray | None = None,
) -> np.ndarray:
    """Return the 24-D observation (float32).

    p, v_world : drone position / velocity in world NED.
    q          : orientation (w,x,y,z), body->world.
    ang_vel    : body angular rates (roll,pitch,yaw) rad/s.
    gate_map   : [{pos, quat, ...}, ...].
    gate_idx   : index of the current target gate.
    last_action: last normalized [roll,pitch,yaw] rate command in [-1,1].
    """
    p = np.asarray(p, float)
    v_world = np.asarray(v_world, float)
    R_wb = spec.quat_to_R(np.asarray(q, float))
    Rt = R_wb.T  # world -> body

    n_gates = len(gate_map)
    gate_idx = int(np.clip(gate_idx, 0, max(n_gates - 1, 0)))
    cur = gate_map[gate_idx]
    nxt = gate_map[min(gate_idx + 1, n_gates - 1)]

    gc = np.asarray(cur["pos"], float)
    dvec_w = gc - p
    dist = float(np.linalg.norm(dvec_w))
    to_gate_body = Rt @ _unit(dvec_w)

    gnorm_w = spec.quat_to_R(np.asarray(cur["quat"], float)) @ np.array([1.0, 0, 0])
    gate_normal_body = Rt @ gnorm_w

    vel_body = (Rt @ v_world) / V_SCALE
    ang = np.asarray(ang_vel, float) / RATE_SCALE
    gravity_body = Rt @ np.array([0.0, 0.0, 1.0])

    yaw_align = float(np.arctan2(to_gate_body[1], to_gate_body[0]))

    nc = np.asarray(nxt["pos"], float)
    ndvec_w = nc - p
    ndist = float(np.linalg.norm(ndvec_w))
    to_next_body = Rt @ _unit(ndvec_w)

    la = np.zeros(3) if last_action is None else np.asarray(last_action, float)[:3]

    obs = np.zeros(spec.OBS_DIM, dtype=np.float32)
    L = spec.OBS_LAYOUT
    obs[L["to_gate_body"]] = to_gate_body
    obs[L["dist_to_gate"]] = min(dist / DIST_SCALE, 5.0)
    obs[L["gate_normal_body"]] = gate_normal_body
    obs[L["vel_body"]] = np.clip(vel_body, -5, 5)
    obs[L["ang_vel"]] = np.clip(ang, -3, 3)
    obs[L["gravity_body"]] = gravity_body
    obs[L["yaw_align"]] = yaw_align / np.pi
    obs[L["to_next_gate_body"]] = to_next_body
    obs[L["dist_to_next_gate"]] = min(ndist / DIST_SCALE, 5.0)
    obs[L["last_action"]] = np.clip(la, -1, 1)
    return obs


def _selftest():
    gate_map = [
        {"pos": [5.0, 0.0, 0.0], "quat": [0, 0, 0, 1], "w": 1.5, "h": 1.5},
        {"pos": [10.0, 2.0, -1.0], "quat": [0, 0, 0, 1], "w": 1.5, "h": 1.5},
    ]
    p = np.array([0.0, 0.0, 0.0])
    v = np.array([3.0, 0.0, 0.0])  # moving North (forward)
    q = np.array([1.0, 0, 0, 0])  # facing North
    obs = build_observation(
        p, v, q, np.zeros(3), gate_map, 0, last_action=np.array([0.1, -0.2, 0.0])
    )
    L = spec.OBS_LAYOUT
    assert obs.shape == (spec.OBS_DIM,), obs.shape
    assert obs.dtype == np.float32
    tg = obs[L["to_gate_body"]]
    assert tg[0] > 0.99, f"gate ahead -> body-forward, got {tg}"
    assert abs(obs[L["dist_to_gate"]][0] - 0.5) < 1e-5, "5m -> 0.5 scaled"
    assert obs[L["vel_body"]][0] > 0, "forward velocity positive in body x"
    g = obs[L["gravity_body"]]
    assert abs(g[2] - 1.0) < 1e-5, f"level flight -> gravity down body z, got {g}"
    tn = obs[L["to_next_gate_body"]]
    assert tn[0] > 0 and tn[1] > 0, "next gate ahead-right"
    assert np.all(np.isfinite(obs))
    print(f"[selftest] obs={np.round(obs, 2)}")
    print(f"[selftest] OK — 24-D obs, all slices sane, dtype={obs.dtype}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
