"""VQ2 RL observation: vision gate-pose + GPEstimation ego-state -> stacked vector.

Reuses `rl.observation.build_observation_body` for the 24-D body-frame core (the
single obs contract shared with the headless pipeline), then appends detection
confidence, a gate-visible flag, and the attitude quaternion. Frames are stacked
(default 6) to give the MLP short-term memory so brief gate dropouts don't blind
the policy. Everything here comes from VQ2-available signals only.

Frame layout (FRAME_DIM = 30):
    [0:24]  build_observation_body core (to_gate, dist, gate_normal, vel_body,
            ang_vel, gravity_body, to_next=0, dist_next=cap, last_action[:3])
    [24]    confidence  (YOLO box conf of the chosen gate, 0 if none)
    [25]    gate_visible (1.0 if a fresh confident gate pose exists else 0.0)
    [26:30] attitude quaternion (w,x,y,z) from GPEstimation
"""

from __future__ import annotations

from collections import deque

import numpy as np

from rl import spec
from rl.observation import build_observation_body

# Distance used for "gate unknown / far" (matches the obs core's cap of 5.0 after
# /DIST_SCALE; anything >= this normalizes to the max).
_DIST_UNKNOWN = 10.0

FRAME_DIM = spec.OBS_DIM + 2 + 4          # 24 + conf + visible + quat = 30
STACK = 6
POLICY_OBS_DIM = FRAME_DIM * STACK        # 180


def _gravity_body(q) -> np.ndarray:
    """Gravity NED (0,0,1) rotated into body FRD via q=(w,x,y,z) Body->World:
    g_body = R_wb^T @ [0,0,1] (third column of R_wb^T)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([2 * (x * z - w * y),
                     2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)], dtype=np.float64)


def build_frame(pose_est: dict | None, conf: float, ego: dict,
                last_action=None) -> np.ndarray:
    """One FRAME_DIM observation frame.

    pose_est    : `_yolo_pose_estimate(data)` dict or None (no fresh gate).
    conf        : YOLO box confidence of the chosen gate (0.0 if none).
    ego         : `GPEstimation.snapshot()` dict (quat, vel_body, rates_body_dps).
    last_action : normalized [-1,1]^4 previous action (or None).
    """
    q = np.asarray(ego["quat"], dtype=np.float64)                       # (w,x,y,z)
    vel_body = np.asarray(ego["vel_body"], dtype=np.float64)
    ang_vel = np.radians(np.asarray(ego["rates_body_dps"], dtype=np.float64))  # deg/s -> rad/s
    gravity_body = _gravity_body(q)

    if pose_est is not None:
        to_gate = np.array([pose_est["body_x_m"], pose_est["body_y_m"],
                            pose_est["body_z_m"]], dtype=np.float64)
        dist = float(np.linalg.norm(to_gate))
        normal = np.asarray(pose_est.get("normal_body", [1.0, 0.0, 0.0]),
                            dtype=np.float64)
        visible = 1.0
    else:
        to_gate = np.zeros(3, dtype=np.float64)
        dist = _DIST_UNKNOWN
        normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        visible = 0.0
        conf = 0.0

    core = build_observation_body(
        to_gate, dist, normal, vel_body, ang_vel, gravity_body,
        to_next_gate_body=np.zeros(3, dtype=np.float64),
        dist_to_next_gate=_DIST_UNKNOWN,
        last_action=(None if last_action is None
                     else np.asarray(last_action, dtype=np.float64)[:3]),
    )
    return np.concatenate([core, [float(conf), float(visible)], q]).astype(np.float32)


class ObsStacker:
    """Fixed-length frame stack -> flat POLICY_OBS_DIM vector."""

    def __init__(self, stack: int = STACK):
        self.stack = stack
        self._buf: deque = deque(maxlen=stack)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._buf.clear()
        for _ in range(self.stack):
            self._buf.append(frame)
        return self.get()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._buf.append(frame)
        return self.get()

    def get(self) -> np.ndarray:
        return np.concatenate(list(self._buf)).astype(np.float32)


if __name__ == "__main__":  # quick sanity
    ego = {"quat": np.array([1.0, 0, 0, 0]), "vel_body": np.array([3.0, 0, 0]),
           "rates_body_dps": np.array([0.0, 0, 0])}
    pose = {"body_x_m": 5.0, "body_y_m": 0.0, "body_z_m": 0.0,
            "normal_body": np.array([1.0, 0, 0])}
    f_vis = build_frame(pose, 0.9, ego, last_action=np.zeros(4))
    f_blind = build_frame(None, 0.0, ego, last_action=np.zeros(4))
    assert f_vis.shape == (FRAME_DIM,), f_vis.shape
    assert f_vis[25] == 1.0 and f_blind[25] == 0.0        # visible flag
    assert not np.isnan(f_vis).any() and not np.isnan(f_blind).any()
    st = ObsStacker()
    o = st.reset(f_vis)
    assert o.shape == (POLICY_OBS_DIM,), o.shape
    print("[vq2.observation] selftest OK  frame", FRAME_DIM, "policy", POLICY_OBS_DIM)
