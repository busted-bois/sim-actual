"""VQ2RealEnv -- Gymnasium environment over the REAL VQ2 simulator.

Wraps `SimInterface` (MAVLink + shared `data`) and a `GPEstimation` ego-state
estimator fed from the live IMU. The policy sees only VQ2-available signals
(YOLO+PnP gate pose + GPEstimation velocity/attitude); no ground truth. Each
`step` converts the policy action to an attitude-rate command via the lightweight
controller and sends it at DECISION_HZ; `reset` teleports + re-aligns via the
reset harness. Reward/termination come from `rl.vq2.reward`.

This is a REAL-TIME env (one instance, no fast-forward) -- see the plan; use with
a BC warm-start and expect long wall-clock.
"""

from __future__ import annotations

import math
import os
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.gp_estimation import GPEstimation
from simulator.gp_vision import _yolo_pose_estimate, best_pose_gate
from rl import spec
from rl.sim_interface import SimInterface
from rl.vq2 import controller, reward as reward_mod
from rl.vq2.observation import ObsStacker, build_frame, POLICY_OBS_DIM
from rl.vq2.reset import GatePassTracker, reset_episode

DECISION_HZ = 50
_DT = 1.0 / DECISION_HZ

# MAVLink COLLISION.threat_level: 0 NONE, 1 LOW, 2 HIGH. The sim streams these as
# PROXIMITY warnings near structures (the working GP pilot ignores low ones and
# keeps flying); only a HIGH-threat event is treated as a real crash. Override
# with env var VQ2_HARD_THREAT to recalibrate once we see live values.
HARD_THREAT = int(os.environ.get("VQ2_HARD_THREAT", "2"))


class VQ2RealEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_seconds: float = 30.0, num_gates: int = 6,
                 ip: str = "127.0.0.1", mav_port: int = 14550):
        super().__init__()
        self.sim = SimInterface(ip=ip, mav_port=mav_port)
        self.sim.data["_quiet_vision"] = True   # silence per-frame [vision] GATE spam
        self.est = GPEstimation(self.sim.data)
        self.est.start()
        self.gates = GatePassTracker()
        self.stacker = ObsStacker()
        self.max_steps = int(max_seconds * DECISION_HZ)
        self.num_gates = num_gates

        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (POLICY_OBS_DIM,), np.float32)

        self._steps = 0
        self._prev_action = np.zeros(spec.ACTION_DIM, np.float32)
        self._prev_dist: float | None = None
        self._last_collision = None

    # ---- helpers ---------------------------------------------------------
    def _read(self):
        """Return (ego snapshot, pose_est or None, conf)."""
        ego = self.est.snapshot()
        pose_est = _yolo_pose_estimate(self.sim.data)
        g = best_pose_gate(self.sim.data)
        conf = float(g["conf"]) if g else 0.0
        return ego, pose_est, conf

    def _frame(self, ego, pose_est, conf, last_action):
        return build_frame(pose_est, conf, ego, last_action=last_action)

    def _flipped(self, ego) -> bool:
        w, x, y, z = ego["quat"]
        g_down = 1 - 2 * (x * x + y * y)   # gravity_body z-component
        return g_down < 0.0

    def _out_of_bounds(self, ego) -> bool:
        p = np.asarray(ego["pos_ned"], float)   # relative to spawn (drifts)
        # loose guards; the sim COLLISION event is the primary crash signal.
        return bool(p[2] > 4.0 or p[2] < -40.0 or math.hypot(p[0], p[1]) > 60.0)

    # ---- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_episode(self.sim, self.est, settle_s=0.5)
        self.gates.reset(self.sim)
        self._steps = 0
        self._prev_action = np.zeros(spec.ACTION_DIM, np.float32)
        self._prev_dist = None
        self._last_collision = self.sim.data.get("last_collision")
        self._n_hard = self._n_soft = 0
        self._last_threat, self._last_delta = 0, -1.0
        self._ep_parts = {}
        self._ep_min_range = float("inf")
        ego, pose_est, conf = self._read()
        obs = self.stacker.reset(self._frame(ego, pose_est, conf, self._prev_action))
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)

        # act: policy -> absolute attitude (deg) -> quat wire (the SAME channel
        # the GP pilot flies on), held for one decision tick.
        roll_deg, pitch_deg, yaw_deg, th = controller.action_to_attitude(action)
        self.sim.send_attitude_quat_deg(roll_deg, pitch_deg, yaw_deg, th)
        time.sleep(_DT)

        # observe.
        ego, pose_est, conf = self._read()
        self._steps += 1

        # events.
        gate_passed = self.gates.update(self.sim, pose_est)
        course_complete = self.gates.n_passed >= self.num_gates
        # COLLISION is a threat/proximity stream, not a crash flag. Only a
        # HIGH-threat event ends the episode; low-threat = proximity nudge.
        col = self.sim.data.get("last_collision")
        col_evt = col is not None and col != self._last_collision
        self._last_collision = col
        threat = int(col[1]) if col else 0
        delta = float(col[2]) if col else -1.0
        collision = col_evt and threat >= HARD_THREAT
        soft_collision = col_evt and not collision
        if col_evt:
            self._n_hard += int(collision)
            self._n_soft += int(soft_collision)
            self._last_threat, self._last_delta = threat, delta
        flipped = self._flipped(ego)
        oob = self._out_of_bounds(ego)
        timeout = self._steps >= self.max_steps

        # distance to the currently-visible gate (for progress shaping).
        if pose_est is not None:
            dist = float(math.sqrt(pose_est["body_x_m"] ** 2 + pose_est["body_y_m"] ** 2
                                   + pose_est["body_z_m"] ** 2))
            visible = True
        else:
            dist, visible = None, False

        r, terminated, truncated, reason, parts = reward_mod.step(reward_mod.StepCtx(
            prev_dist=self._prev_dist, dist=dist, visible=visible,
            vel_body=np.asarray(ego["vel_body"], float),
            action=action, prev_action=self._prev_action,
            gate_passed=gate_passed, course_complete=course_complete,
            collision=collision, soft_collision=soft_collision,
            out_of_bounds=oob, flipped=flipped, timeout=timeout,
        ))
        # accumulate per-term reward sums + closest gate range over the episode.
        for k, v in parts.items():
            self._ep_parts[k] = self._ep_parts.get(k, 0.0) + v
        if visible and dist is not None:
            self._ep_min_range = min(self._ep_min_range, dist)

        # on a gate pass the visible gate switches (range jumps) -> drop prev_dist
        # so the switch isn't scored as a huge negative progress.
        self._prev_dist = None if (gate_passed or not visible) else dist
        self._prev_action = action

        obs = self.stacker.push(self._frame(ego, pose_est, conf, action))
        p = np.asarray(ego["pos_ned"], float)
        info = {"reason": reason, "gates_passed": self.gates.n_passed,
                "steps": self._steps,
                "gate_range": dist if dist is not None else -1.0,
                "gate_visible": visible,
                "min_gate_range": self.gates._min_r,
                "alt_m": -float(p[2]),          # NED down -> altitude above spawn
                "fwd_m": float(p[0]),           # forward distance from spawn
                "n_hard_col": self._n_hard, "n_soft_col": self._n_soft,
                "last_threat": self._last_threat, "last_delta": self._last_delta,
                "ep_min_range": (self._ep_min_range if self._ep_min_range != float("inf")
                                 else None),
                "reward_parts": dict(self._ep_parts)}
        return obs, r, terminated, truncated, info

    def close(self):
        try:
            self.est.stop()
        finally:
            self.sim.close()
