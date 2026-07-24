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


class VQ2RealEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_seconds: float = 30.0, num_gates: int = 6,
                 ip: str = "127.0.0.1", mav_port: int = 14550):
        super().__init__()
        self.sim = SimInterface(ip=ip, mav_port=mav_port)
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
        ego, pose_est, conf = self._read()
        obs = self.stacker.reset(self._frame(ego, pose_est, conf, self._prev_action))
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)

        # act: policy angle-targets -> rates -> MAVLink, held for one decision tick.
        ego = self.est.snapshot()
        roll = math.radians(ego["att_deg"][0])
        pitch = math.radians(ego["att_deg"][1])
        rr, pr, yr, th = controller.action_to_rates(action, roll, pitch)
        self.sim.send_attitude_rates(rr, pr, yr, th)
        time.sleep(_DT)

        # observe.
        ego, pose_est, conf = self._read()
        self._steps += 1

        # events.
        gate_passed = self.gates.update(self.sim, pose_est)
        course_complete = self.gates.n_passed >= self.num_gates
        col = self.sim.data.get("last_collision")
        collision = col is not None and col != self._last_collision
        self._last_collision = col
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

        r, terminated, truncated, reason = reward_mod.step(reward_mod.StepCtx(
            prev_dist=self._prev_dist, dist=dist, visible=visible,
            vel_body=np.asarray(ego["vel_body"], float),
            action=action, prev_action=self._prev_action,
            gate_passed=gate_passed, course_complete=course_complete,
            collision=collision, out_of_bounds=oob, flipped=flipped, timeout=timeout,
        ))

        # on a gate pass the visible gate switches (range jumps) -> drop prev_dist
        # so the switch isn't scored as a huge negative progress.
        self._prev_dist = None if (gate_passed or not visible) else dist
        self._prev_action = action

        obs = self.stacker.push(self._frame(ego, pose_est, conf, action))
        info = {"reason": reason, "gates_passed": self.gates.n_passed,
                "steps": self._steps}
        return obs, r, terminated, truncated, info

    def close(self):
        try:
            self.est.stop()
        finally:
            self.sim.close()
