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
from rl.vq2 import controller, reward as reward_mod, success
from rl.vq2.observation import ObsStacker, build_frame, POLICY_OBS_DIM
from rl.vq2.reset import GatePassTracker, reset_episode

DECISION_HZ = 50
_DT = 1.0 / DECISION_HZ

# The VQ2 course has 17 gates (NOT 6).
DEFAULT_NUM_GATES = 17

# MAVLink COLLISION is a PROXIMITY stream, not a crash flag. MEASURED: threat=2
# events fire at horizontal_minimum_delta ~2.5-3 m -- i.e. 3 m AWAY from the gate
# frame, which the drone MUST approach to thread the 1.5 m opening. The proven GP
# pilot ignores collisions and flies through. So a collision terminates the
# episode ONLY on actual CONTACT: delta below VQ2_CONTACT_DELTA_M. Default 0.0
# disables collision-termination entirely (flip / out-of-bounds / timeout catch
# real crashes); set e.g. VQ2_CONTACT_DELTA_M=0.3 to re-enable contact-only kills.
HARD_THREAT = int(os.environ.get("VQ2_HARD_THREAT", "2"))
CONTACT_DELTA_M = float(os.environ.get("VQ2_CONTACT_DELTA_M", "0.0"))

# The estimator attitude/position is unreliable (esp. right after a teleport, and
# it drifts). Don't let a transient glitch end the episode: ignore flip/OOB for a
# grace window while it settles, and require the signal to persist a few steps.
RESET_GRACE_STEPS = int(os.environ.get("VQ2_RESET_GRACE", "15"))
TERM_DEBOUNCE = int(os.environ.get("VQ2_TERM_DEBOUNCE", "5"))


class VQ2RealEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_seconds: float = 30.0, num_gates: int = DEFAULT_NUM_GATES,
                 ip: str = "127.0.0.1", mav_port: int = 14550,
                 record_success: bool = False, use_gp_flight: bool = True):
        super().__init__()
        # Fly through the PROVEN auto_gp.py stack (GP pilot lifecycle: arm/GO/re-arm
        # + robust control loop) with the policy overriding the steering. Set
        # use_gp_flight=False (or VQ2_GP_FLIGHT=0) to use the bare SimInterface.
        self._gp_flight = use_gp_flight and os.environ.get("VQ2_GP_FLIGHT", "1") != "0"
        if self._gp_flight:
            from rl.vq2.gp_flight import GPFlightInterface
            self.sim = GPFlightInterface(ip=ip, mav_port=mav_port)
        else:
            self.sim = SimInterface(ip=ip, mav_port=mav_port)
        self.sim.data["_quiet_vision"] = True   # silence per-frame [vision] GATE spam
        self.est = GPEstimation(self.sim.data)
        self.est.start()
        self.gates = GatePassTracker()
        self.stacker = ObsStacker()
        self.max_steps = int(max_seconds * DECISION_HZ)
        self.num_gates = num_gates
        # When True, the instant the policy passes a NEW gate this episode, the
        # segment start->that-gate is saved permanently (incremental BC data).
        self.record_success = record_success

        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (POLICY_OBS_DIM,), np.float32)

        # Wait for the race GO (countdown elapsed) before handing control to the
        # policy, so every episode starts from the same settled post-countdown
        # state with the race live. VQ2_WAIT_GO=0 disables (start immediately) to
        # A/B whether it changes the learned routes. Falls back to immediate-go
        # after `go_timeout_s` (a teleport may not issue a fresh countdown).
        self._wait_go = os.environ.get("VQ2_WAIT_GO", "1") != "0"
        self._go_timeout_s = float(os.environ.get("VQ2_GO_TIMEOUT_S", "8.0"))

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
        # Hold until a FRESH race GO so every episode starts from the same settled,
        # race-live state. Two phases so we don't latch the PREVIOUS episode's
        # stale flying=True (which started episodes mid-teleport -> berserk flights):
        #   1) wait for the pilot to LEAVE flying (it detects the teleport clock
        #      reset and drops to WAIT/countdown),
        #   2) then wait for it to RE-ENTER flying (countdown elapsed = fresh GO).
        if self._wait_go and hasattr(self.sim, "flying"):
            t0 = time.time()
            while self.sim.flying() and time.time() - t0 < 5.0:
                time.sleep(0.05)      # phase 1: old flight must end first
            t0 = time.time()
            while not self.sim.flying() and time.time() - t0 < self._go_timeout_s:
                time.sleep(0.05)      # phase 2: wait for the new countdown's GO
            waited = time.time() - t0
            print(f"[vq2.env] fresh GO after {waited:.1f}s "
                  f"({'flying' if self.sim.flying() else 'timeout -> immediate-go'})",
                  flush=True)
        self.gates.reset(self.sim)
        self._steps = 0
        self._prev_action = np.zeros(spec.ACTION_DIM, np.float32)
        self._prev_dist = None
        self._last_collision = self.sim.data.get("last_collision")
        self._n_hard = self._n_soft = 0
        self._last_threat, self._last_delta = 0, -1.0
        self._ep_parts = {}
        self._ep_min_range = float("inf")
        # success-segment buffer (start -> current), saved on each new gate pass.
        self._seg_obs, self._seg_act = [], []
        self._seg_rew, self._seg_simus, self._seg_gidx = [], [], []
        self._ep_max_gate = 0
        self._ep_collisions = 0
        self._flip_streak = self._oob_streak = 0
        ego, pose_est, conf = self._read()
        obs = self.stacker.reset(self._frame(ego, pose_est, conf, self._prev_action))
        self._last_obs = obs                 # the obs the policy will condition on
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)

        # act: RESIDUAL RL -- the policy's action is a small correction ADDED to
        # the GP base command (stable-by-construction). Fallback path (bare
        # SimInterface) sends absolute attitude as before.
        if self._gp_flight:
            dr, dp, dy, dth = controller.action_to_residual(action)
            self.sim.send_residual_deg(dr, dp, dy, dth)
        else:
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
        # Only ACTUAL contact ends the episode; proximity warnings (delta ~3 m,
        # which you must pass through to thread a gate) are a soft nudge.
        contact = (col_evt and threat >= HARD_THREAT
                   and 0.0 <= delta < CONTACT_DELTA_M)
        collision = contact
        soft_collision = col_evt and not contact
        if col_evt:
            self._n_hard += int(collision)
            self._n_soft += int(soft_collision)
            self._last_threat, self._last_delta = threat, delta
        # estimator-based crash signals -- debounced + grace-windowed (see above).
        self._flip_streak = self._flip_streak + 1 if self._flipped(ego) else 0
        self._oob_streak = self._oob_streak + 1 if self._out_of_bounds(ego) else 0
        past_grace = self._steps > RESET_GRACE_STEPS
        flipped = past_grace and self._flip_streak >= TERM_DEBOUNCE
        oob = past_grace and self._oob_streak >= TERM_DEBOUNCE
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

        # ---- incremental success recording (training only) ----------------
        # Buffer (obs the policy saw, action it took). On passing a NEW gate,
        # save the whole start->gate segment IMMEDIATELY (not at episode end).
        if self.record_success:
            imu = self.sim.data.get("imu") or {}
            self._seg_obs.append(self._last_obs)
            self._seg_act.append(action)
            self._seg_rew.append(float(r))
            self._seg_simus.append(int(imu.get("time_us") or 0))
            self._seg_gidx.append(int(self.sim.data.get("active_gate_index", -1) or -1))
            self._ep_collisions += int(collision)
            if gate_passed and self.gates.n_passed > self._ep_max_gate:
                self._ep_max_gate = self.gates.n_passed
                try:
                    path = success.save_segment(
                        self.gates.n_passed, self._seg_obs, self._seg_act,
                        reward=self._seg_rew, sim_us=self._seg_simus,
                        gate_index=self._seg_gidx,
                        gates_reached=self.gates.n_passed, collisions=self._ep_collisions)
                    print(f"[vq2.env] SAVED success segment -> {path} "
                          f"({len(self._seg_act)} steps, {self._ep_collisions} collisions)",
                          flush=True)
                except Exception as exc:
                    print(f"[vq2.env] segment save failed: {exc}", flush=True)
        self._last_obs = obs

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
