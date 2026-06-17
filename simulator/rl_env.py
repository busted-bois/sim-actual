"""Gymnasium env wrapping the live FlightSim for residual-RL training (P1).

The policy never flies directly: each step it sets a guidance residual on the proven
pilot (`pilot.residual`), the pilot's 250 Hz inner loop executes, and after one RL tick
(~1/20 s real time) we read the resulting state for obs/reward.

Reverse curriculum (grill Q8): episodes begin near the hardest gate and expand backward.
Since `sim_reset` always starts at gate 0, we ferry there with the BARE pilot (identity
residual) until `active_gate_index` reaches the curriculum start gate, then hand control to
the policy. This relies on the pilot reliably traversing all gates (plan P0.5).

This module imports gymnasium and connects to the sim on construction, so it is NOT
imported by `make sim` and is not exercised by the pure-logic unit tests.
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.rl_core import (
    ACTION_DIM,
    OBS_DIM,
    RL_DT_S,
    action_to_residual,
    approach_dir,
    build_observation,
    dist_to_gate_center,
    finish_reward,
    frame_looks_corrupted,
    gate_aperture_radius,
    gate_plane_progress,
    identity_residual,
    lateral_miss_distance,
    reverse_curriculum_start_gate,
    step_reward,
    MISS_PENALTY,
    CRASH_PENALTY,
    TIMEOUT_PENALTY,
)


class ResidualRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        server_ip: str = "127.0.0.1",
        server_udp_port: int = 14550,
        max_episode_steps: int = 1500,
        ferry_timeout_s: float = 60.0,
        hardest_gate: int = 2,
    ):
        super().__init__()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.ferry_timeout_s = ferry_timeout_s
        self.hardest_gate = hardest_gate

        # Curriculum state (set by the trainer between rollouts).
        self._progress01 = 0.0
        self._time_weight = 0.0

        # --- bring up the sim connection + background control loop ---
        from simulator.setup import setup_components

        self._boot_ms = int(time.time() * 1000)
        self.shared = {}
        comps = setup_components(self.shared, self._boot_ms, server_ip, server_udp_port)
        self.controller = comps["controller"]
        self.pilot = self.controller.pilot
        self.pilot.external_residual = True  # env owns the residual; no self-policy
        self.pilot.residual = identity_residual()

        self._running = True
        self._ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._ctrl_thread.start()

        # episode trackers
        self._steps = 0
        self._ep_start_t = 0.0
        self._target_idx = 0
        self._prev_dist = 0.0
        self._prev_plane = -1.0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    def _control_loop(self):
        while self._running:
            self.controller.update()  # arms + pilot.tick + send + 1/250 sleep

    def set_curriculum(self, progress01: float, time_weight: float) -> None:
        self._progress01 = float(progress01)
        self._time_weight = float(time_weight)

    # ------------------------------------------------------------------
    def _gates(self):
        return self.shared.get("track_gates") or []

    def _drone_pos(self):
        od = self.shared.get("odometry")
        if od is None:
            return None
        return (od["x"], od["y"], od["z"])

    def _build_obs(self) -> np.ndarray:
        pos = self._drone_pos() or (0.0, 0.0, 0.0)
        vel = self.pilot._vel_est
        att = self.shared.get("attitude") or {}
        roll = float(att.get("roll", 0.0))
        pitch = float(att.get("pitch", 0.0))
        yaw = float(self.shared.get("yaw_rad", 0.0))
        idx = int(self.shared.get("active_gate_index", 0) or 0)
        return build_observation(
            pos,
            vel,
            roll,
            pitch,
            yaw,
            self._gates(),
            idx,
            self.pilot.residual,
            self._last_action,
        )

    def _wait_for_race(self, timeout_s: float = 30.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if (
                self.shared.get("armed")
                and self._drone_pos() is not None
                and self.shared.get("active_gate_index") is not None
                and self._gates()
            ):
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        gates = None
        for _attempt in range(3):
            self.pilot.external_residual = True
            self.pilot.residual = identity_residual()
            self.shared["collision"] = None
            self.controller.send_sim_reset_command()
            time.sleep(1.0)
            if not self._wait_for_race():
                continue
            # P0 contingency: a prior crash can corrupt the sim frame so the soft reset
            # leaves the drone km from the pad. Training on that is garbage — detect it and
            # halt loudly rather than silently learning from a broken frame. (Auto-relaunch
            # isn't possible: FlightSim.exe needs an interactive login.)
            if frame_looks_corrupted(self._drone_pos()):
                continue  # retry the soft reset; if it never clears, we raise below
            gates = self._gates()
            start_gate = reverse_curriculum_start_gate(
                self._progress01, len(gates), self.hardest_gate
            )
            if self._ferry_to(start_gate):
                break
        else:
            # All attempts failed — most likely a corrupted frame that soft reset can't fix.
            if frame_looks_corrupted(self._drone_pos()):
                raise RuntimeError(
                    "Sim frame corrupted (drone km from pad after reset) — soft reset can't "
                    "recover. Fully restart FlightSim.exe, log in, start a race, then resume "
                    "training. Consider the surrogate-sim route to avoid live-sim corruption."
                )
        if gates is None:
            gates = self._gates()

        self._steps = 0
        self._ep_start_t = time.monotonic()
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._target_idx = int(self.shared.get("active_gate_index", 0) or 0)
        pos = self._drone_pos() or (0.0, 0.0, 0.0)
        if gates and self._target_idx < len(gates):
            self._prev_dist = dist_to_gate_center(pos, gates[self._target_idx])
            self._prev_plane = gate_plane_progress(
                pos, gates[self._target_idx], approach_dir(gates, self._target_idx, pos)
            )
        return self._build_obs(), {}

    def _ferry_to(self, start_gate: int) -> bool:
        """Fly the bare pilot until active_gate_index reaches start_gate (or it's 0)."""
        if start_gate <= 0:
            return True
        self.pilot.residual = identity_residual()
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.ferry_timeout_s:
            if self.shared.get("collision") is not None:
                return False  # crashed while ferrying -> retry reset
            if int(self.shared.get("active_gate_index", 0) or 0) >= start_gate:
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.pilot.last_action = action
        self.pilot.residual = action_to_residual(action)
        self._last_action = action

        time.sleep(RL_DT_S)  # let the inner loop fly this residual for one RL tick
        self._steps += 1

        gates = self._gates()
        pos = self._drone_pos() or (0.0, 0.0, 0.0)
        idx = int(self.shared.get("active_gate_index", 0) or 0)
        n_gates = len(gates)

        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        # --- finish ---
        if n_gates > 0 and idx >= n_gates:
            ep_time = time.monotonic() - self._ep_start_t
            reward += finish_reward(ep_time, self._time_weight)
            terminated = True
            info["finished"] = True
            return self._build_obs(), float(reward), terminated, truncated, info

        # --- crash ---
        if self.shared.get("collision") is not None:
            reward += CRASH_PENALTY
            terminated = True
            info["crashed"] = True
            return self._build_obs(), float(reward), terminated, truncated, info

        target = gates[idx] if (gates and idx < n_gates) else None

        # gate passed cleanly this step?
        gate_passed = idx > self._target_idx
        if gate_passed:
            self._target_idx = idx
            if target is not None:
                self._prev_dist = dist_to_gate_center(pos, target)
                self._prev_plane = gate_plane_progress(
                    pos, target, approach_dir(gates, idx, pos)
                )

        # --- progress reward toward current target ---
        if target is not None:
            cur_dist = dist_to_gate_center(pos, target)
            reward += step_reward(self._prev_dist, cur_dist, gate_passed, action)
            self._prev_dist = cur_dist

            # --- miss: crossed the gate plane outside the aperture w/o a pass ---
            appr = approach_dir(gates, idx, pos)
            plane = gate_plane_progress(pos, target, appr)
            if self._prev_plane < 0.0 and plane > 0.3 and not gate_passed:
                if lateral_miss_distance(pos, target, appr) > gate_aperture_radius(
                    target
                ):
                    reward += MISS_PENALTY
                    terminated = True
                    info["missed"] = True
            self._prev_plane = plane

        # --- timeout ---
        if not terminated and self._steps >= self.max_episode_steps:
            reward += TIMEOUT_PENALTY
            truncated = True

        return self._build_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        self._running = False
        try:
            self._ctrl_thread.join(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass
