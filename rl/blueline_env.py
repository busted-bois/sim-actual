"""Blue-line corridor Gymnasium env for PPO (no YOLO).

Observation = HSV dual-cyan corridor features + IMU vel/rates.
Action      = attitude-quat wireform [roll_deg, pitch_deg, yaw_deg, thrust]
              matching BlueLinePilot / set_attitude_quat_deg.

Default plant is a lightweight internal physics + synthetic blue-line
projection (fast offline training). Swap the PLACEHOLDER hooks below to
drive the live sim (`make blueline` stack).

    uv run -m rl.blueline_env --selftest
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import spec
from simulator.blue_line_pilot import (
    CY_TARGET,
    MAX_BANK_DEG as ROLL_CLIP,
    PITCH_WIRE_MAX_DEG as PITCH_CLIP,
    THRUST_MAX,
    THRUST_MIN,
    YAW_ERR_MAX_DEG as YAW_CLIP,
)
from simulator.gp_pilot import HOVER_THRUST

# ---- obs / action layout ----------------------------------------------------
# [found, cx, cy, heading_err, width, left_found, right_found, vx,vy,vz, wx,wy,wz]
OBS_DIM = 13
ACT_DIM = 4
# Policy outputs [-1,1]^4; scale_action maps to BlueLinePilot attitude-quat cmds.

DECISION_HZ = 50
SIM_DT = 1 / 100.0
SUBSTEPS = max(1, int((1 / DECISION_HZ) / SIM_DT))
MASS_THRUST_ACCEL = spec.GRAVITY / max(HOVER_THRUST, 1e-3)
ATT_TAU = 0.08  # s, first-order attitude tracking
RATE_TAU = 0.05
DRAG = 0.12
G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])

# Curriculum: scale 3 -> 15 gates.
CURRICULUM = [
    {"num_gates": 3, "spawn_dist": 5.0, "jitter": 0.4},
    {"num_gates": 6, "spawn_dist": 6.0, "jitter": 0.8},
    {"num_gates": 10, "spawn_dist": 6.5, "jitter": 1.2},
    {"num_gates": 15, "spawn_dist": 7.0, "jitter": 1.5},
]

GATE_PASS_REWARD = 50.0
CRASH_PENALTY = -100.0
CY_TARGET_OBS = float(CY_TARGET)
WIDTH_LO, WIDTH_HI = 0.15, 0.85


def scale_action(a: np.ndarray) -> np.ndarray:
    """Map [-1,1]^4 -> [roll_deg, pitch_deg, yaw_deg, thrust] (BlueLinePilot clips)."""
    a = np.clip(np.asarray(a, np.float64), -1.0, 1.0)
    roll = float(a[0] * ROLL_CLIP)
    pitch = float(a[1] * PITCH_CLIP)
    yaw = float(a[2] * YAW_CLIP)
    thrust = float(THRUST_MIN + 0.5 * (a[3] + 1.0) * (THRUST_MAX - THRUST_MIN))
    return np.array([roll, pitch, yaw, thrust], dtype=np.float64)


# =============================================================================
# PLACEHOLDER — plug live sim API here
# =============================================================================
class SimAPI:
    """Boundary to the real simulator. Default = None (use InternalPlant).

    Wire these to your stack, e.g.:
      send_attitude_quat_deg -> controller.set_attitude_quat_deg
      read_blue_line         -> shared_data["blue_line"] / detect_blue_lines
      read_imu               -> odometry / imu rates + linear vel
      read_active_gate       -> shared_data["active_gate_index"]
      check_crash            -> ground / gate collision / race fail
      reset_episode          -> disarm/reset/arm/start race
      advance                -> controller.update() + sleep(1/hz)
    """

    def send_attitude_quat_deg(
        self, roll: float, pitch: float, yaw: float, thrust: float
    ) -> None:
        raise NotImplementedError

    def read_blue_line(self) -> dict:
        """Return blue_line dict (found, cx_norm, cy_norm, heading_err, ...)."""
        raise NotImplementedError

    def read_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lin_vel_ned[3], ang_vel_rpy[3]) in m/s and rad/s."""
        raise NotImplementedError

    def read_active_gate(self) -> int:
        raise NotImplementedError

    def check_crash(self) -> str | None:
        """None if ok, else reason string ('ground'|'gate'|'timeout'|...)."""
        raise NotImplementedError

    def reset_episode(self) -> None:
        raise NotImplementedError

    def advance(self, dt: float) -> None:
        raise NotImplementedError


# =============================================================================
# Internal offline plant (synthetic corridor vision)
# =============================================================================
def _quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def _rpy_to_quat(roll, pitch, yaw):
    """ZYX yaw-pitch-roll (rad) -> (w,x,y,z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _quat_to_rpy(q):
    w, x, y, z = q
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


class InternalPlant:
    """Lightweight quadrotor + synthetic blue-line from gate corridor."""

    def __init__(self, num_gates: int, spawn_dist: float, jitter: float, rng):
        self.num_gates = num_gates
        self.spawn_dist = spawn_dist
        self.jitter = jitter
        self.rng = rng
        self.reset()

    def reset(self):
        gates, x = [], 0.0
        for i in range(self.num_gates):
            x += self.spawn_dist if i == 0 else self.rng.uniform(5.0, 7.0)
            y = self.rng.uniform(-self.jitter, self.jitter)
            z = self.rng.uniform(-3.0 - self.jitter * 0.3, -3.0 + self.jitter * 0.3)
            yaw = 0.0 if i == 0 else self.rng.uniform(-0.35, 0.35)
            gates.append(
                {
                    "pos": [x, y, z],
                    "quat": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    "w": spec.GATE_SIZE_M,
                    "h": spec.GATE_SIZE_M,
                }
            )
        self.gate_map = gates
        g0 = np.array(gates[0]["pos"])
        self.p = np.array(
            [
                0.0,
                self.rng.uniform(-0.4, 0.4),
                g0[2] + self.rng.uniform(-0.4, 0.4),
            ]
        )
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.gate_idx = 0
        self._cmd = np.array([0.0, 0.0, 0.0, HOVER_THRUST])
        self._prev_signed = self._signed_dist(0)
        self.crash: str | None = None
        self.gate_just_passed = False

    def send_attitude_quat_deg(self, roll, pitch, yaw, thrust):
        self._cmd = np.array(
            [
                float(np.clip(roll, -ROLL_CLIP, ROLL_CLIP)),
                float(np.clip(pitch, -PITCH_CLIP, PITCH_CLIP)),
                float(np.clip(yaw, -YAW_CLIP, YAW_CLIP)),
                float(np.clip(thrust, THRUST_MIN, THRUST_MAX)),
            ]
        )

    def _gate_frame(self, idx):
        g = self.gate_map[idx]
        gc = np.array(g["pos"])
        n = spec.quat_to_R(np.array(g["quat"])) @ np.array([1.0, 0.0, 0.0])
        return gc, n

    def _signed_dist(self, idx):
        gc, n = self._gate_frame(idx)
        return float(n @ (self.p - gc))

    def advance(self, dt: float):
        self.gate_just_passed = False
        if self.crash:
            return
        n_sub = max(1, int(round(dt / SIM_DT)))
        roll_d, pitch_d, yaw_d, thrust = self._cmd
        # Relative yaw cmd (deg) about current heading — matches pilot wireform.
        cur_r, cur_p, cur_y = _quat_to_rpy(self.q)
        tgt_q = _rpy_to_quat(
            np.radians(roll_d),
            np.radians(pitch_d),
            cur_y + np.radians(yaw_d),
        )
        for _ in range(n_sub):
            # Attitude error -> body rate command (small-angle quat).
            q_err = _quat_mult(
                _quat_norm(np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]])),
                tgt_q,
            )
            if q_err[0] < 0:
                q_err = -q_err
            omega_cmd = (2.0 / ATT_TAU) * q_err[1:4]
            omega_cmd = np.clip(omega_cmd, -2.0, 2.0)
            self.omega += (omega_cmd - self.omega) * (SIM_DT / RATE_TAU)
            self.q = _quat_norm(
                _quat_mult(self.q, np.array([1.0, *(0.5 * self.omega * SIM_DT)]))
            )
            R = spec.quat_to_R(self.q)
            f_body = np.array([0.0, 0.0, -MASS_THRUST_ACCEL * thrust])
            a = R @ f_body + G_WORLD - DRAG * self.v
            self.v += a * SIM_DT
            self.p += self.v * SIM_DT

        self._update_gate_and_crash()

    def _update_gate_and_crash(self):
        if self.gate_idx >= len(self.gate_map):
            return
        gc, n = self._gate_frame(self.gate_idx)
        signed = float(n @ (self.p - gc))
        dist = float(np.linalg.norm(gc - self.p))

        if self._prev_signed < 0.0 <= signed:
            g = self.gate_map[self.gate_idx]
            R = spec.quat_to_R(np.array(g["quat"]))
            rel = self.p - gc
            dy = abs(float(R[:, 1] @ rel))
            dz = abs(float(R[:, 2] @ rel))
            hw = 0.5 * float(g["w"])
            hh = 0.5 * float(g["h"])
            if dy < hw and dz < hh:
                self.gate_idx += 1
                self.gate_just_passed = True
                if self.gate_idx < len(self.gate_map):
                    self._prev_signed = self._signed_dist(self.gate_idx)
                return
            self.crash = "gate"
            return

        if self.p[2] > 0.5:
            self.crash = "ground"
        elif dist > 30.0 or abs(self.p[1]) > 35.0:
            self.crash = "out_of_bounds"
        else:
            gb_z = (spec.quat_to_R(self.q).T @ np.array([0.0, 0.0, 1.0]))[2]
            if gb_z < 0.0:
                self.crash = "flipped"

        self._prev_signed = signed

    def synthetic_blue_line(self) -> dict:
        """Project corridor (gate centers) into image-like normalized errors."""
        if self.gate_idx >= len(self.gate_map):
            return {
                "found": False,
                "cx_norm": 0.0,
                "cy_norm": 0.0,
                "heading_err": 0.0,
                "width_norm": 0.0,
                "left_found": False,
                "right_found": False,
            }

        # Near point: current gate; far: next gate (or same + normal).
        near = np.array(self.gate_map[self.gate_idx]["pos"])
        if self.gate_idx + 1 < len(self.gate_map):
            far = np.array(self.gate_map[self.gate_idx + 1]["pos"])
        else:
            _, n = self._gate_frame(self.gate_idx)
            far = near + 6.0 * n

        # Half-width ribbons in gate-local Y.
        g = self.gate_map[self.gate_idx]
        R_g = spec.quat_to_R(np.array(g["quat"]))
        right = R_g[:, 1]
        half = 0.5 * float(g["w"]) * 0.55
        left_pt = near - half * right
        right_pt = near + half * right

        pts = np.stack([near, far, left_pt, right_pt], axis=0)
        pix, front = spec.project(pts, self.p, self.q)
        if not front[0]:
            return {
                "found": False,
                "cx_norm": 0.0,
                "cy_norm": 0.0,
                "heading_err": 0.0,
                "width_norm": 0.0,
                "left_found": False,
                "right_found": False,
            }

        def _norm_x(u):
            return float(np.clip((u - spec.CX) / (spec.IMG_W * 0.5), -1.5, 1.5))

        def _norm_y(v):
            return float(np.clip((v - spec.CY) / (spec.IMG_H * 0.5), -1.5, 1.5))

        cx = _norm_x(pix[0, 0])
        cy = _norm_y(pix[0, 1])
        heading = 0.0
        if front[1]:
            heading = float(
                np.arctan2(pix[1, 0] - pix[0, 0], max(spec.IMG_H * 0.3, 1.0))
            )
        left_ok = bool(front[2] and 0 <= pix[2, 0] < spec.IMG_W)
        right_ok = bool(front[3] and 0 <= pix[3, 0] < spec.IMG_W)
        width = 0.0
        if left_ok and right_ok:
            width = float(abs(pix[3, 0] - pix[2, 0]) / spec.IMG_W)
        found = left_ok or right_ok
        # Drop if corridor mid is way outside FOV.
        if abs(cx) > 1.35 or abs(cy) > 1.35:
            found = False
        return {
            "found": found,
            "cx_norm": cx if found else 0.0,
            "cy_norm": cy if found else 0.0,
            "heading_err": heading if found else 0.0,
            "width_norm": width if found else 0.0,
            "left_found": left_ok and found,
            "right_found": right_ok and found,
        }


# =============================================================================
# Gymnasium env
# =============================================================================
class DroneBlueLineEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: int = 0,
        max_seconds: float = 45.0,
        seed: int | None = None,
        sim_api: SimAPI | None = None,
    ):
        super().__init__()
        self.stage = int(np.clip(stage, 0, len(CURRICULUM) - 1))
        self.cfg = CURRICULUM[self.stage]
        self.max_steps = int(max_seconds * DECISION_HZ)
        self.sim_api = sim_api
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.plant: InternalPlant | None = None
        self.steps = 0
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_cx = 0.0
        self._prev_cy_err = 0.0
        self._prev_hdg = 0.0
        self._last_gate = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.steps = 0
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_cx = self._prev_cy_err = self._prev_hdg = 0.0
        if self.sim_api is not None:
            self.sim_api.reset_episode()
            self._last_gate = int(self.sim_api.read_active_gate())
        else:
            self.plant = InternalPlant(
                self.cfg["num_gates"],
                self.cfg["spawn_dist"],
                self.cfg["jitter"],
                self.np_random,
            )
            self._last_gate = 0
        bl = self._blue_line()
        self._prev_cx = abs(float(bl.get("cx_norm", 0.0)))
        self._prev_cy_err = abs(float(bl.get("cy_norm", 0.0)) - CY_TARGET_OBS)
        self._prev_hdg = abs(float(bl.get("heading_err", 0.0)))
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        roll, pitch, yaw, thrust = scale_action(action)

        if self.sim_api is not None:
            self.sim_api.send_attitude_quat_deg(roll, pitch, yaw, thrust)
            self.sim_api.advance(1.0 / DECISION_HZ)
        else:
            assert self.plant is not None
            self.plant.send_attitude_quat_deg(roll, pitch, yaw, thrust)
            self.plant.advance(1.0 / DECISION_HZ)

        jerk = float(np.linalg.norm(action - self.last_action))
        self.last_action = action.copy()
        self.steps += 1

        rew, terminated, info = self._reward(jerk)
        truncated = self.steps >= self.max_steps
        if truncated and not terminated:
            rew += CRASH_PENALTY
            info["crash"] = "timeout"
            terminated = True
        return self._obs(), float(rew), bool(terminated), bool(truncated), info

    # ---- sensors ------------------------------------------------------------
    def _blue_line(self) -> dict:
        if self.sim_api is not None:
            return self.sim_api.read_blue_line()
        assert self.plant is not None
        return self.plant.synthetic_blue_line()

    def _imu(self) -> tuple[np.ndarray, np.ndarray]:
        if self.sim_api is not None:
            return self.sim_api.read_imu()
        assert self.plant is not None
        return self.plant.v.copy(), self.plant.omega.copy()

    def _active_gate(self) -> int:
        if self.sim_api is not None:
            return int(self.sim_api.read_active_gate())
        assert self.plant is not None
        return int(self.plant.gate_idx)

    def _crash(self) -> str | None:
        if self.sim_api is not None:
            return self.sim_api.check_crash()
        assert self.plant is not None
        return self.plant.crash

    def _obs(self) -> np.ndarray:
        bl = self._blue_line()
        found = bool(bl.get("found", False))
        if found:
            cx = float(bl.get("cx_norm", 0.0))
            cy = float(bl.get("cy_norm", 0.0))
            hdg = float(bl.get("heading_err", 0.0))
            width = float(bl.get("width_norm", 0.0))
            lf = 1.0 if bl.get("left_found") else 0.0
            rf = 1.0 if bl.get("right_found") else 0.0
            f = 1.0
        else:
            cx = cy = hdg = width = lf = rf = f = 0.0
        vel, ang = self._imu()
        return np.array(
            [f, cx, cy, hdg, width, lf, rf, *vel, *ang],
            dtype=np.float32,
        )

    def _reward(self, jerk: float):
        info: dict = {}
        bl = self._blue_line()
        found = bool(bl.get("found", False))
        cx = abs(float(bl.get("cx_norm", 0.0))) if found else 1.0
        cy_err = abs(float(bl.get("cy_norm", 0.0)) - CY_TARGET_OBS) if found else 1.0
        hdg = abs(float(bl.get("heading_err", 0.0))) if found else 1.0
        width = float(bl.get("width_norm", 0.0)) if found else 0.0
        left = bool(bl.get("left_found", False))
        right = bool(bl.get("right_found", False))

        rew = 0.0
        # Dense: reward decreasing corridor errors.
        rew += 2.0 * (self._prev_cx - cx)
        rew += 1.5 * (self._prev_cy_err - cy_err)
        rew += 1.0 * (self._prev_hdg - hdg)
        self._prev_cx, self._prev_cy_err, self._prev_hdg = cx, cy_err, hdg

        if found and left and right and WIDTH_LO <= width <= WIDTH_HI:
            rew += 0.05

        # Visibility / tracking.
        if not found:
            rew -= 0.5
        elif left != right:
            rew -= 0.2

        # Smoothness.
        rew -= 0.02 * jerk
        rew -= 0.01  # time

        # Gate progress (sim active_gate_index or internal plant).
        gate = self._active_gate()
        n_gates = self.cfg["num_gates"] if self.plant is not None else 15
        if self.plant is not None and self.plant.gate_just_passed:
            rew += GATE_PASS_REWARD
            info["gate_passed"] = True
            self._last_gate = gate
            if gate >= n_gates:
                info["course_complete"] = True
                rew += 30.0
                return rew, True, info
        elif gate > self._last_gate:
            rew += GATE_PASS_REWARD * (gate - self._last_gate)
            info["gate_passed"] = True
            self._last_gate = gate
            if gate >= n_gates:
                info["course_complete"] = True
                rew += 30.0
                return rew, True, info

        crash = self._crash()
        if crash:
            rew += CRASH_PENALTY
            info["crash"] = crash
            return rew, True, info

        return rew, False, info


def make_env(stage: int = 0, seed: int | None = None, sim_api: SimAPI | None = None):
    def _thunk():
        return DroneBlueLineEnv(stage=stage, seed=seed, sim_api=sim_api)

    return _thunk


def _selftest():
    from stable_baselines3.common.env_checker import check_env

    env = DroneBlueLineEnv(stage=0, seed=0)
    check_env(env, warn=True)
    print(f"[selftest] check_env ok (obs {OBS_DIM}, act {ACT_DIM})")

    obs, _ = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    total, steps = 0.0, 0
    term = trunc = False
    info = {}
    # Mild cruise in normalized action space (~ -3° pitch, hover thrust).
    hover_n = (2.0 * (HOVER_THRUST - THRUST_MIN) / (THRUST_MAX - THRUST_MIN)) - 1.0
    cruise = np.array([0.0, -3.0 / PITCH_CLIP, 0.0, hover_n], np.float32)
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(cruise)
        total += r
        steps += 1
    print(f"[selftest] cruise rollout {steps} steps rew={total:.1f} info={info}")

    # Random action must not crash the env API.
    env = DroneBlueLineEnv(stage=0, seed=2)
    obs, _ = env.reset()
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    print("[selftest] OK — blue-line env + placeholders wired")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
