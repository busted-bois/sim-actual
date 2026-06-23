"""Module 7 — Gym training environment + curriculum.

A Gymnasium env wrapping a LIGHTWEIGHT INTERNAL quadrotor physics model. It
does NOT connect to the Anduril simulator — stepping a real sim per RL step
would be far too slow. The live sim is used only in Modules 1-2 (data) and
final evaluation. Domain randomization on the internal model covers the
sim-to-sim gap.

  * Action      : 4-D normalized [-1,1] -> (v_des NED, yaw_rate_des)  [spec.scale_velocity_action]
                  -> geo_control.velocity_to_action -> rates + thrust each substep
  * Observation : 24-D gate-relative vector                              [Module 6]
  * Reward      : dense progress + gate-pass bonus - crash/time/effort
  * Curriculum  : stage 0 single close gate -> 1 two gates -> 2 full 6-gate course

    uv run -m rl.env --selftest
"""

from __future__ import annotations

import argparse
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import spec
from rl.geo_control import GeoGains, velocity_to_action
from rl.observation import build_observation

# Reduced quadrotor parameters.
MASS = 1.0
THRUST_ACCEL = spec.GRAVITY / spec.HOVER_THRUST  # thrust=0.5 -> hover
RATE_TAU = 0.05  # body-rate first-order lag (s)
DRAG = 0.1
SIM_DT = 1 / 100.0
DECISION_HZ = 50
SUBSTEPS = int((1 / DECISION_HZ) / SIM_DT)

G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])

# Inner-loop gains matched to the internal training plant (not live-sim 0.27).
TRAIN_GAINS = GeoGains(
    hover_thrust=spec.HOVER_THRUST,
    thrust_min=0.0,
    thrust_max=1.0,
)

# Curriculum stages: (num_gates, first-gate distance, layout jitter).
CURRICULUM = [
    {"num_gates": 1, "spawn_dist": 5.0, "jitter": 0.5},
    {"num_gates": 2, "spawn_dist": 6.0, "jitter": 1.0},
    {"num_gates": 6, "spawn_dist": 7.0, "jitter": 2.0},
]


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
    return q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])


def _quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def velocity_expert_action(p, v, q, target, max_speed=6.0):
    """Simple outer-loop expert: point v_des at gate + yaw toward it."""
    to_t = np.asarray(target, float) - np.asarray(p, float)
    dist = float(np.linalg.norm(to_t))
    speed = min(max_speed, max(1.5, 0.45 * dist))
    v_des = (to_t / dist) * speed if dist > 1e-6 else np.zeros(3)
    _, _, yaw = _quat_to_rpy(np.asarray(q, float))
    bearing = math.atan2(to_t[1], to_t[0])
    yaw_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate_des = 0.8 * yaw_err
    return spec.encode_velocity_action(v_des, yaw_rate_des)


class GateRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: int = 0,
        gate_map: list | None = None,
        max_seconds: float = 20.0,
        seed: int | None = None,
        geo_gains: GeoGains = TRAIN_GAINS,
    ):
        super().__init__()
        self.stage = int(np.clip(stage, 0, len(CURRICULUM) - 1))
        self.cfg = CURRICULUM[self.stage]
        self.user_gate_map = gate_map
        self.max_steps = int(max_seconds * DECISION_HZ)
        self.geo_gains = geo_gains
        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (spec.OBS_DIM,), np.float32)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

    # ---- layout ---------------------------------------------------------
    def _make_gate_map(self):
        if self.user_gate_map is not None:
            return [dict(g) for g in self.user_gate_map[: self.cfg["num_gates"]]]
        n = self.cfg["num_gates"]
        d0, j = self.cfg["spawn_dist"], self.cfg["jitter"]
        gates, x = [], 0.0
        for i in range(n):
            x += d0 if i == 0 else self.np_random.uniform(5.0, 7.0)
            y = self.np_random.uniform(-j, j)
            z = self.np_random.uniform(-3.0 - j, -3.0 + j)  # NED altitude
            yaw = self.np_random.uniform(-0.4, 0.4) if i > 0 else 0.0
            gates.append(
                {
                    "pos": [x, y, z],
                    "quat": [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    "w": spec.GATE_SIZE_M,
                    "h": spec.GATE_SIZE_M,
                }
            )
        return gates

    def _reset_state(self):
        self.gate_map = self._make_gate_map()
        g0 = np.array(self.gate_map[0]["pos"])
        # Spawn behind gate 0, facing roughly toward it, near gate altitude.
        self.p = np.array(
            [
                0.0,
                self.np_random.uniform(-0.5, 0.5),
                g0[2] + self.np_random.uniform(-0.5, 0.5),
            ]
        )
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.gate_idx = 0
        self.steps = 0
        self.last_action = np.zeros(4)
        self._prev_dist = float(np.linalg.norm(g0 - self.p))
        self._prev_signed = self._signed_dist(self.gate_idx)

    # ---- gate geometry --------------------------------------------------
    def _gate_frame(self, idx):
        g = self.gate_map[idx]
        gc = np.array(g["pos"])
        n = spec.quat_to_R(np.array(g["quat"])) @ np.array([1.0, 0, 0])
        return gc, n

    def _signed_dist(self, idx):
        gc, n = self._gate_frame(idx)
        return float(n @ (self.p - gc))

    def _obs(self):
        return build_observation(
            self.p,
            self.v,
            self.q,
            self.omega,
            self.gate_map,
            self.gate_idx,
            self.last_action[:3],
        )

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        v_des, yaw_rate_des = spec.scale_velocity_action(action)

        for _ in range(SUBSTEPS):
            roll, pitch, yaw = _quat_to_rpy(self.q)
            rates_thrust = velocity_to_action(
                v_des,
                yaw_rate_des,
                self.v,
                roll,
                pitch,
                yaw,
                self.geo_gains,
            )
            omega_cmd = rates_thrust[:3]
            thrust = float(rates_thrust[3])

            self.omega += (omega_cmd - self.omega) * (SIM_DT / RATE_TAU)
            self.q = _quat_norm(
                _quat_mult(self.q, np.array([1.0, *(0.5 * self.omega * SIM_DT)]))
            )
            R = spec.quat_to_R(self.q)
            f_body = np.array([0.0, 0.0, -THRUST_ACCEL * thrust])  # thrust up (body -z)
            a = R @ f_body + G_WORLD - DRAG * self.v
            self.v += a * SIM_DT
            self.p += self.v * SIM_DT

        self.steps += 1
        self.last_action = action
        rew, terminated, info = self._reward_done()
        truncated = self.steps >= self.max_steps
        return self._obs(), float(rew), bool(terminated), bool(truncated), info

    # ---- reward / termination ------------------------------------------
    def _reward_done(self):
        info = {}
        gc, n = self._gate_frame(self.gate_idx)
        dist = float(np.linalg.norm(gc - self.p))
        signed = float(n @ (self.p - gc))

        rew = 0.0
        rew += 2.0 * (self._prev_dist - dist)  # dense progress
        rew -= 0.01  # time penalty
        rew -= 0.001 * float(self.last_action @ self.last_action)  # effort

        # Velocity-toward-gate shaping.
        to_gate = gc - self.p
        ng = np.linalg.norm(to_gate)
        if ng > 1e-6:
            rew += 0.05 * float(self.v @ (to_gate / ng))

        terminated = False
        # Gate plane crossing: signed dist flips - to + while within opening.
        if self._prev_signed < 0.0 <= signed:
            # Square opening: offsets along the gate's local right/down axes
            # vs its reported w/h.
            g = self.gate_map[self.gate_idx]
            R = spec.quat_to_R(np.array(g["quat"]))
            rel = self.p - gc
            dy = abs(float(R[:, 1] @ rel))  # width axis (right)
            dz = abs(float(R[:, 2] @ rel))  # height axis (down)
            hw = 0.5 * float(g.get("w", spec.GATE_SIZE_M))
            hh = 0.5 * float(g.get("h", spec.GATE_SIZE_M))
            if dy < hw and dz < hh:  # passed through opening
                margin = min(1.0 - dy / hw, 1.0 - dz / hh)  # 0 at edge, 1 centered
                rew += 10.0 + 3.0 * margin
                self.gate_idx += 1
                info["gate_passed"] = True
                if self.gate_idx >= len(self.gate_map):
                    rew += 30.0  # course complete
                    info["course_complete"] = True
                    terminated = True
                else:
                    ngc = np.array(self.gate_map[self.gate_idx]["pos"])
                    self._prev_dist = float(np.linalg.norm(ngc - self.p))
                    self._prev_signed = self._signed_dist(self.gate_idx)
                    return rew, terminated, info
            else:
                rew -= 5.0  # crashed into gate frame / missed opening

        # Crash conditions.
        if self.p[2] > 0.5:  # below ground (NED z down)
            rew -= 20.0
            terminated = True
            info["crash"] = "ground"
        elif dist > 25.0 or abs(self.p[1]) > 30.0:  # flew away
            rew -= 20.0
            terminated = True
            info["crash"] = "out_of_bounds"
        # Excessive tilt (upside down): gravity in body z goes negative.
        gb_z = (spec.quat_to_R(self.q).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            rew -= 20.0
            terminated = True
            info["crash"] = "flipped"

        self._prev_dist = dist
        self._prev_signed = signed
        return rew, terminated, info


def make_env(stage=0, gate_map=None, seed=None):
    def _thunk():
        return GateRacingEnv(stage=stage, gate_map=gate_map, seed=seed)

    return _thunk


# ---------------------------------------------------------------------------
def _selftest():
    from stable_baselines3.common.env_checker import check_env

    env = GateRacingEnv(stage=0, seed=0)
    check_env(env, warn=True)
    print("[selftest] SB3 check_env passed (obs 24, act 4 hybrid)")

    # Random rollout — env must run and terminate sanely.
    obs, _ = env.reset(seed=1)
    assert obs.shape == (24,)
    total, steps = 0.0, 0
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        steps += 1
    print(f"[selftest] random rollout {steps} steps reward={total:.1f} info={info}")

    # Velocity-setpoint expert sanity (stage 0 must mostly clear; later stages logged).
    for stage in range(len(CURRICULUM)):
        passes = total_gates = 0
        trials = 6 if stage == 0 else 3
        for trial in range(trials):
            e = GateRacingEnv(stage=stage, seed=100 + trial, max_seconds=40.0)
            o, _ = e.reset()
            total_gates += len(e.gate_map)
            term = trunc = False
            while not (term or trunc):
                tgt = np.array(e.gate_map[e.gate_idx]["pos"])
                o, r, term, trunc, info = e.step(
                    velocity_expert_action(e.p, e.v, e.q, tgt)
                )
                if info.get("gate_passed"):
                    passes += 1
        print(
            f"[selftest] stage {stage}: expert passed {passes}/{total_gates} "
            f"gate-crossings over {trials} episodes",
            flush=True,
        )
        if stage == 0:
            assert passes >= max(1, trials - 1), (
                f"stage-0 expert broken ({passes}/{total_gates})"
            )
    print("[selftest] OK — hybrid env steps, rewards, gate-pass + curriculum wired")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
