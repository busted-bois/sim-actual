"""VQ2 racing environment — fast internal physics, camera-rendered observation.

Two constraints shape this env, both measured rather than assumed.

1. Bulk RL needs ~1e7 steps. The live simulator is hard-capped at 50 steps/s by
   a sleep in its step loop, with resets costing 1-21 s against episodes of
   ~0.5 s; the previous attempt bought 423k steps over two days and peaked at
   2 gates. Measured here: this env runs ~2.6k steps/s end-to-end through SB3
   PPO, so 1e7 is about an hour. Bulk learning must happen here.

2. The observation must be one the qualifier can actually deliver. The older
   rl/environment/env.py hands over world position, world velocity and an
   absolute gate map — all three are withheld under the VQ2 block (probe,
   2026-08-01: ODOMETRY / ATTITUDE / LOCAL_POSITION_NED absent even when
   explicitly requested; track_info=0). So this env keeps the privileged state
   internally for physics and reward, and RENDERS what the policy sees through
   rl.core.spec's real pinhole model and 20 deg camera up-tilt, then degrades it
   the way the real detector degrades: fixed ~15 Hz update, dropout, PnP noise,
   and keypoints that vanish as the gate leaves frame.

Reward keeps the official objective sparse — completion pays COMPLETE_BONUS —
with an annealable shaping scaffold on top. Shaping is keyed to the gate INDEX,
never to range-to-whatever-the-detector-likes; that mistake is what produced
the previous attempt's mean per-episode progress of -42 while it flew forward
at +26.

    uv run python -m unittest tests.test_vq2_env
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.core import spec
from rl.core import vq2_observation as vo

# --- geometry ---------------------------------------------------------------
# The flying code (simulator/gate_pnp.py) models a 1.5 m opening inside a 2.7 m
# outer frame, corroborated by measurement (inner/outer width ratio 0.560 over
# 551 detected gates vs the 0.556 implied). spec.GATE_SIZE_M = 2.72 is applied
# elsewhere as the opening, which is wrong by 1.81x — do not use it here.
GATE_OPENING_M = 1.5
GATE_OUTER_M = 2.7
HALF_OPENING = GATE_OPENING_M / 2.0
HALF_OUTER = GATE_OUTER_M / 2.0

# --- control / physics ------------------------------------------------------
DECISION_HZ = 50
SIM_DT = 1 / 200.0
SUBSTEPS = int((1 / DECISION_HZ) / SIM_DT)
G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])

# --- perception model -------------------------------------------------------
DETECTOR_HZ = 15.0  # measured 12-16 Hz in flight logs
DETECT_MAX_RANGE_M = 45.0
PNP_RANGE_NOISE_FRAC = 0.04
PNP_BEARING_NOISE_RAD = 0.010
ACTION_LATENCY_TICKS = 1  # ~20 ms of command transport at 50 Hz

# --- reward -----------------------------------------------------------------
COMPLETE_BONUS = 100.0
GATE_BONUS = 10.0
CRASH_PENALTY = 10.0
TIME_PENALTY = 0.02
MAX_HORIZ_ACCEL = 8.0  # ~39 deg of lean; beyond this the rate loop cannot hold it

COURSE_GATES = 17  # the VQ2 course length; the gate_idx feature normalizes by it

CURRICULUM = [
    {"n_gates": 1, "spacing": (10.0, 16.0), "jitter": 0.8},
    {"n_gates": 3, "spacing": (12.0, 20.0), "jitter": 1.6},
    {"n_gates": 8, "spacing": (12.0, 24.0), "jitter": 2.4},
    {"n_gates": 17, "spacing": (12.0, 28.0), "jitter": 3.0},
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
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


class VQ2RaceEnv(gym.Env):
    """Gymnasium env whose observation is what a VQ2 camera can deliver."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_gates: int = 17,
        seed: int | None = None,
        domain_rand: bool = True,
        detector_dropout: float = 0.18,  # measured ~18% of frames carry no box
        shaping: float = 1.0,
        max_seconds: float = 90.0,
        spacing: tuple[float, float] = (12.0, 28.0),
        jitter: float = 3.0,
        dr_scale: float = 1.0,
    ):
        super().__init__()
        self.dr_scale = float(dr_scale)
        self.n_gates = int(n_gates)
        self.domain_rand = bool(domain_rand)
        self.detector_dropout = float(detector_dropout)
        self.shaping = float(shaping)
        self.max_steps = int(max_seconds * DECISION_HZ)
        self.spacing = spacing
        self.jitter = float(jitter)

        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Box(
            -vo.OBS_ABS_MAX, vo.OBS_ABS_MAX, (vo.OBS_DIM,), np.float32
        )
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Normalize gate_idx by the REAL course length, never by the stage's
        # gate count. Otherwise the same observation value means different
        # things per stage (always 0 in a 1-gate stage, sweeping 0->1 in a
        # 17-gate one) and the curriculum changes the meaning of the input
        # rather than the difficulty. Measured: BC on stage-0 demos alone
        # scores 0.75, BC on all four stages mixed scores 0.00 -- four times
        # the data, strictly worse. It also has to match deployment, where the
        # course is always COURSE_GATES long.
        self.tracker = vo.GateFeatureTracker(n_gates=COURSE_GATES)
        self._build()

    # ---- course ------------------------------------------------------------
    def _build(self):
        lo, hi = self.spacing
        j = self.jitter
        gates, x, y, z = [], 0.0, 0.0, -4.0
        heading = 0.0
        for i in range(self.n_gates):
            x += float(self.np_random.uniform(lo, hi))
            y += float(self.np_random.uniform(-j, j))
            z += float(self.np_random.uniform(-j * 0.5, j * 0.5))
            z = float(np.clip(z, -18.0, -1.5))
            if i > 0:
                heading = float(self.np_random.uniform(-0.35, 0.35))
            gates.append(
                {
                    "pos": [x, y, z],
                    "quat": [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)],
                }
            )
        self.gates = gates

        # Plant. env.py's docstring claimed domain randomization and shipped
        # none; every constant was fixed, which is the classic sim-to-real trap.
        #
        # dr_scale widens the sampling range from nominal (0.0) to full (1.0).
        # It exists because a BC prior cloned on the nominal plant is BRITTLE to
        # exactly two of these, measured on stage 0 (1.00 = clears the gate):
        #     rate_gain     1.8 -> 0.00   2.7 -> 1.00   3.6 -> 0.00
        #     thrust_accel  0.9x-> 0.00   1.0x-> 1.00   1.1x-> 0.00
        #     drag / rate_tau / hover_thrust / latency:  0.95-1.00 throughout
        # On the full range the clone scores 0.37, so PPO launched straight into
        # it has almost no successful trajectory to reinforce and walks away
        # from the warm start. Ramp dr_scale up instead of starting at 1.0.
        if self.domain_rand and self.dr_scale > 0.0:
            u, s = self.np_random.uniform, float(np.clip(self.dr_scale, 0.0, 1.0))

            def jit(nominal, half):
                return float(nominal + s * half * u(-1.0, 1.0))

            self.hover_thrust = float(np.clip(jit(0.27, 0.27 * 0.15), 0.15, 0.45))
            self.thrust_accel = float(
                jit(spec.GRAVITY / 0.27, (spec.GRAVITY / 0.27) * 0.10)
            )
            self.rate_tau = float(np.clip(jit(0.05, 0.03), 0.02, 0.12))
            self.drag = float(np.clip(jit(0.15, 0.15), 0.02, 0.40))
            # The live plant amplifies rate commands ~2.7-3x (memory, live
            # confirmed) and has never been measured on this machine.
            self.rate_gain = float(np.clip(jit(2.7, 0.9), 1.0, 4.5))
            self.latency = int(round(np.clip(jit(1.0, 1.2), 0, 3)))
        else:
            self.hover_thrust = 0.27
            self.thrust_accel = spec.GRAVITY / 0.27
            self.rate_tau = 0.05
            self.drag = 0.15
            self.rate_gain = 2.7
            self.latency = ACTION_LATENCY_TICKS

        self.p = np.array([0.0, 0.0, float(gates[0]["pos"][2])])
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.gate_idx = 0
        self.steps = 0
        self.t = 0.0
        self.last_action = np.zeros(4)
        self._act_queue = [np.zeros(4) for _ in range(self.latency)]
        self._prev_signed = self._signed(0)
        self._prev_dist = float(np.linalg.norm(np.asarray(gates[0]["pos"]) - self.p))
        self._next_detect_t = 0.0
        self.tracker.n_gates = COURSE_GATES
        self.tracker.reset()

    # ---- gate geometry -----------------------------------------------------
    def _gate_normal(self, i):
        g = self.gates[i]
        return spec.quat_to_R(np.asarray(g["quat"], dtype=float)) @ np.array(
            [1.0, 0, 0]
        )

    def _gate_axes(self, i):
        """(right, down) axes spanning the gate plane."""
        R = spec.quat_to_R(np.asarray(self.gates[i]["quat"], dtype=float))
        return R[:, 1], R[:, 2]

    def _signed(self, i):
        gc = np.asarray(self.gates[i]["pos"], dtype=float)
        return float(self._gate_normal(i) @ (self.p - gc))

    def _corners_world(self, i):
        g = self.gates[i]
        R = spec.quat_to_R(np.asarray(g["quat"], dtype=float))
        half = HALF_OUTER
        local = np.array(
            [
                [0.0, -half, -half],
                [0.0, +half, -half],
                [0.0, +half, +half],
                [0.0, -half, +half],
            ]
        )
        return np.asarray(g["pos"], dtype=float) + local @ R.T

    # ---- perception --------------------------------------------------------
    def _render_detection(self):
        """Simulate YOLO+PnP for the current target gate, or None."""
        if self.gate_idx >= len(self.gates):
            return None
        if self.t < self._next_detect_t:
            return None
        self._next_detect_t = self.t + 1.0 / DETECTOR_HZ

        i = self.gate_idx
        gc = np.asarray(self.gates[i]["pos"], dtype=float)
        R_wb = spec.quat_to_R(self.q)
        gb = R_wb.T @ (gc - self.p)
        rng = float(np.linalg.norm(gb))
        if rng > DETECT_MAX_RANGE_M or gb[0] <= vo.MIN_FORWARD_M:
            return None

        # Keypoint visibility straight through the real camera model, which
        # already carries the 20 deg up-tilt.
        px, in_front = spec.project(self._corners_world(i), self.p, self.q)
        inb = (
            in_front
            & (px[:, 0] >= 0)
            & (px[:, 0] < spec.IMG_W)
            & (px[:, 1] >= 0)
            & (px[:, 1] < spec.IMG_H)
        )
        n_corners = int(np.count_nonzero(inb))
        if n_corners < 2:
            return None
        if self.np_random.random() < self.detector_dropout:
            return None

        noisy_rng = rng * (1.0 + PNP_RANGE_NOISE_FRAC * self.np_random.normal())
        noisy_rng = max(noisy_rng, vo.MIN_RANGE_M)
        d = gb / max(rng, 1e-9)
        d = d + PNP_BEARING_NOISE_RAD * self.np_random.normal(size=3)
        d = d / max(float(np.linalg.norm(d)), 1e-9)

        n_visible = int(np.clip(n_corners * 2, 0, 8))  # 4 outer -> 8 keypoints
        normal_body = R_wb.T @ (-self._gate_normal(i))
        return {
            "gate_pos_body": d * noisy_rng,
            "normal_body": normal_body,
            "range_m": noisy_rng,
            "reproj_px": 0.6,
            "n_visible": n_visible,
            "method": "ippe-yolo8",
            "conf": float(np.clip(0.55 + 0.05 * n_corners, 0.0, 0.99)),
        }

    def _obs(self, detection):
        gravity_body = spec.quat_to_R(self.q).T @ np.array([0.0, 0.0, 1.0])
        return self.tracker.update(
            self.t,
            detection,
            self.omega,
            gravity_body,
            self.last_action,
            min(self.gate_idx, self.n_gates - 1),
        )

    # ---- gym ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._build()
        return self._obs(self._render_detection()), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(4), -1, 1)
        self._act_queue.append(action.copy())
        applied = self._act_queue.pop(0) if self._act_queue else action

        roll, pitch, yaw, thrust = spec.scale_action(applied)
        omega_cmd = np.array([roll, pitch, yaw]) * self.rate_gain

        for _ in range(SUBSTEPS):
            self.omega += (omega_cmd - self.omega) * (SIM_DT / self.rate_tau)
            self.q = _quat_norm(
                _quat_mult(self.q, np.array([1.0, *(0.5 * self.omega * SIM_DT)]))
            )
            R = spec.quat_to_R(self.q)
            a = R @ np.array([0.0, 0.0, -self.thrust_accel * thrust])
            a = a + G_WORLD - self.drag * self.v
            self.v += a * SIM_DT
            self.p += self.v * SIM_DT

        self.steps += 1
        self.t += 1.0 / DECISION_HZ
        self.last_action = action

        reward, terminated, info = self._reward()
        truncated = self.steps >= self.max_steps
        obs = self._obs(self._render_detection())
        info.setdefault("gates_passed", self.gate_idx)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _reward(self):
        info: dict = {}
        rew = -TIME_PENALTY

        if self.gate_idx >= len(self.gates):
            return rew, True, {"course_complete": True}

        gc = np.asarray(self.gates[self.gate_idx]["pos"], dtype=float)
        dist = float(np.linalg.norm(gc - self.p))
        signed = self._signed(self.gate_idx)

        # Shaping is a scaffold: keyed to THIS gate's index, annealable to zero.
        rew += self.shaping * 0.5 * (self._prev_dist - dist)

        crossed = self._prev_signed < 0.0 <= signed
        if crossed:
            right, down = self._gate_axes(self.gate_idx)
            rel = self.p - gc
            dy, dz = abs(float(right @ rel)), abs(float(down @ rel))
            if dy < HALF_OPENING and dz < HALF_OPENING:
                # NOT scaled by `shaping`. Annealing the dense range-based
                # shaping to zero is the point; annealing the gate event to
                # zero as well leaves a 17-gate course with completion as its
                # only signal, which a policy at 0 gates never discovers -- its
                # best strategy becomes hovering to dodge the crash penalty,
                # which is exactly what stage 3 produced (gates=0.00 across
                # 400k steps). A gate pass is a discrete, unambiguous,
                # oracle-supplied event; it stays.
                rew += GATE_BONUS
                self.gate_idx += 1
                info["gate_passed"] = True
                if self.gate_idx >= len(self.gates):
                    rew += COMPLETE_BONUS  # the sparse objective
                    info["course_complete"] = True
                    info["gates_passed"] = self.gate_idx
                    return rew, True, info
                ngc = np.asarray(self.gates[self.gate_idx]["pos"], dtype=float)
                self._prev_dist = float(np.linalg.norm(ngc - self.p))
                self._prev_signed = self._signed(self.gate_idx)
                self.tracker.update(
                    self.t,
                    None,
                    self.omega,
                    spec.quat_to_R(self.q).T @ np.array([0.0, 0.0, 1.0]),
                    self.last_action,
                    self.gate_idx,
                )
                return rew, False, info
            if dy < HALF_OUTER and dz < HALF_OUTER:
                # Struck the frame. Terminating: otherwise grazing a gate is
                # nearly free and the policy learns to shave it.
                info["crash"] = "gate_frame"
                return rew - CRASH_PENALTY, True, info

        if self.p[2] > 0.0:
            info["crash"] = "ground"
            return rew - CRASH_PENALTY, True, info
        if dist > 70.0 or abs(self.p[2]) > 60.0:
            info["crash"] = "out_of_bounds"
            return rew - CRASH_PENALTY, True, info
        if (spec.quat_to_R(self.q).T @ np.array([0.0, 0.0, 1.0]))[2] < 0.0:
            info["crash"] = "flipped"
            return rew - CRASH_PENALTY, True, info

        self._prev_dist = dist
        self._prev_signed = signed
        return rew, False, info

    # ---- privileged expert (demos + solvability gate) ----------------------
    def expert_action(self) -> np.ndarray:
        """Geometric controller on the true state. Training-side only — it reads
        privileged state and can never run at race time; it exists to generate
        BC demonstrations and to prove the env is solvable at all.

        The action wire carries body RATES, so this closes an attitude loop and
        emits K*(desired_angle - current_angle). Commanding a rate as though it
        were an angle just spins the aircraft — the same confusion the repo
        records as the reason a policy emitting absolute attitude could never
        learn to stay upright.
        """
        if self.gate_idx >= len(self.gates):
            return np.zeros(4, dtype=np.float32)

        gc = np.asarray(self.gates[self.gate_idx]["pos"], dtype=float)
        to_gate = gc - self.p
        rng = max(float(np.linalg.norm(to_gate)), 1e-6)
        dir_w = to_gate / rng

        # Aim at the gate, then hold the desired velocity with an accel loop.
        # Both the horizontal and vertical accel demands are clamped: an
        # unbounded velocity error at 20 m range asks for ~26 m/s^2, which is
        # ~70 deg of tilt, and the rate loop simply flips the aircraft chasing
        # an attitude the plant cannot hold.
        speed = float(np.clip(2.5 + 0.55 * rng, 2.5, 12.0))
        a_des = 1.4 * (speed * dir_w - self.v)
        a_h = a_des[:2]
        n_h = float(np.linalg.norm(a_h))
        if n_h > MAX_HORIZ_ACCEL:
            a_h = a_h * (MAX_HORIZ_ACCEL / n_h)
        a_des = np.array([a_h[0], a_h[1], float(np.clip(a_des[2], -5.0, 5.0))])
        f_req = a_des - G_WORLD + self.drag * self.v
        T_acc = float(np.linalg.norm(f_req))
        if T_acc < 1e-6:
            return np.array([0, 0, 0, 2 * self.hover_thrust - 1], dtype=np.float32)

        b3_des = -f_req / T_acc  # body z (down) axis we want, in world
        roll_c, pitch_c, yaw_c = self._euler()
        yaw_des = float(np.arctan2(dir_w[1], dir_w[0]))

        # De-rotate the desired thrust axis by the CURRENT yaw to read off the
        # roll/pitch it implies.
        cy, sy = np.cos(yaw_c), np.sin(yaw_c)
        bx = cy * b3_des[0] + sy * b3_des[1]
        by = -sy * b3_des[0] + cy * b3_des[1]
        bz = b3_des[2]
        roll_des = float(np.arcsin(np.clip(-by, -1.0, 1.0)))
        pitch_des = float(np.arctan2(bx, max(bz, 1e-3)))

        wrap = lambda e: (e + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
        omega = np.array(
            [
                5.0 * wrap(roll_des - roll_c),
                5.0 * wrap(pitch_des - pitch_c),
                2.5 * wrap(yaw_des - yaw_c),
            ]
        )
        omega = np.clip(omega, -4.0, 4.0)

        thrust = float(np.clip(T_acc / self.thrust_accel, 0.0, 1.0))
        # Undo the plant's rate amplification so scale_action round-trips.
        cmd = omega / max(self.rate_gain, 1e-6)
        a = np.array(
            [
                cmd[0] / spec.MAX_ROLL_RATE,
                cmd[1] / spec.MAX_PITCH_RATE,
                cmd[2] / spec.MAX_YAW_RATE,
                2.0 * thrust - 1.0,
            ]
        )
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def _euler(self):
        """(roll, pitch, yaw) ZYX from the body quaternion."""
        w, x, y, z = self.q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)


def make_env(stage: int = 3, seed: int | None = None, **kw):
    cfg = CURRICULUM[int(np.clip(stage, 0, len(CURRICULUM) - 1))]

    def _thunk():
        return VQ2RaceEnv(
            n_gates=cfg["n_gates"],
            seed=seed,
            spacing=cfg["spacing"],
            jitter=cfg["jitter"],
            **kw,
        )

    return _thunk
