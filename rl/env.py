"""Module 7 — Gym training environment + curriculum.

A Gymnasium env wrapping a LIGHTWEIGHT INTERNAL quadrotor physics model. It
does NOT connect to the Anduril simulator — stepping a real sim per RL step
would be far too slow. The live sim is used only in Modules 1-2 (data) and
final evaluation. Domain randomization on the internal model covers the
sim-to-sim gap.

  * Action      : 4-D normalized [-1,1] -> (roll,pitch,yaw rate, thrust)  [spec.scale_action]
  * Observation : 24-D gate-relative vector                              [Module 6]
  * Reward      : dense progress + gate-pass bonus - crash/time/effort
  * Curriculum  : stage 0 single close gate -> 1 two gates -> 2 full 6-gate course

    uv run -m rl.env --selftest
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import spec
from rl.calibration import load_calibration
from rl.observation import build_observation

# Reduced quadrotor parameters — measured overrides when calibration exists.
_CAL = load_calibration()
MASS = 1.0
THRUST_ACCEL = float(_CAL.get("thrust_accel", spec.GRAVITY / spec.HOVER_THRUST))
RATE_TAU = float(_CAL.get("rate_tau_s", 0.05))  # body-rate first-order lag (s)
# DRAG was a hardcoded 0.1 guess (terminal ~37 m/s → the deployed policy
# overshot every gate). Now calibration-backed: `make dynamics` fits it from a
# live coast-down. Default unchanged so behavior is identical until measured.
DRAG = float(_CAL.get("drag", 0.1))
# Soft speed-envelope cap. Even before drag is fit, keep the policy from
# learning 37 m/s dashes it can't reproduce on the real plant. Fitted `v_max`
# (terminal speed at cruise tilt) overrides the conservative default.
V_MAX = float(_CAL.get("v_max", 20.0))
VMAX_DAMP = 2.0  # extra drag (1/s) per m/s above V_MAX
SIM_DT = 1 / 100.0
DECISION_HZ = 50
SUBSTEPS = int((1 / DECISION_HZ) / SIM_DT)

G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])

# --- Domain randomization (sim-to-real robustness) --------------------------
# Applied per-episode ONLY when randomize=True (training). Selftest/eval/demos
# run the nominal calibrated plant. Ranges cover the residual gap between this
# internal model and the calibrated live plant, so the policy tolerates what it
# actually meets on deploy instead of overfitting one fixed set of dynamics.
DR_THRUST_SCALE = (0.85, 1.15)  # mass + thrust-curve error
DR_RATE_TAU_SCALE = (0.7, 1.4)  # inner-loop lag / rate-gain spread
DR_DRAG_SCALE = (0.5, 1.5)
DR_LATENCY_STEPS = (0, 2)  # decision-step command delay (MAVLink + inner loop)
DR_OBS_POS_BIAS = 0.05  # m, per-episode constant estimator offset
DR_OBS_VEL_BIAS = 0.3  # m/s — larger: the deploy velocity (IMU+vision fusion)
DR_OBS_POS_NOISE = 0.03  # m, per-step
DR_OBS_VEL_NOISE = 0.3  # m/s, per-step — carries real error, so train tolerant
DR_OBS_RATE_NOISE = 0.02  # rad/s, per-step
DR_OBS_ANG_NOISE = 0.02  # rad, per-step attitude noise
DR_MASK_NEXT_PROB = 0.5  # fraction of episodes with the next-gate obs hidden
DR_RANDOM_START_PROB = 0.5  # fraction of TRAINING episodes spawned mid-course
# VQ2 gate pose comes from YOLO+PnP (not a map), so it carries its own error
# independent of the state estimator — noise the gate-relative obs slices too.
DR_GATE_DIR_NOISE = 0.03  # unit-vector component noise (~2 deg)
DR_GATE_DIST_NOISE = 0.05  # fractional range error (~5%)

K_SMOOTH = 0.02  # action-delta (jerk) penalty — smooth rate cmds transfer better

# --- Diagnostic corruption (flag-gated, eval only). Tests whether the LIVE
# observation statistics collapse the EXISTING policy, to prove the OOD source
# before spending a retrain. RL_CORR_VEL: replace the ~clean velocity slot with
# live-magnitude correlated velocity noise (live vision-diff vel is ~6 m/s std
# with spurious lateral/vertical; training vel is ~0.3). RL_CORR_JIT: big
# correlated gate-pose jitter (~40% range) like real YOLO+PnP.
_CORR_VEL = os.environ.get("RL_CORR_VEL", "") == "1"
_CORR_JIT = os.environ.get("RL_CORR_JIT", "") == "1"
CORR_VEL_STD = 6.0  # m/s, matches live vision-diff velocity spread
CORR_JIT_DIR = 0.15  # ~9 deg bearing bounce
CORR_JIT_DIST = 0.4  # 40% range jitter

# Curriculum stages: num_gates, first-gate distance, layout jitter, episode
# time budget. Final stage is VQ2 full-course length (17 gates) so the policy
# trains on sustained gate-to-gate runs, not just short sprints — its time
# budget scales up so a full course fits in one episode.
# `vel_walk` = per-stage CAP on correlated velocity-observation noise (m/s std)
# modeling the live vision-differenced velocity. OFF (0.0) by default: training
# through ~3 m/s velocity noise proved UNLEARNABLE (the signal degrades below
# what precise gate-chaining needs; retrain 2026-07-21 collapsed clean 15.4->5.0
# AND stayed bad under noise). The real fix was deploy-side (rotation-compensated
# vision velocity) — see simulator/rl_pilot. Kept as a knob for future modest DR
# (<=1.5) once the deploy velocity is measured clean.
CURRICULUM = [
    {
        "num_gates": 1,
        "spawn_dist": 5.0,
        "jitter": 0.5,
        "max_seconds": 20.0,
        "vel_walk": 0.0,
    },
    {
        "num_gates": 2,
        "spawn_dist": 6.0,
        "jitter": 1.0,
        "max_seconds": 20.0,
        "vel_walk": 0.0,
    },
    {
        "num_gates": 6,
        "spawn_dist": 7.0,
        "jitter": 2.0,
        "max_seconds": 30.0,
        "vel_walk": 0.0,
    },
    {
        "num_gates": 17,
        "spawn_dist": 7.0,
        "jitter": 2.0,
        "max_seconds": 70.0,
        "vel_walk": 0.0,
    },
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


class GateRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: int = 0,
        gate_map: list | None = None,
        max_seconds: float | None = None,
        seed: int | None = None,
        randomize: bool = False,
    ):
        super().__init__()
        self.stage = int(np.clip(stage, 0, len(CURRICULUM) - 1))
        self.cfg = CURRICULUM[self.stage]
        self.user_gate_map = gate_map
        # Per-stage episode budget (a 17-gate course needs far more than 20 s);
        # an explicit max_seconds still wins for callers that set it.
        ms = (
            max_seconds
            if max_seconds is not None
            else self.cfg.get("max_seconds", 20.0)
        )
        self.max_steps = int(ms * DECISION_HZ)
        self.randomize = bool(randomize)
        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, (spec.POLICY_OBS_DIM,), np.float32
        )
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
                    # w/h = flyable INNER opening (pass-check + reward use these).
                    "w": spec.GATE_OPENING_M,
                    "h": spec.GATE_OPENING_M,
                }
            )
        return gates

    def _reset_state(self):
        self.gate_map = self._make_gate_map()
        n = len(self.gate_map)
        # Random-start (TRAINING only): spawn behind a random gate so the policy
        # trains on DEEP gates too. From a gate-0 start it crashes before
        # reaching the tail, so the last gates are never practiced and
        # completion stalls. Eval (randomize=False) always starts at gate 0 for
        # an honest full-course metric — and the gate-0 spawn is unchanged so
        # metrics stay comparable across runs.
        start_idx = 0
        if self.randomize and n > 2 and self.np_random.uniform() < DR_RANDOM_START_PROB:
            start_idx = int(self.np_random.integers(1, n - 1))
        gs = np.array(self.gate_map[start_idx]["pos"])
        if start_idx == 0:
            # Original spawn: behind gate 0, near its altitude, from rest.
            self.p = np.array(
                [
                    0.0,
                    self.np_random.uniform(-0.5, 0.5),
                    gs[2] + self.np_random.uniform(-0.5, 0.5),
                ]
            )
            self.v = np.zeros(3)
        else:
            # Mid-course start: a few metres behind the gate, carrying the
            # forward speed you'd actually have there (not hovering).
            back = float(self.np_random.uniform(3.0, 5.0))
            self.p = np.array(
                [
                    gs[0] - back,
                    gs[1] + self.np_random.uniform(-0.5, 0.5),
                    gs[2] + self.np_random.uniform(-0.5, 0.5),
                ]
            )
            self.v = np.array([float(self.np_random.uniform(1.0, 2.5)), 0.0, 0.0])
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.omega = np.zeros(3)
        self.gate_idx = start_idx
        self.steps = 0
        self.last_action = np.zeros(4)
        self._prev_dist = float(np.linalg.norm(gs - self.p))
        self._prev_signed = self._signed_dist(self.gate_idx)
        self._velc = np.zeros(3)  # correlated velocity-noise state (AR1)
        self._jit = np.zeros(3)
        self._sample_dynamics()  # sets self._obs_noise
        # Per-episode velocity-noise magnitude, annealed by stage; models the
        # live vision-differenced velocity the policy meets on deploy.
        cap = float(self.cfg.get("vel_walk", 0.0))
        self._vel_walk_std = (
            float(self.np_random.uniform(0.0, cap)) if self._obs_noise else 0.0
        )
        # Prime the frame stack with copies of the first frame (no history yet).
        f0 = self._frame()
        self._stack = deque(
            (f0.copy() for _ in range(spec.OBS_STACK)), maxlen=spec.OBS_STACK
        )

    def _stacked(self):
        return np.concatenate(self._stack, dtype=np.float32)

    def _push_obs(self):
        self._stack.append(self._frame())
        return self._stacked()

    def _sample_dynamics(self):
        """Per-episode effective plant + estimator params (domain randomization).

        Nominal (calibrated) values when randomize=False so selftest/eval/demos
        are deterministic; sampled around them when randomize=True so the policy
        is robust to the internal-vs-live plant gap on deploy."""
        self._action_buf: list = []
        if not self.randomize:
            self._thrust_accel = THRUST_ACCEL
            self._rate_tau = RATE_TAU
            self._drag = DRAG
            self._latency_steps = 0
            self._obs_pos_bias = np.zeros(3)
            self._obs_vel_bias = np.zeros(3)
            self._obs_noise = False
            self._mask_next = False
            return
        r = self.np_random
        self._thrust_accel = THRUST_ACCEL * float(r.uniform(*DR_THRUST_SCALE))
        self._rate_tau = RATE_TAU * float(r.uniform(*DR_RATE_TAU_SCALE))
        self._drag = DRAG * float(r.uniform(*DR_DRAG_SCALE))
        self._latency_steps = int(
            r.integers(DR_LATENCY_STEPS[0], DR_LATENCY_STEPS[1] + 1)
        )
        self._obs_pos_bias = r.normal(0.0, DR_OBS_POS_BIAS, 3)
        self._obs_vel_bias = r.normal(0.0, DR_OBS_VEL_BIAS, 3)
        self._obs_noise = True
        # VQ2 has no gate map: the next gate is only in the observation when a
        # SECOND gate is actually visible to YOLO. Mask it in a fraction of
        # episodes so the policy can fly on the current gate alone and never
        # depends on a next-gate vector it won't always get at deploy.
        self._mask_next = bool(r.uniform() < DR_MASK_NEXT_PROB)

    # ---- gate geometry --------------------------------------------------
    def _gate_frame(self, idx):
        g = self.gate_map[idx]
        gc = np.array(g["pos"])
        n = spec.quat_to_R(np.array(g["quat"])) @ np.array([1.0, 0, 0])
        return gc, n

    def _signed_dist(self, idx):
        gc, n = self._gate_frame(idx)
        return float(n @ (self.p - gc))

    def _frame(self):
        """One 24-D observation frame (pre-stack)."""
        p, v, q, omega = self.p, self.v, self.q, self.omega
        if self._obs_noise:
            # Model the EKF/odometry error the policy meets live: per-episode
            # bias + per-step noise on the state (NOT the gate map — perception
            # is decoupled, gate pose comes from vision at deploy).
            r = self.np_random
            p = p + self._obs_pos_bias + r.normal(0.0, DR_OBS_POS_NOISE, 3)
            v = v + self._obs_vel_bias + r.normal(0.0, DR_OBS_VEL_NOISE, 3)
            omega = omega + r.normal(0.0, DR_OBS_RATE_NOISE, 3)
            q = _quat_norm(
                _quat_mult(
                    q, np.array([1.0, *(0.5 * r.normal(0.0, DR_OBS_ANG_NOISE, 3))])
                )
            )
        obs = build_observation(
            p, v, q, omega, self.gate_map, self.gate_idx, self.last_action[:3]
        )
        L = spec.OBS_LAYOUT
        if self._obs_noise:  # YOLO+PnP gate-pose error
            r = self.np_random
            obs[L["to_gate_body"]] += r.normal(0.0, DR_GATE_DIR_NOISE, 3)
            obs[L["gate_normal_body"]] += r.normal(0.0, DR_GATE_DIR_NOISE, 3)
            obs[L["dist_to_gate"]] *= 1.0 + r.normal(0.0, DR_GATE_DIST_NOISE)
            obs[L["to_next_gate_body"]] += r.normal(0.0, DR_GATE_DIR_NOISE, 3)
        if self._mask_next:  # VQ2: next gate not always visible
            obs[L["to_next_gate_body"]] = 0.0
            obs[L["dist_to_next_gate"]] = 0.0
        # Correlated velocity-observation noise = the live vision-differenced
        # velocity. Training: per-episode annealed std (self._vel_walk_std).
        # Eval: RL_CORR_VEL forces the fixed live magnitude as a stress test.
        vstd = CORR_VEL_STD if _CORR_VEL else self._vel_walk_std
        if vstd > 0.0:
            self._velc = 0.6 * self._velc + 0.4 * self.np_random.normal(0, vstd, 3)
            obs[L["vel_body"]] = np.clip(obs[L["vel_body"]] + self._velc / 10.0, -5, 5)
        if _CORR_JIT:  # diagnostic: live-magnitude correlated gate-pose jitter
            self._jit = 0.7 * self._jit + 0.3 * self.np_random.normal(
                0, CORR_JIT_DIR, 3
            )
            obs[L["to_gate_body"]] = obs[L["to_gate_body"]] + self._jit
            obs[L["dist_to_gate"]] *= (
                1.0 + 0.7 * self._jit[0] / CORR_JIT_DIR * CORR_JIT_DIST
            )
        return obs

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()
        return self._stacked(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        # Action latency: apply the command from _latency_steps decisions ago
        # (models MAVLink round-trip + inner-loop delay). No-op when latency=0.
        self._action_buf.append(action.copy())
        if len(self._action_buf) > self._latency_steps:
            applied = self._action_buf.pop(0)
        else:
            applied = action
        roll_r, pitch_r, yaw_r, thrust = spec.scale_action(applied)
        omega_cmd = np.array([roll_r, pitch_r, yaw_r])
        ta, tau, drag = self._thrust_accel, self._rate_tau, self._drag

        for _ in range(SUBSTEPS):
            self.omega += (omega_cmd - self.omega) * (SIM_DT / tau)
            self.q = _quat_norm(
                _quat_mult(self.q, np.array([1.0, *(0.5 * self.omega * SIM_DT)]))
            )
            R = spec.quat_to_R(self.q)
            f_body = np.array([0.0, 0.0, -ta * thrust])  # thrust up (body -z)
            a = R @ f_body + G_WORLD - drag * self.v
            speed = float(np.linalg.norm(self.v))
            if speed > V_MAX:  # soft envelope cap — no 37 m/s dashes
                a -= VMAX_DAMP * (speed - V_MAX) * (self.v / speed)
            self.v += a * SIM_DT
            self.p += self.v * SIM_DT

        self.steps += 1
        prev_action = self.last_action
        self.last_action = action
        rew, terminated, info = self._reward_done(prev_action)
        truncated = self.steps >= self.max_steps
        if terminated or truncated:
            # Episode-end summary for the training metrics callback.
            info["gates_passed_total"] = int(self.gate_idx)
            info["completed"] = bool(info.get("course_complete", False))
        return self._push_obs(), float(rew), bool(terminated), bool(truncated), info

    # ---- reward / termination ------------------------------------------
    def _reward_done(self, prev_action):
        info = {}
        gc, n = self._gate_frame(self.gate_idx)
        dist = float(np.linalg.norm(gc - self.p))
        signed = float(n @ (self.p - gc))

        rew = 0.0
        rew += 2.0 * (self._prev_dist - dist)  # dense progress
        rew -= 0.01  # time penalty
        rew -= 0.001 * float(self.last_action @ self.last_action)  # effort
        da = self.last_action - prev_action  # action smoothness (jerk penalty)
        rew -= K_SMOOTH * float(da @ da)

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
            hw = 0.5 * float(g.get("w", spec.GATE_OPENING_M))
            hh = 0.5 * float(g.get("h", spec.GATE_OPENING_M))
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


def make_env(stage=0, gate_map=None, seed=None, randomize=False):
    def _thunk():
        return GateRacingEnv(
            stage=stage, gate_map=gate_map, seed=seed, randomize=randomize
        )

    return _thunk


# ---------------------------------------------------------------------------
def _selftest():
    from stable_baselines3.common.env_checker import check_env

    env = GateRacingEnv(stage=0, seed=0)
    check_env(env, warn=True)
    print("[selftest] SB3 check_env passed (obs 24, act 4)")

    # Random rollout — env must run and terminate sanely.
    obs, _ = env.reset(seed=1)
    assert obs.shape == (spec.POLICY_OBS_DIM,)
    total, steps = 0.0, 0
    term = trunc = False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        steps += 1
    print(f"[selftest] random rollout {steps} steps reward={total:.1f} info={info}")

    # Geometric expert controller should fly the full course on every stage.
    from rl.control import geometric_action

    for stage in range(len(CURRICULUM)):
        passes = total_gates = 0
        trials = 6
        for trial in range(trials):
            e = GateRacingEnv(stage=stage, seed=100 + trial)
            o, _ = e.reset()
            total_gates += len(e.gate_map)
            term = trunc = False
            while not (term or trunc):
                tgt = np.array(e.gate_map[e.gate_idx]["pos"])
                o, r, term, trunc, info = e.step(geometric_action(e.p, e.v, e.q, tgt))
                if info.get("gate_passed"):
                    passes += 1
        print(
            f"[selftest] stage {stage}: expert passed {passes}/{total_gates} "
            f"gate-crossings over {trials} episodes"
        )
        assert passes >= trials, f"expert should clear stage {stage} gates"

    # GP (AndurilGP guidance) expert should clear stage 0 like the geometric one.
    from rl.gp_expert import GPExpert

    gp = GPExpert()
    gp_passes = 0
    trials = 6
    for trial in range(trials):
        e = GateRacingEnv(stage=0, seed=200 + trial)
        e.reset()
        gp.reset()
        term = trunc = False
        while not (term or trunc):
            a = gp.act(e.p, e.v, e.q, e.gate_map, e.gate_idx)
            _, _, term, trunc, info = e.step(a)
            if info.get("gate_passed"):
                gp_passes += 1
                break
    print(f"[selftest] GP expert passed {gp_passes}/{trials} stage-0 gates")
    assert gp_passes >= trials - 1, "GP expert should clear nearly all stage-0 gates"

    # Domain randomization smoke: env must run + stay finite with DR on, and the
    # GP expert (which flies a randomized plant) should still clear MOST gates —
    # if DR is so wide the expert fails, BC would clone garbage.
    dr_env = GateRacingEnv(stage=0, seed=7, randomize=True)
    o, _ = dr_env.reset(seed=7)
    assert np.all(np.isfinite(o)), "DR obs must be finite"
    for _ in range(50):
        o, r, term, trunc, _ = dr_env.step(dr_env.action_space.sample())
        assert np.all(np.isfinite(o)) and np.isfinite(r), "DR step must stay finite"
        if term or trunc:
            break
    dr_passes = 0
    for trial in range(8):
        e = GateRacingEnv(stage=0, seed=300 + trial, randomize=True)
        e.reset()
        gp.reset()
        term = trunc = False
        while not (term or trunc):
            a = gp.act(e.p, e.v, e.q, e.gate_map, e.gate_idx)
            _, _, term, trunc, info = e.step(a)
            if info.get("gate_passed"):
                dr_passes += 1
                break
    print(f"[selftest] GP expert passed {dr_passes}/8 stage-0 gates under DR")
    # Fixed expert degrades under DR (the RL policy instead LEARNS through it);
    # this only guards against DR so wide the plant is unflyable.
    assert dr_passes >= 4, "DR too wide — expert fails; narrow the ranges"
    print("[selftest] OK — env steps, rewards, gate-pass, curriculum + DR wired")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
