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

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl import spec
from rl.observation import build_observation

# Reduced quadrotor parameters. Nominal values match MEASURED real-sim dynamics
# (hover thrust ~0.27 -> thrust-accel ~36 m/s^2, pure rate integrator). The exact
# thrust-accel is uncertain, so _reset_state() randomizes it WIDE per episode.
MASS = 1.0
THRUST_ACCEL = spec.GRAVITY / spec.HOVER_THRUST  # nominal ~36; expert (control.py) uses this
RATE_TAU = 0.05  # body-rate first-order lag (s)
DRAG = 0.1
SIM_DT = 1 / 100.0
DECISION_HZ = 50
SUBSTEPS = int((1 / DECISION_HZ) / SIM_DT)

# Per-episode domain-randomization bands. Thrust band is deliberately wide to
# cover the open-loop measurement uncertainty (hover 0.19-0.33 -> accel ~30-52).
DR_HOVER_RANGE = (0.19, 0.33)
DR_RATE_TAU_RANGE = (0.04, 0.08)
DR_DRAG_RANGE = (0.05, 0.15)
DR_LATENCY_PROB = 0.5  # chance an episode applies a 1-step action latency

# Perception-noise bands — privileged-obs training proxy for GateNet->PnP->EKF.
DR_POS_NOISE_RANGE = (0.03, 0.15)  # EKF position drift std (m)
DR_VEL_NOISE_RANGE = (0.05, 0.25)  # velocity estimate std (m/s)
DR_ATT_NOISE_RANGE = (0.005, 0.02)  # attitude estimate std (rad)
DR_GATE_RANGE_FRAC = (0.01, 0.03)  # PnP gate-pos noise as a fraction of range

G_WORLD = np.array([0.0, 0.0, spec.GRAVITY])

# --- Reward shaping (MonoRace-style perception + smoothness terms) -----------
# Functional forms follow the A2RL'25 MonoRace reward. The lambdas are scaled to
# THIS env's reward magnitudes (per-step shaping is O(0.01-0.6); gate events are
# O(10-30)) and to our attitude-rate+thrust action space -- NOT the paper's raw
# 5-motor-PWM Table 1 values. Retune against the real table when available.
V_MAX = 15.0  # m/s; caps single-step progress reward (~0.3 m/step at 50 Hz)
DECISION_DT = 1.0 / DECISION_HZ
PERC_ANGLE_THRESH = np.pi / 3.0  # 60deg: target-gate-off-axis penalty turns on
LAMBDA_PERC = 0.05  # x theta_cam (rad, 1.05..pi) -> 0.05..0.16 when triggered
LAMBDA_RATE = 5e-4  # x ||omega||^2 (omega rad/s, up to ~41) -> <=0.02
LAMBDA_DRATE = 0.02  # x sum max(|du_i|-deadband, 0) over the 4 action dims
DRATE_THRESH = 0.05  # per-dim normalized-action change allowed free per step

# Camera optical axis (cam +z) in the body frame, for the gate-in-view penalty.
# Constant: depends only on the fixed mount tilt (CAM_TILT_DEG).
_CAM_AXIS_BODY = spec.R_BODY_CAM @ np.array([0.0, 0.0, 1.0])

# Curriculum stages. Synthetic stages procedurally lay out gates ahead (+x);
# the final stage loads the REAL captured course (rl/data/gate_map.json).
CURRICULUM = [
    {"num_gates": 1, "spawn_dist": 5.0, "jitter": 0.5},  # 0: single gate, close fixed
    {"num_gates": 1, "rand_dist": (3.0, 10.0), "jitter": 2.0},  # 1: single, random spawn
    {"num_gates": 2, "spawn_dist": 6.0, "jitter": 1.0},  # 2: two gates
    # 3-6: real course, grown one gate at a time to bridge the 2->6 chaining gap
    # (a single 2->6 jump left the policy at 0% completion). ~7.5 s/gate budget.
    {"num_gates": 3, "real_course": True, "max_seconds": 25.0},  # 3
    {"num_gates": 4, "real_course": True, "max_seconds": 32.0},  # 4
    {"num_gates": 5, "real_course": True, "max_seconds": 38.0},  # 5
    {"num_gates": 6, "real_course": True, "max_seconds": 45.0},  # 6: full real course
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
        max_seconds: float = 20.0,
        seed: int | None = None,
        domain_rand: bool = True,
        perception_noise: bool = True,
    ):
        super().__init__()
        self.stage = int(np.clip(stage, 0, len(CURRICULUM) - 1))
        self.cfg = CURRICULUM[self.stage]
        self.user_gate_map = gate_map
        self.max_steps = int(self.cfg.get("max_seconds", max_seconds) * DECISION_HZ)
        # Toggles: selftests / the geometric expert run on clean, nominal physics.
        self.domain_rand = domain_rand
        self.perception_noise = perception_noise
        self.action_space = spaces.Box(-1.0, 1.0, (spec.ACTION_DIM,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (spec.OBS_DIM,), np.float32)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()

    # ---- layout ---------------------------------------------------------
    def _make_gate_map(self):
        if self.user_gate_map is not None:
            return [dict(g) for g in self.user_gate_map[: self.cfg["num_gates"]]]
        if self.cfg.get("real_course"):
            return self._real_course_map()
        n, j = self.cfg["num_gates"], self.cfg["jitter"]
        rand_dist = self.cfg.get("rand_dist")
        gates, x = [], 0.0
        for i in range(n):
            if i == 0:
                x += self.np_random.uniform(*rand_dist) if rand_dist else self.cfg["spawn_dist"]
            else:
                x += self.np_random.uniform(5.0, 7.0)
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

    def _real_course_map(self):
        """Load the captured real course (first ``num_gates`` of it, so the
        intermediate stages train on a prefix of the true course), converting
        gate orientation into the env's convention. The sim's gates use local +y
        as the opening normal; the env (and observation/deploy) treat local +x as
        the normal, so post-rotate each quat by Rz(90deg) -> R[:,0] becomes the
        true opening normal (verified: it aligns with the -x course direction)."""
        from rl.sim_interface import load_gate_map

        qz = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        gates = []
        for g in load_gate_map()[: self.cfg["num_gates"]]:
            g = dict(g)
            g["quat"] = _quat_norm(
                _quat_mult(np.asarray(g["quat"], float), qz)
            ).tolist()
            gates.append(g)
        return gates

    def _sample_dynamics(self):
        """Per-episode plant parameters (domain randomization)."""
        if self.domain_rand:
            lo, hi = DR_HOVER_RANGE
            hover = self.np_random.uniform(lo, hi)
            self.thrust_accel = spec.GRAVITY / hover
            self.rate_tau = self.np_random.uniform(*DR_RATE_TAU_RANGE)
            self.drag = self.np_random.uniform(*DR_DRAG_RANGE)
            self.latency = self.np_random.uniform() < DR_LATENCY_PROB
        else:
            self.thrust_accel = THRUST_ACCEL
            self.rate_tau = RATE_TAU
            self.drag = DRAG
            self.latency = False
        self._delayed_action = np.zeros(4)
        # Per-episode perception-noise scales (independent of domain_rand).
        if self.perception_noise:
            self.pos_noise = self.np_random.uniform(*DR_POS_NOISE_RANGE)
            self.vel_noise = self.np_random.uniform(*DR_VEL_NOISE_RANGE)
            self.att_noise = self.np_random.uniform(*DR_ATT_NOISE_RANGE)
            self.gate_range_frac = self.np_random.uniform(*DR_GATE_RANGE_FRAC)
        else:
            self.pos_noise = self.vel_noise = self.att_noise = 0.0
            self.gate_range_frac = 0.0

    def _reset_state(self):
        self._sample_dynamics()
        self._percept_state = {}  # last-seen gate estimates (Phase 2 perception)
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
        # Face gate 0; the random-spawn stage adds an approach-angle offset.
        bearing = float(np.arctan2(g0[1] - self.p[1], g0[0] - self.p[0]))
        if self.cfg.get("rand_dist"):
            bearing += float(self.np_random.uniform(-0.5, 0.5))
        self.q = np.array([np.cos(bearing / 2), 0.0, 0.0, np.sin(bearing / 2)])
        self.omega = np.zeros(3)
        self.gate_idx = 0
        self.steps = 0
        self.last_action = np.zeros(4)
        self._prev_action = np.zeros(4)  # prior step's action, for smoothness term
        self._prev_dist = float(np.linalg.norm(g0 - self.p))
        self._prev_signed = self._signed_dist(self.gate_idx)
        # Safety envelope from the course geometry (handles the long real course
        # as well as the short synthetic ones).
        zs = [g["pos"][2] for g in self.gate_map]
        self.z_lo, self.z_hi = min(zs) - 5.0, max(zs) + 5.0
        pts = [self.p] + [np.asarray(g["pos"], float) for g in self.gate_map]
        gaps = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
        self.oob_dist = max(25.0, 1.6 * max(gaps))

    # ---- gate geometry --------------------------------------------------
    def _gate_frame(self, idx):
        g = self.gate_map[idx]
        gc = np.array(g["pos"])
        n = spec.quat_to_R(np.array(g["quat"])) @ np.array([1.0, 0, 0])
        return gc, n

    def _signed_dist(self, idx):
        gc, n = self._gate_frame(idx)
        return float(n @ (self.p - gc))

    # ---- perception (privileged-obs proxy for GateNet->PnP->EKF) ---------
    def _gate_visible(self, gate_pos):
        """Is the gate center inside the (tilted) camera FOV from the TRUE pose?"""
        px, in_front = spec.project(np.asarray([gate_pos]), self.p, self.q)
        if not in_front[0]:
            return False
        u, v = px[0]
        return 0.0 <= u < spec.IMG_W and 0.0 <= v < spec.IMG_H

    def _perceive(self):
        """Noisy (p,v,q) + a gate map whose current+next gate positions are
        perception estimates. Each gate is perceived INDEPENDENTLY: its world
        estimate refreshes (with PnP noise) only while it is itself visible, and
        is stale-held otherwise — so the next-gate lookahead keeps updating even
        as the current gate drops out of frame just before passing."""
        rng = self.np_random
        p_est = self.p + rng.normal(0, self.pos_noise, 3)
        v_est = self.v + rng.normal(0, self.vel_noise, 3)
        dq = np.array([1.0, *(0.5 * rng.normal(0, self.att_noise, 3))])
        q_est = _quat_norm(_quat_mult(self.q, dq))

        gm = [dict(g) for g in self.gate_map]
        n = len(gm)
        cur = min(self.gate_idx, n - 1)  # clip: gate_idx == n on the terminal obs
        for idx in {cur, min(cur + 1, n - 1)}:
            gpos = np.array(self.gate_map[idx]["pos"])
            if self._gate_visible(gpos):
                dist = float(np.linalg.norm(gpos - self.p))
                sigma = self.gate_range_frac * dist + 1e-3
                self._percept_state[idx] = gpos + rng.normal(0, sigma, 3)
            est = self._percept_state.get(idx)
            if est is None:  # never seen yet -> seed with truth (avoids garbage)
                est = gpos
                self._percept_state[idx] = est
            gm[idx]["pos"] = np.asarray(est).tolist()
        return p_est, v_est, q_est, gm

    def _obs(self):
        if self.perception_noise:
            p, v, q, gm = self._perceive()
        else:
            p, v, q, gm = self.p, self.v, self.q, self.gate_map
        return build_observation(
            p, v, q, self.omega, gm, self.gate_idx, self.last_action[:3]
        )

    # ---- gym API --------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        # Optional 1-step actuation latency (domain randomization).
        applied = self._delayed_action if self.latency else action
        self._delayed_action = action
        roll_r, pitch_r, yaw_r, thrust = spec.scale_action(applied)
        omega_cmd = np.array([roll_r, pitch_r, yaw_r])

        for _ in range(SUBSTEPS):
            self.omega += (omega_cmd - self.omega) * (SIM_DT / self.rate_tau)
            self.q = _quat_norm(
                _quat_mult(self.q, np.array([1.0, *(0.5 * self.omega * SIM_DT)]))
            )
            R = spec.quat_to_R(self.q)
            # thrust up (body -z), scaled by the per-episode thrust authority.
            f_body = np.array([0.0, 0.0, -self.thrust_accel * thrust])
            a = R @ f_body + G_WORLD - self.drag * self.v
            self.v += a * SIM_DT
            self.p += self.v * SIM_DT

        self.steps += 1
        self._prev_action = self.last_action
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

        # Diagnostics (read by the training callback; carried on every return
        # path since info is the same dict). raw_prog is UNCAPPED true-state
        # progress, to check whether the V_MAX cap ever engages.
        info["raw_prog"] = self._prev_dist - dist
        info["act_norm"] = float(np.linalg.norm(self.last_action))
        info["omega_norm"] = float(np.linalg.norm(self.omega))

        rew = 0.0
        # Dense progress, capped at v_max*dt so a single large step (e.g. a
        # stale-hold perception estimate snapping back) can't spike the reward.
        # Only the positive (forward) side is capped; regress stays fully penalized.
        prog = min(self._prev_dist - dist, V_MAX * DECISION_DT)
        rew += 2.0 * prog
        rew -= 0.01  # time penalty
        rew -= 0.001 * float(self.last_action @ self.last_action)  # effort

        # Velocity-toward-gate shaping.
        to_gate = gc - self.p
        ng = np.linalg.norm(to_gate)
        if ng > 1e-6:
            rew += 0.05 * float(self.v @ (to_gate / ng))

        # Perception penalty: discourage flying with the target gate far off the
        # camera optical axis (it drops from frame -> stale perception). MonoRace
        # p_perc; uses TRUE pose since reward is a privileged signal.
        if ng > 1e-6:
            axis_world = spec.quat_to_R(self.q) @ _CAM_AXIS_BODY
            cos_th = float(np.clip(axis_world @ (to_gate / ng), -1.0, 1.0))
            theta_cam = np.arccos(cos_th)
            if theta_cam > PERC_ANGLE_THRESH:
                rew -= LAMBDA_PERC * float(theta_cam)

        # Body-rate magnitude penalty (MonoRace p_rate, ||Omega||^2).
        rew -= LAMBDA_RATE * float(self.omega @ self.omega)

        # Action smoothness (anti bang-bang). Our 4-D attitude-rate+thrust action
        # stands in for MonoRace's 5 motor PWMs; jerky rate commands are the same
        # liability (untrackable by the rate loop, sim-to-real gap).
        du = np.abs(self.last_action - self._prev_action) - DRATE_THRESH
        rew -= LAMBDA_DRATE * float(np.sum(np.maximum(du, 0.0)))

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

        # Crash conditions (envelope derived per-episode from the course).
        if self.p[2] > self.z_hi or self.p[2] < self.z_lo:
            rew -= 20.0
            terminated = True
            info["crash"] = "altitude"
        elif dist > self.oob_dist or abs(self.p[1]) > 30.0:  # flew away
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
    print("[selftest] SB3 check_env passed (obs 24, act 4)")

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

    # Geometric expert controller should fly the full course on every stage.
    from rl.control import geometric_action

    for stage in range(len(CURRICULUM)):
        passes = total_gates = 0
        trials = 6
        for trial in range(trials):
            e = GateRacingEnv(
                stage=stage,
                seed=100 + trial,
                domain_rand=False,
                perception_noise=False,
            )
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

    # Perception: current + next gate are perceived INDEPENDENTLY. With the drone
    # just past gate 0 (behind = invisible) and gate 1 ahead (visible), gate 0's
    # estimate must stale-hold while gate 1's keeps refreshing.
    pe = GateRacingEnv(stage=1, seed=3, domain_rand=False, perception_noise=True)
    pe.reset()
    pe.gate_map = [
        {"pos": [5.0, 0.0, 0.0], "quat": [1.0, 0, 0, 0], "w": 2.72, "h": 2.72},
        {"pos": [11.0, 0.0, 0.0], "quat": [1.0, 0, 0, 0], "w": 2.72, "h": 2.72},
    ]
    pe.gate_idx, pe._percept_state = 0, {}
    pe.p, pe.q = np.array([6.0, 0.0, 0.0]), np.array([1.0, 0, 0, 0])
    pe.v, pe.omega = np.zeros(3), np.zeros(3)
    assert not pe._gate_visible(np.array([5.0, 0, 0])), "gate behind = invisible"
    assert pe._gate_visible(np.array([11.0, 0, 0])), "gate ahead = visible"
    pe._perceive()
    cur0, nxt0 = pe._percept_state[0].copy(), pe._percept_state[1].copy()
    pe.p = np.array([7.0, 0.0, 0.0])
    pe._perceive()
    assert np.array_equal(pe._percept_state[0], cur0), "invisible gate stale-held"
    assert not np.array_equal(pe._percept_state[1], nxt0), "visible next gate refreshes"
    o = pe._obs()
    assert np.all(np.isfinite(o)) and pe.observation_space.contains(o)
    print("[selftest] perception: independent per-gate stale-hold OK")
    print(
        f"[selftest] OK — env steps, rewards, gate-pass + "
        f"{len(CURRICULUM)}-stage curriculum wired"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
