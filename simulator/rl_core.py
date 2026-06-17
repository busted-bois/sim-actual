"""Pure (numpy-only) core for the residual-RL speed layer.

NO gymnasium / torch / sim imports here on purpose: every function is a pure
transform of plain numbers so it can be unit-tested without the simulator
(`uv run pytest`). The Gym env (`rl_env.py`), the deploy runtime
(`policy_runtime.py`), and the trainer (`rl_train.py`) all build on these.

The RL policy does NOT fly the drone directly. It emits a 3-vector action in
[-1, 1] that maps to a small GUIDANCE residual on top of the proven waypoint
pilot:
  - speed_mult     scales the pilot's target speed (can brake below OR push above)
  - lookahead_m    how far past the gate the carrot sits (line aggressiveness)
  - lateral_offset_m  shift the carrot perpendicular to the track (cut the corner)

The pilot's inner velocity/tilt/thrust loop and all its safety bounds stay
untouched (see pilot.py). Action ranges are WIDER SYMMETRIC per the Round-2
design grill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# Cadence
# --------------------------------------------------------------------------------------
RL_HZ = 20.0  # residual decision rate; held across the pilot's 250 Hz control ticks
RL_DT_S = 1.0 / RL_HZ

# --------------------------------------------------------------------------------------
# Action space -> guidance residual bounds  (WIDER SYMMETRIC, per grill Q5)
# --------------------------------------------------------------------------------------
SPEED_MULT_MIN = 0.4
SPEED_MULT_MAX = 2.5
LOOKAHEAD_MIN_M = 1.0
LOOKAHEAD_MAX_M = 8.0
LATERAL_OFFSET_MAX_M = 1.5  # symmetric: [-1.5, +1.5]

# Hard safety ceiling on the resulting target speed regardless of multiplier. The pilot
# was tuned slow+safe; this caps how fast the residual can ever drive it.
ABS_MAX_TARGET_SPEED_MPS = 12.0

ACTION_DIM = 3

# --------------------------------------------------------------------------------------
# Observation layout (~32D)
#   ego(8): vx_body, vy_body, vz, horiz_speed, roll, pitch, alt_err_next, dist_next
#   setpoint(3): speed_mult, lookahead, lateral_offset   (each normalized to ~[-1,1])
#   last_action(3)
#   next NUM_LOOKAHEAD_GATES gates x 6: body offset-to-center (3, scaled) + body approach-dir (3)
# --------------------------------------------------------------------------------------
NUM_LOOKAHEAD_GATES = 3
_PER_GATE = 6
OBS_DIM = 8 + 3 + 3 + NUM_LOOKAHEAD_GATES * _PER_GATE  # = 32

# Scales used to keep observation entries roughly O(1).
_POS_SCALE_M = 30.0  # gate offsets are tens of metres apart
_VEL_SCALE = 10.0
_DIST_SCALE_M = 30.0

# --------------------------------------------------------------------------------------
# Reward weights
# --------------------------------------------------------------------------------------
PROGRESS_W = 1.0  # per-metre closed toward the next gate (dense, drives learning)
GATE_BONUS = 10.0  # on each active_gate_index increment (clean pass)
ACTION_PEN = 0.02  # L2 penalty on the action -> stay near the pilot, smooth
CRASH_PENALTY = -50.0  # COLLISION -> terminal
MISS_PENALTY = -30.0  # flew past a gate plane outside the aperture -> terminal
TIMEOUT_PENALTY = -5.0
FINISH_BONUS = 50.0  # full track complete
# Completion-gated speed: time cost is only charged at a FULL finish, scaled by the
# curriculum time-weight (0 during the reliability phase, annealed up for speed).
TIME_COST_PER_S = 1.0


@dataclass
class GuidanceResidual:
    """Setpoint deltas the policy lays over the pilot. Identity = bare pilot."""

    speed_mult: float = 1.0
    lookahead_m: float = 3.0  # matches pilot.LOOKAHEAD_M default
    lateral_offset_m: float = 0.0


def identity_residual() -> GuidanceResidual:
    return GuidanceResidual()


def _lerp(t01: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t01


def action_to_residual(action) -> GuidanceResidual:
    """Map a policy action in [-1, 1]^3 to a bounded guidance residual.

    Out-of-range actions are clipped first, so a mis-scaled policy can never
    drive the setpoints outside the safe envelope.
    """
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} elements, got {a.shape[0]}")
    speed_mult = _lerp((a[0] + 1.0) * 0.5, SPEED_MULT_MIN, SPEED_MULT_MAX)
    lookahead = _lerp((a[1] + 1.0) * 0.5, LOOKAHEAD_MIN_M, LOOKAHEAD_MAX_M)
    lateral = a[2] * LATERAL_OFFSET_MAX_M
    return GuidanceResidual(
        speed_mult=float(speed_mult),
        lookahead_m=float(lookahead),
        lateral_offset_m=float(lateral),
    )


def residual_to_norm_action(res: GuidanceResidual) -> np.ndarray:
    """Inverse of action_to_residual (used to embed the current setpoint in the obs)."""
    a0 = (
        2.0 * (res.speed_mult - SPEED_MULT_MIN) / (SPEED_MULT_MAX - SPEED_MULT_MIN)
        - 1.0
    )
    a1 = (
        2.0 * (res.lookahead_m - LOOKAHEAD_MIN_M) / (LOOKAHEAD_MAX_M - LOOKAHEAD_MIN_M)
        - 1.0
    )
    a2 = res.lateral_offset_m / LATERAL_OFFSET_MAX_M
    return np.clip(np.array([a0, a1, a2], dtype=np.float64), -1.0, 1.0)


# --------------------------------------------------------------------------------------
# Geometry helpers (NED; z is +down)
# --------------------------------------------------------------------------------------
def gate_center(gate) -> tuple[float, float, float]:
    p = gate["position_ned"]
    return (float(p[0]), float(p[1]), float(p[2]))


def _norm3(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    return (v[0] / n, v[1] / n, v[2] / n), n


def approach_dir(gates, idx, drone_pos=None) -> tuple[float, float, float]:
    """Down-track through-direction for gate `idx`, from gate-to-gate geometry.

    Mirrors pilot._approach_dir so the obs/line geometry matches what the pilot flies:
    never flips backward when the drone overshoots.
    """
    center = gate_center(gates[idx])
    if idx + 1 < len(gates):
        nxt = gate_center(gates[idx + 1])
        d = (nxt[0] - center[0], nxt[1] - center[1], nxt[2] - center[2])
    elif idx > 0:
        prev = gate_center(gates[idx - 1])
        d = (center[0] - prev[0], center[1] - prev[1], center[2] - prev[2])
    elif drone_pos is not None:
        d = (
            center[0] - drone_pos[0],
            center[1] - drone_pos[1],
            center[2] - drone_pos[2],
        )
    else:
        d = (-1.0, 0.0, 0.0)
    u, n = _norm3(d)
    if n < 1e-6:
        return (-1.0, 0.0, 0.0)
    return u


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def gate_normal(gate, travel_hint=None) -> tuple[float, float, float]:
    """Through-direction (plane normal) of a gate, decoded from its orientation quat.

    The sim's gate model has its local Y axis as the through-axis (verified: the Round-1
    quat maps local Y -> world -X, matching 'fly through along -X'). Crossing ALONG this
    normal keeps the drone on the gate's central axis (locked y,z) instead of cutting toward
    the next gate and clipping the frame. `travel_hint` (e.g. gate-to-gate direction) orients
    the sign so it points the way we're going.
    """
    q = gate["orientation_ned"]  # (w, x, y, z)
    n = _quat_rotate(q, (0.0, 1.0, 0.0))
    u, mag = _norm3(n)
    if mag < 1e-6:
        return (-1.0, 0.0, 0.0)
    if travel_hint is not None:
        dot = u[0] * travel_hint[0] + u[1] * travel_hint[1] + u[2] * travel_hint[2]
        if dot < 0.0:
            u = (-u[0], -u[1], -u[2])
    return u


def ned_vec_to_body(dn: float, de: float, dd: float, yaw: float):
    """Rotate an NED vector into body frame (forward, right, down) about yaw."""
    fwd = dn * math.cos(yaw) + de * math.sin(yaw)
    right = -dn * math.sin(yaw) + de * math.cos(yaw)
    return fwd, right, dd


def gate_plane_progress(pos, gate, appr) -> float:
    """Signed along-track distance of the drone past the gate plane.

    <0 = approaching, >0 = gone past the gate center along the through-direction.
    """
    c = gate_center(gate)
    return (
        (pos[0] - c[0]) * appr[0]
        + (pos[1] - c[1]) * appr[1]
        + (pos[2] - c[2]) * appr[2]
    )


def lateral_miss_distance(pos, gate, appr) -> float:
    """Perpendicular distance from the drone to the gate's through-axis (metres)."""
    c = gate_center(gate)
    d = (pos[0] - c[0], pos[1] - c[1], pos[2] - c[2])
    along = d[0] * appr[0] + d[1] * appr[1] + d[2] * appr[2]
    perp = (d[0] - along * appr[0], d[1] - along * appr[1], d[2] - along * appr[2])
    return math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2)


def gate_aperture_radius(gate) -> float:
    """Half the smaller gate dimension — the clear opening the drone must fit through."""
    w = float(gate.get("width", 2.72))
    h = float(gate.get("height", w))
    return 0.5 * min(w, h)


def dist_to_gate_center(pos, gate) -> float:
    c = gate_center(gate)
    return math.sqrt((pos[0] - c[0]) ** 2 + (pos[1] - c[1]) ** 2 + (pos[2] - c[2]) ** 2)


# --------------------------------------------------------------------------------------
# Fly-away / frame-corruption guards
#
# A severe fly-away corrupts the sim's coordinate frame (drone reported km from the pad,
# unrecoverable by soft reset — verified 2026-06-15). These bound the damage: the pilot
# arrests when it gets implausibly far from its target gate (so it never reaches the
# runaway regime), and the env detects a corrupted frame after a reset.
# --------------------------------------------------------------------------------------
FLYAWAY_FROM_TARGET_M = (
    60.0  # gates are <=37 m apart, lateral within +-5 m; >60 m = lost
)
FRAME_CORRUPT_ORIGIN_M = 300.0  # right after a reset the drone should be near the pad


def is_flyaway(pos, target_center, limit: float = FLYAWAY_FROM_TARGET_M) -> bool:
    """True if the drone is implausibly far from its target gate -> arrest control."""
    d = math.sqrt(
        (pos[0] - target_center[0]) ** 2
        + (pos[1] - target_center[1]) ** 2
        + (pos[2] - target_center[2]) ** 2
    )
    return d > limit


def frame_looks_corrupted(pos, limit: float = FRAME_CORRUPT_ORIGIN_M) -> bool:
    """True if the pose right after a reset is nowhere near the origin -> frame corrupted,
    needs a full sim restart (soft reset can't recover it)."""
    if pos is None:
        return False
    return math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2) > limit


# --------------------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------------------
def build_observation(
    pos,
    vel_ned,
    roll: float,
    pitch: float,
    yaw: float,
    gates,
    active_idx: int,
    residual: GuidanceResidual,
    last_action,
) -> np.ndarray:
    """Assemble the fixed-size (OBS_DIM) observation. Pure function of state."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    n_gates = len(gates)

    vx_b, vy_b, _ = ned_vec_to_body(vel_ned[0], vel_ned[1], vel_ned[2], yaw)
    horiz_speed = math.hypot(vel_ned[0], vel_ned[1])

    # next gate (clamped to last) for ego-relative altitude/distance terms
    nxt = min(active_idx, n_gates - 1) if n_gates > 0 else 0
    if n_gates > 0:
        c = gate_center(gates[nxt])
        alt_err = pos[2] - c[2]
        dist_next = dist_to_gate_center(pos, gates[nxt])
    else:
        alt_err = 0.0
        dist_next = 0.0

    obs[0] = vx_b / _VEL_SCALE
    obs[1] = vy_b / _VEL_SCALE
    obs[2] = vel_ned[2] / _VEL_SCALE
    obs[3] = horiz_speed / _VEL_SCALE
    obs[4] = roll
    obs[5] = pitch
    obs[6] = alt_err / _POS_SCALE_M
    obs[7] = dist_next / _DIST_SCALE_M

    obs[8:11] = residual_to_norm_action(residual)

    la = np.asarray(last_action, dtype=np.float32).reshape(-1)
    if la.shape[0] >= ACTION_DIM:
        obs[11:14] = np.clip(la[:ACTION_DIM], -1.0, 1.0)

    base = 14
    for k in range(NUM_LOOKAHEAD_GATES):
        gi = active_idx + k
        off = base + k * _PER_GATE
        if 0 <= gi < n_gates:
            c = gate_center(gates[gi])
            dn, de, dd = c[0] - pos[0], c[1] - pos[1], c[2] - pos[2]
            fwd, right, down = ned_vec_to_body(dn, de, dd, yaw)
            obs[off + 0] = fwd / _POS_SCALE_M
            obs[off + 1] = right / _POS_SCALE_M
            obs[off + 2] = down / _POS_SCALE_M
            appr = approach_dir(gates, gi, pos)
            afwd, aright, adown = ned_vec_to_body(appr[0], appr[1], appr[2], yaw)
            obs[off + 3] = afwd
            obs[off + 4] = aright
            obs[off + 5] = adown
        # else: zero-padded (past the end of the track)
    return obs


# --------------------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------------------
def step_reward(prev_dist: float, cur_dist: float, gate_passed: bool, action) -> float:
    """Dense per-step reward: progress toward the next gate + clean-pass bonus
    - action magnitude penalty (smoothness / stay near pilot)."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    r = PROGRESS_W * (prev_dist - cur_dist)
    if gate_passed:
        r += GATE_BONUS
    r -= ACTION_PEN * float(np.dot(a, a))
    return float(r)


def finish_reward(episode_time_s: float, time_weight: float) -> float:
    """Terminal reward on a FULL finish. Speed (time cost) is gated on completion
    and scaled by the curriculum time_weight (0 => reliability phase)."""
    return FINISH_BONUS - time_weight * TIME_COST_PER_S * episode_time_s


# --------------------------------------------------------------------------------------
# Curriculum schedules
# --------------------------------------------------------------------------------------
def reverse_curriculum_start_gate(
    progress01: float, num_gates: int, hardest_gate: int = 2
) -> int:
    """Reverse curriculum: begin episodes near the hardest gate, expand backward to
    gate 0 as training progresses. progress01 in [0,1]."""
    progress01 = min(1.0, max(0.0, progress01))
    earliest = int(round(hardest_gate * (1.0 - progress01)))
    return max(0, min(earliest, num_gates - 1))


def time_weight_schedule(
    progress01: float, reliability_frac: float = 0.5, max_weight: float = 1.0
) -> float:
    """0 during the reliability phase (first `reliability_frac` of training), then
    anneal linearly up to `max_weight` for the speed phase."""
    progress01 = min(1.0, max(0.0, progress01))
    if progress01 <= reliability_frac:
        return 0.0
    return max_weight * (progress01 - reliability_frac) / (1.0 - reliability_frac)
