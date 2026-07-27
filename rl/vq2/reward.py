"""Reward + termination for the VQ2 gate-racing env.

Pure functions (no sim/env state) so they unit-test offline. The env passes in
the quantities it already computed for the observation. Terminal events
(collision / crash / out-of-bounds / course-complete) end the episode as
`terminated`; the step/time limit ends it as `truncated` (Gymnasium semantics).

All reward terms from the spec map here:
  positive: gate passed, progress toward gate, forward speed, smoothness, finish-fast
  negative: collision, out-of-bounds, crash/flip, excessive angular rate,
            attitude oscillation, moving away from gate, timeout
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- tunable weights (kept close to rl/env.py's shaping so BC transfers) -----
GATE_PASS_BONUS = 10.0        # + per gate cleared
COURSE_BONUS = 30.0           # + on final gate (course complete)
PROGRESS_K = 2.0              # + per metre closed toward the gate (also - when opening)
FWD_SPEED_K = 0.05            # + per (m/s) forward body velocity
SMOOTH_K = 0.02               # - per ‖Δaction‖² (jerk / oscillation)
OSC_K = 0.01                  # - per ‖roll/pitch rate cmd‖² (attitude oscillation)
RATE_K = 0.005               # - per ‖ang-rate cmd‖² (excessive angular rate)
TIME_K = 0.01                 # - per step (finish quickly)
COLLISION_PEN = -20.0
SOFT_COLLISION_PEN = -0.02    # - proximity warning (fires constantly near gates;
                              #   keep tiny so approaching a gate isn't punished)
OOB_PEN = -20.0
CRASH_PEN = -20.0
TIMEOUT_PEN = -5.0


@dataclass
class StepCtx:
    prev_dist: float | None      # range to current gate last step (None if was unseen)
    dist: float | None           # range to current gate this step (None if unseen)
    visible: bool
    vel_body: np.ndarray         # (3,) body FRD m/s
    action: np.ndarray           # (4,) normalized this step
    prev_action: np.ndarray      # (4,) normalized last step
    gate_passed: bool
    course_complete: bool
    collision: bool              # HARD contact -> terminal
    out_of_bounds: bool
    flipped: bool
    timeout: bool
    soft_collision: bool = False  # proximity warning -> small penalty, keep flying


def step(ctx: StepCtx):
    """-> (reward, terminated, truncated, reason, parts).

    `parts` is a per-term breakdown of this step's reward (progress, fwd, smooth,
    osc, rate, time, soft_col, gate_bonus, terminal) so the trainer can log which
    components actually fire -- i.e. verify the reward function is working."""
    p = {}

    # progress toward the visible gate (positive closing, negative opening).
    p["progress"] = (PROGRESS_K * (ctx.prev_dist - ctx.dist)
                     if ctx.visible and ctx.prev_dist is not None and ctx.dist is not None
                     else 0.0)

    # forward speed encouragement.
    p["fwd"] = FWD_SPEED_K * max(0.0, float(ctx.vel_body[0]))

    # smoothness / oscillation / rate penalties.
    da = np.asarray(ctx.action, float) - np.asarray(ctx.prev_action, float)
    a3 = np.asarray(ctx.action, float)[:3]
    p["smooth"] = -SMOOTH_K * float(np.dot(da, da))
    p["osc"] = -OSC_K * float(a3[0] * a3[0] + a3[1] * a3[1])   # roll/pitch target churn
    p["rate"] = -RATE_K * float(np.dot(a3, a3))                # overall angular effort

    # time penalty (finish quickly).
    p["time"] = -TIME_K

    # proximity warning (low-threat COLLISION) -- nudge away, don't end the run.
    p["soft_col"] = SOFT_COLLISION_PEN if ctx.soft_collision else 0.0

    # gate events.
    p["gate_bonus"] = GATE_PASS_BONUS if ctx.gate_passed else 0.0

    # terminal events.
    terminated = False
    truncated = False
    reason = ""
    p["terminal"] = 0.0
    if ctx.course_complete:
        p["terminal"] = COURSE_BONUS
        terminated = True
        reason = "course_complete"
    elif ctx.collision:
        p["terminal"] = COLLISION_PEN
        terminated = True
        reason = "collision"
    elif ctx.flipped:
        p["terminal"] = CRASH_PEN
        terminated = True
        reason = "flipped"
    elif ctx.out_of_bounds:
        p["terminal"] = OOB_PEN
        terminated = True
        reason = "out_of_bounds"
    elif ctx.timeout:
        p["terminal"] = TIMEOUT_PEN
        truncated = True
        reason = "timeout"

    return float(sum(p.values())), terminated, truncated, reason, p


if __name__ == "__main__":  # quick sanity
    base = dict(prev_dist=5.0, dist=4.0, visible=True, vel_body=np.array([3.0, 0, 0]),
                action=np.zeros(4), prev_action=np.zeros(4), gate_passed=False,
                course_complete=False, collision=False, out_of_bounds=False,
                flipped=False, timeout=False)
    r, t, tr, _, parts = step(StepCtx(**base))
    assert r > 0 and not t and not tr, (r, t, tr)          # closing distance -> +
    assert abs(sum(parts.values()) - r) < 1e-9             # parts sum to total
    r2, _, _, _, _ = step(StepCtx(**{**base, "prev_dist": 4.0, "dist": 5.0}))
    assert r2 < r                                          # opening distance -> less
    rc, tc, _, why, _ = step(StepCtx(**{**base, "collision": True}))
    assert tc and why == "collision" and rc < -10          # collision terminates, big -
    rs, ts, _, _, ps = step(StepCtx(**{**base, "soft_collision": True}))
    assert not ts and ps["soft_col"] < 0                   # proximity -> penalty, NOT terminal
    print("[vq2.reward] selftest OK", round(r, 3), round(r2, 3), round(rc, 3))
