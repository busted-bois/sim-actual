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
    collision: bool
    out_of_bounds: bool
    flipped: bool
    timeout: bool


def step(ctx: StepCtx):
    """-> (reward: float, terminated: bool, truncated: bool, reason: str)."""
    r = 0.0

    # progress toward the visible gate (positive closing, negative opening).
    if ctx.visible and ctx.prev_dist is not None and ctx.dist is not None:
        r += PROGRESS_K * (ctx.prev_dist - ctx.dist)

    # forward speed encouragement.
    r += FWD_SPEED_K * max(0.0, float(ctx.vel_body[0]))

    # smoothness / oscillation / rate penalties.
    da = np.asarray(ctx.action, float) - np.asarray(ctx.prev_action, float)
    r -= SMOOTH_K * float(np.dot(da, da))
    a3 = np.asarray(ctx.action, float)[:3]
    r -= OSC_K * float(a3[0] * a3[0] + a3[1] * a3[1])   # roll/pitch target churn
    r -= RATE_K * float(np.dot(a3, a3))                 # overall angular effort

    # time penalty (finish quickly).
    r -= TIME_K

    # gate events.
    if ctx.gate_passed:
        r += GATE_PASS_BONUS

    # terminal events.
    terminated = False
    truncated = False
    reason = ""
    if ctx.course_complete:
        r += COURSE_BONUS
        terminated = True
        reason = "course_complete"
    elif ctx.collision:
        r += COLLISION_PEN
        terminated = True
        reason = "collision"
    elif ctx.flipped:
        r += CRASH_PEN
        terminated = True
        reason = "flipped"
    elif ctx.out_of_bounds:
        r += OOB_PEN
        terminated = True
        reason = "out_of_bounds"
    elif ctx.timeout:
        r += TIMEOUT_PEN
        truncated = True
        reason = "timeout"

    return float(r), terminated, truncated, reason


if __name__ == "__main__":  # quick sanity
    base = dict(prev_dist=5.0, dist=4.0, visible=True, vel_body=np.array([3.0, 0, 0]),
                action=np.zeros(4), prev_action=np.zeros(4), gate_passed=False,
                course_complete=False, collision=False, out_of_bounds=False,
                flipped=False, timeout=False)
    r, t, tr, _ = step(StepCtx(**base))
    assert r > 0 and not t and not tr, (r, t, tr)          # closing distance -> +
    r2, _, _, _ = step(StepCtx(**{**base, "prev_dist": 4.0, "dist": 5.0}))
    assert r2 < r                                          # opening distance -> less
    rc, tc, _, why = step(StepCtx(**{**base, "collision": True}))
    assert tc and why == "collision" and rc < -10          # collision terminates, big -
    print("[vq2.reward] selftest OK", round(r, 3), round(r2, 3), round(rc, 3))
