"""Translational dynamics identification of the real sim — fit + persist.

Closed-loop controllers kept diverging, so measure the plant directly. The
attitude harness (`make attitude-harness`) measures hover/rate/latency but
*derives* thrust_accel = g/hover (an assumption, not a measurement) and never
measures aerodynamic drag — which is why the internal training plant (rl/env.py
DRAG=0.1) let the deployed policy build ~37 m/s and overshoot every gate.

This module measures the two missing translational parameters on the live sim
and writes them into flightlab/calibration.json (merged, preserving the
harness's keys), so rl/env.py trains on the real thrust curve and drag:

  * thrust_accel (m/s^2 per unit thrust) + hover_thrust — vertical thrust sweep,
    linear-fit a_z = k_T * thrust - g  →  thrust_accel = k_T, hover = g / k_T.
  * drag (1/s) + v_max (m/s)            — forward tilt-and-hold to terminal
    speed; at terminal a_horiz = 0 → thrust_accel * sin(theta) = drag * v, using
    the MEASURED pitch (odometry quaternion), robust to the rate-gain unknown.

Run with the sim in a TRAINING session (odometry velocity must be available):

    uv run -m rl.dynamics_id            # measure, fit, and write calibration
    uv run -m rl.dynamics_id --dry-run  # measure + fit, print, do NOT write
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from rl.calibration import CAL_PATH, load_calibration
from rl.sim_interface import SimInterface

HZ = 100.0
SAMPLE_HZ = 50.0
GRAVITY = 9.81


def _pitch(q) -> float:
    """Body pitch angle (rad) from quaternion (w,x,y,z)."""
    w, x, y, z = q
    return math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))


def _slope(ts: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs ts (i.e. d(ys)/dt)."""
    t = np.asarray(ts, float)
    y = np.asarray(ys, float)
    if len(t) < 2:
        return float("nan")
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    m, _b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def collect(sim, roll, pitch, yaw, thrust, secs, skip_s=0.0):
    """Hold a command for `secs`; return samples after `skip_s` as
    (t, vz, v_horiz, pitch_rad) lists on a SAMPLE_HZ clock."""
    t0 = time.time()
    nxt = 0.0
    ts, vz, vh, pit = [], [], [], []
    while time.time() - t0 < secs:
        sim.send_attitude_rates(roll, pitch, yaw, thrust)
        el = time.time() - t0
        if el >= nxt:
            snap = sim.snapshot()
            v = snap.vel_ned
            if el >= skip_s:
                ts.append(el)
                vz.append(float(v[2]))
                vh.append(float(math.hypot(v[0], v[1])))
                pit.append(_pitch(snap.quat))
            nxt += 1.0 / SAMPLE_HZ
        time.sleep(1.0 / HZ)
    return ts, vz, vh, pit


def measure_vertical(sim):
    """Vertical thrust sweep → (thrust_accel, hover_thrust, sweep_rows)."""
    # Levels bracketing the ~0.27 live hover; each measures a_z = d(vz)/dt.
    levels = [0.30, 0.24, 0.34, 0.22, 0.38]
    sweep = []
    for thr in levels:
        sim.send_attitude_rates(0, 0, 0, 0.27)  # settle near hover first
        time.sleep(0.4)
        ts, vz, _vh, _pit = collect(sim, 0, 0, 0, thr, secs=1.0, skip_s=0.25)
        # NED z is down-positive; vertical accel UP = -d(vz)/dt.
        a_up = -_slope(ts, vz)
        if math.isfinite(a_up):
            sweep.append((thr, a_up))
            print(f"[dyn] thrust={thr:.3f}  a_up={a_up:+.2f} m/s^2", flush=True)
    if len(sweep) < 3:
        return None, None, sweep
    thr = np.array([s[0] for s in sweep])
    a = np.array([s[1] for s in sweep])
    # a_up = k_T*thrust - g  → slope k_T = thrust_accel, hover where a_up = 0.
    A = np.vstack([thr, np.ones_like(thr)]).T
    k_t, b = np.linalg.lstsq(A, a, rcond=None)[0]
    hover = float(-b / k_t) if abs(k_t) > 1e-6 else float("nan")
    return float(k_t), hover, sweep


def measure_horizontal(sim, thrust_accel, hover):
    """Forward tilt-and-hold to terminal speed → (drag, v_max).

    Pitch-rate pulse tilts the drone; holding zero rate keeps the tilt (this
    plant is a near-integrator, no auto-level), so horizontal speed climbs to
    terminal. At terminal: thrust_accel*sin(|pitch|) = drag*v_horiz."""
    thr = float(np.clip(hover + 0.03, 0.2, 0.5)) if math.isfinite(hover) else 0.30
    # Establish a forward tilt with a short pitch-rate pulse (~1 s), then hold.
    collect(sim, 0, 0.5, 0, thr, secs=1.0)
    # Hold the tilt and let horizontal speed approach terminal.
    ts, _vz, vh, pit = collect(sim, 0, 0, 0, thr, secs=3.0, skip_s=1.5)
    if not vh:
        return None, None
    v_term = float(np.median(vh[-max(1, len(vh) // 3):]))  # last-third median
    theta = float(np.median([abs(p) for p in pit[-max(1, len(pit) // 3):]]))
    sin_t = math.sin(theta)
    if v_term < 0.5 or sin_t < 0.05 or not math.isfinite(thrust_accel):
        print(
            f"[dyn] horizontal fit unreliable (v_term={v_term:.1f} "
            f"pitch={math.degrees(theta):.0f}deg) — skipping drag/v_max",
            flush=True,
        )
        return None, None
    drag = thrust_accel * sin_t / v_term
    print(
        f"[dyn] terminal v={v_term:.1f} m/s at pitch={math.degrees(theta):.0f}deg "
        f"→ drag={drag:.3f} /s",
        flush=True,
    )
    return float(drag), v_term


def merge_write(updates: dict) -> None:
    """Load existing calibration.json, update measured keys, write back."""
    cal = load_calibration()
    cal.update({k: v for k, v in updates.items() if v is not None})
    cal["dynamics_id_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(CAL_PATH, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"[dyn] wrote {CAL_PATH}: {updates}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run", action="store_true", help="measure + fit but do not write"
    )
    args = ap.parse_args()

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[dyn] no telemetry — is the sim running in a TRAINING session?", flush=True)
        os._exit(1)
    sim.reset_sim()
    time.sleep(3)
    sim.arm()
    time.sleep(0.5)

    print("[dyn] === vertical thrust sweep (thrust_accel + hover) ===", flush=True)
    thrust_accel, hover, sweep = measure_vertical(sim)
    if thrust_accel is None:
        print("[dyn] vertical sweep failed (need >=3 clean levels)", flush=True)
        sim.send_attitude_rates(0, 0, 0, 0.0)
        os._exit(1)
    print(
        f"[dyn] FIT thrust_accel={thrust_accel:.2f} m/s^2/unit  hover={hover:.3f}  "
        f"(vs derived g/hover={GRAVITY / hover:.2f})",
        flush=True,
    )

    print("[dyn] === forward coast (drag + v_max) ===", flush=True)
    drag, v_max = measure_horizontal(sim, thrust_accel, hover)

    sim.send_attitude_rates(0, 0, 0, 0.0)  # cut

    updates = {
        "thrust_accel": round(thrust_accel, 4),
        "hover_thrust": round(hover, 4) if math.isfinite(hover) else None,
        "drag": round(drag, 4) if drag else None,
        "v_max": round(v_max, 3) if v_max else None,
    }
    print(f"[dyn] result: {updates}", flush=True)
    if args.dry_run:
        print("[dyn] --dry-run: not writing calibration.json", flush=True)
    else:
        merge_write(updates)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
