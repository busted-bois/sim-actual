"""Box-based visual-servo flight — fly THROUGH the gate the detector SEES.

Uses ONLY the YOLO detection's bounding-box CENTER (no PnP, no broadcast map):
keep the locked gate centred in the image and fly forward, holding heading through
detection dropouts and aiming at the opening centre. No keypoints needed — so it
works with the synthetic-trained detector (box mAP ~0.96) even before the corners
converge. Pure vision, no hardcoded coordinates.

    uv run -m rl.fly_servo                 # start the race, ENTER at countdown
    uv run -m rl.fly_servo --no-view       # headless
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from rl.fly2 import (
    HOVER_T,
    K_ATT,
    KD_Z,
    KP_Z,
    RATE_CLIP,
    SIGN_PITCH,
    SIGN_ROLL,
    SIGN_YAW,
    YAW_CLIP,
    rpy,
)
from rl.fly_vision import VisionThread
from rl.gate_tracker import GateTracker
from rl.sim_interface import SimInterface

HZ = 150.0
FRESH_S = 0.4  # a detection newer than this = "fresh" -> CHASE
COMMIT_S = 1.2  # keep flying toward a just-lost gate this long -> COMMIT
YAW_GAIN = 0.6  # yaw rate per unit image x-offset of the gate
ALIGN_NX = 0.35  # |nx| below this = aligned enough for full forward lean
FWD_LEAN = 0.11  # forward pitch (rad) when aligned / committing
APPROACH_LEAN = 0.05  # gentler forward pitch while still turning toward the gate
PASS_RFRAC = 0.18  # gate bbox fills this frac of the frame -> we're on it (passed)
SEARCH_YAW = 0.4  # sweep rate when re-acquiring a lost gate
SEARCH_CREEP = 0.03  # forward lean while searching
Z_AIM = -1.3  # NED altitude target vs spawn (negative = higher); aims at the
# opening centre instead of clipping the bottom bar.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--no-view", dest="view", action="store_false")
    ap.add_argument("--no-wait", dest="wait", action="store_false")
    args = ap.parse_args()

    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[servo] no telemetry — is the sim running?", flush=True)
        os._exit(1)
    tracker = GateTracker()  # fed by VisionThread; unused here but keeps it happy
    vision = VisionThread(sim, tracker, view=args.view)
    vision.start()

    if args.wait:
        try:
            input("[servo] READY -- press ENTER the moment the countdown hits 0...")
        except EOFError:
            pass
    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    hold_z = s0.pos_ned[2]
    print(f"[servo] flying (box visual servo); hold_z={hold_z:.1f}", flush=True)

    nx_s = 0.0
    last_seen = 0.0
    ever_seen = False
    was_close = False
    n_passed = 0
    sweep_dir = 1.0
    last_log = 0.0
    t0 = time.time()
    reason = "timeout"
    while time.time() - t0 < args.seconds:
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / HZ)
            continue
        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        roll, pitch, _ = rpy(snap.quat)
        z, vz = p[2], v[2]
        now = time.time()

        best = vision.best
        fresh = (
            best is not None
            and vision.best_t > 0
            and (time.monotonic() - vision.best_t < FRESH_S)
        )
        if fresh:
            nx, _, r_frac = best
            nx_s = 0.6 * nx_s + 0.4 * nx  # smooth image-jitter on the lock
            last_seen = now
            ever_seen = True
            if r_frac >= PASS_RFRAC:  # gate fills the frame -> we're flying through it
                was_close = True

        tgt_roll = 0.0
        z_target = hold_z + Z_AIM  # aim at the opening centre, never dive
        if now - last_seen < COMMIT_S:
            if fresh:
                # CHASE: steer toward the gate (only when we actually see it).
                state = "CHASE"
                yaw_cmd = float(
                    np.clip(SIGN_YAW * YAW_GAIN * nx_s, -YAW_CLIP, YAW_CLIP)
                )
                tgt_pitch = -(FWD_LEAN if abs(nx_s) < ALIGN_NX else APPROACH_LEAN)
            else:
                # COMMIT: lost the detection briefly — fly STRAIGHT forward (hold
                # heading) instead of yawing toward a stale offset.
                state = "COMMIT"
                yaw_cmd = 0.0
                tgt_pitch = -FWD_LEAN
        else:
            # Lost it for a while. If we'd just filled the frame, we passed it.
            if was_close:
                n_passed += 1
                was_close = False
                print(f"[servo] gate {n_passed}/{args.gates} PASSED", flush=True)
                if n_passed >= args.gates:
                    reason = "COURSE COMPLETE"
                    break
            state = "SEARCH"
            if not ever_seen:
                # Start of run: gate 0 is straight ahead — hold heading and creep.
                yaw_cmd = 0.0
            else:
                if (now % 4.0) < 0.02:
                    sweep_dir = -sweep_dir
                yaw_cmd = float(
                    np.clip(SIGN_YAW * SEARCH_YAW * sweep_dir, -YAW_CLIP, YAW_CLIP)
                )
            tgt_pitch = -SEARCH_CREEP

        roll_cmd = float(
            np.clip(SIGN_ROLL * K_ATT * (tgt_roll - roll), -RATE_CLIP, RATE_CLIP)
        )
        pitch_cmd = float(
            np.clip(SIGN_PITCH * K_ATT * (tgt_pitch - pitch), -RATE_CLIP, RATE_CLIP)
        )
        thrust = float(np.clip(HOVER_T + KP_Z * (z - z_target) + KD_Z * vz, 0.18, 0.5))
        sim.send_attitude_rates(roll_cmd, pitch_cmd, yaw_cmd, thrust)

        if args.view and vision.annotated is not None:
            cv2.imshow("servo", vision.annotated)
            cv2.waitKey(1)

        if now - last_log > 0.5:
            last_log = now
            r = best[2] if best else 0.0
            print(
                f"[servo] [{now - t0:5.1f}s] g{n_passed} {state} det={vision.n_raw} "
                f"nx_s={nx_s:+.2f} r={r:.2f} yolo={vision.infer_ms:.0f}ms "
                f"z={z:+.1f} thr={thrust:.2f}",
                flush=True,
            )
        time.sleep(1 / HZ)

    sim.send_attitude_rates(0, 0, 0, HOVER_T)
    print(f"[servo] done: {reason}; passed {n_passed}/{args.gates}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
