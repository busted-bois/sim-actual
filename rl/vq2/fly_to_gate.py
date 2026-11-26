"""Minimal vision servo: fly straight at the gate using YOLO/PnP + the GP pilot's
exact steering law, routed through the RL env's OWN command path
(`SimInterface.send_attitude_quat_deg`) and the raw `_yolo_pose_estimate` the RL
policy sees. NO neural net, NO demos.

Purpose: prove the vision->control link works end-to-end through the RL plumbing.
  * If this threads gate 1, the infrastructure (vision read + quat wire + signs)
    is sound and the RL failure is purely the learned policy.
  * If it does NOT, there is a real frame/sign bug that breaks everyone.

Law (from simulator/gp_pilot.py, verified constants):
  bearing = atan2(by, bx)                       # by=gate RIGHT, bx=gate AHEAD
  desired_roll = clip(K_BEARING*bearing, ±14)
  roll_cmd  = (desired_roll - roll_deg) * KR    # KR = -1  (sim roll inverted)
  pitch_cmd = (DESIRED_PITCH_DEG - pitch_deg)*KP  # KP = +1, -2deg forward lean
  yaw_cmd   = bearing * KY                       # KY = -1  (sim yaw inverted)
  thrust    = HOVER - clip(bz,±3)*K_P_THRUST     # null vertical offset

    uv run -m rl.vq2.fly_to_gate            # start the race, then run (Ctrl+C stops)
    uv run -m rl.vq2.fly_to_gate --seconds 20
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from simulator.gp_estimation import GPEstimation
from simulator.gp_vision import _yolo_pose_estimate
from rl.sim_interface import SimInterface

# Verified GP-pilot constants (simulator/gp_pilot.py).
HOVER_THRUST = 0.264
DESIRED_PITCH_DEG = -2.0        # slight nose-down = forward cruise
K_BEARING = 2.5
MAX_BANK_DEG = 14.0
K_P_THRUST = 0.03
KP, KR, KY = 1.0, -1.0, -1.0    # sim: roll + yaw INVERTED, pitch not
PITCH_WIRE_MAX_DEG = 18.0
HZ = 50

# Near-gate commitment (mirrors the GP pilot's pass-through suppression): once the
# gate is within COMMIT_BX_M, STOP re-targeting and hold the approach trajectory
# for COMMIT_TICKS so the detector can't yank us onto a background gate. At
# ~2.8 m/s, ~45 ticks (0.9 s) carries us through a gate that was 2.5 m ahead.
COMMIT_BX_M = 2.5
COMMIT_TICKS = 45


def servo(seconds: float = 30.0):
    sim = SimInterface()
    sim.data["_quiet_vision"] = True
    est = GPEstimation(sim.data)
    est.start()
    print("[fly_to_gate] connected. Start the race. Servoing to the visible gate.",
          flush=True)
    t0 = time.time()
    last_arm = 0.0
    n = 0
    commit = 0                    # ticks left in "thread it, don't re-target" mode
    last_cmd = (0.0, DESIRED_PITCH_DEG, 0.0, HOVER_THRUST)
    last_bx = float("inf")        # last seen forward range (for commit-on-loss)
    passes = 0
    try:
        while time.time() - t0 < seconds:
            now = time.time()
            if not sim.data.get("armed", False) and now - last_arm > 1.0:
                sim.arm()
                last_arm = now

            ego = est.snapshot()
            roll_deg, pitch_deg = ego["att_deg"][0], ego["att_deg"][1]
            pose = _yolo_pose_estimate(sim.data)

            if commit > 0:
                # committed: hold the approach trajectory (level roll/yaw toward
                # current attitude, keep forward lean) so we thread straight through
                # instead of chasing the next gate the detector now reports.
                commit -= 1
                _, p0, _, t0c = last_cmd
                roll_cmd = (0.0 - roll_deg) * KR      # keep wings level through it
                pitch_cmd = (DESIRED_PITCH_DEG - pitch_deg) * KP
                yaw_cmd = 0.0
                thrust = t0c
                if commit == 0:
                    passes += 1
                    print(f"  [threaded gate ~#{passes}] resuming search", flush=True)
            elif pose is not None and math.isfinite(pose["body_x_m"]) and pose["body_x_m"] > 0.1:
                bx, by, bz = pose["body_x_m"], pose["body_y_m"], pose["body_z_m"]
                bearing = float(np.clip(math.degrees(math.atan2(by, bx)), -25.0, 25.0))
                desired_roll = float(np.clip(K_BEARING * bearing, -MAX_BANK_DEG, MAX_BANK_DEG))
                roll_cmd = (desired_roll - roll_deg) * KR
                pitch_cmd = float(np.clip((DESIRED_PITCH_DEG - pitch_deg) * KP,
                                          -PITCH_WIRE_MAX_DEG, PITCH_WIRE_MAX_DEG))
                yaw_cmd = bearing * KY
                thrust = HOVER_THRUST - float(np.clip(bz, -3.0, 3.0)) * K_P_THRUST
                last_cmd = (roll_cmd, pitch_cmd, yaw_cmd, thrust)
                last_bx = bx
                if bx < COMMIT_BX_M:
                    commit = COMMIT_TICKS      # close enough: commit to threading it
                    print(f"  >> committing to gate (fwd={bx:.1f}m right={by:+.1f}m)",
                          flush=True)
                elif n % 10 == 0:
                    print(f"  gate: fwd={bx:5.1f} right={by:+5.1f} down={bz:+5.1f}  "
                          f"bearing={bearing:+5.1f}  -> roll={roll_cmd:+5.1f} "
                          f"pitch={pitch_cmd:+5.1f} yaw={yaw_cmd:+5.1f} thr={thrust:.2f}",
                          flush=True)
            elif last_bx < 4.0:
                # gate vanished while CLOSE (it filled/left the frame) -> we're
                # threading it: commit forward instead of stalling into a search.
                commit = COMMIT_TICKS
                last_bx = float("inf")
                roll_cmd = (0.0 - roll_deg) * KR
                pitch_cmd = (DESIRED_PITCH_DEG - pitch_deg) * KP
                yaw_cmd = 0.0
                thrust = HOVER_THRUST
                print("  >> gate lost up close -> committing to thread it", flush=True)
            else:
                # no gate anywhere: level the wings, hold hover, creep forward.
                roll_cmd = (0.0 - roll_deg) * KR
                pitch_cmd = (DESIRED_PITCH_DEG - pitch_deg) * KP
                yaw_cmd = 0.0
                thrust = HOVER_THRUST
                last_bx = float("inf")
                if n % 25 == 0:
                    print("  (no gate in view — holding level, creeping forward)", flush=True)

            sim.send_attitude_quat_deg(roll_cmd, pitch_cmd, yaw_cmd, thrust)
            n += 1
            time.sleep(1.0 / HZ)
    except KeyboardInterrupt:
        print("\n[fly_to_gate] stopped.", flush=True)
    finally:
        est.stop()
        sim.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    servo(args.seconds)
