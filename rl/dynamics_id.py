"""Open-loop dynamics characterization of the real sim.

Closed-loop controllers kept diverging, so measure the plant directly. Reset
to spawn, then apply known attitude-rate + thrust steps and log the attitude
(roll/pitch/yaw) and velocity response. Answers:
  * Does a zero rate-command HOLD attitude (pure integrator) or AUTO-LEVEL?
  * What is hover thrust (zero vertical accel)?
  * Sign + scale of pitch-rate -> pitch-angle.

    uv run -m rl.dynamics_id
"""

import os
import sys
import time


from rl.sim_interface import SimInterface
from simulator.transforms import quat_to_yaw

HZ = 100.0


def log_row(tag, t, snap):
    q = snap.quat
    # roll/pitch from quaternion
    import math

    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = quat_to_yaw(*q)
    p = snap.pos_ned
    v = snap.vel_ned
    print(
        f"[dyn] {tag} t={t:4.1f} rpy=({math.degrees(roll):+5.0f},"
        f"{math.degrees(pitch):+5.0f},{math.degrees(yaw):+5.0f})d "
        f"z={p[2]:+5.1f} v=({v[0]:+4.1f},{v[1]:+4.1f},{v[2]:+4.1f})",
        flush=True,
    )


def phase(sim, tag, roll, pitch, yaw, thrust, secs):
    t0 = time.time()
    nxt = 0.0
    while time.time() - t0 < secs:
        sim.send_attitude_rates(roll, pitch, yaw, thrust)
        el = time.time() - t0
        if el >= nxt:
            log_row(tag, el, sim.snapshot())
            nxt += 0.5
        time.sleep(1 / HZ)


def main():
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[dyn] no telemetry", flush=True)
        os._exit(1)
    sim.reset_sim()
    time.sleep(3)
    sim.arm()
    time.sleep(0.5)
    print(
        "[dyn] === A: zero rates, thrust=0.55 (hold vs auto-level? hover?) ===",
        flush=True,
    )
    phase(sim, "A", 0, 0, 0, 0.55, 3.0)
    print(
        "[dyn] === B: pitch_rate=+0.15, thrust=0.55 (rate->angle sign/scale) ===",
        flush=True,
    )
    phase(sim, "B", 0, 0.15, 0, 0.55, 2.5)
    print(
        "[dyn] === C: zero rates again (does pitch return to 0 = auto-level?) ===",
        flush=True,
    )
    phase(sim, "C", 0, 0, 0, 0.55, 2.5)
    print("[dyn] === D: yaw_rate=+0.5 (yaw sign/scale) ===", flush=True)
    phase(sim, "D", 0, 0, 0.5, 0.55, 2.0)
    sim.send_attitude_rates(0, 0, 0, 0.5)
    print("[dyn] done", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
