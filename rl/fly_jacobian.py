"""Tier-A Jacobian demo on the LIVE FlightSim (no ML weights).

Same stack as ``rl/demo_jacobian`` but on the real sim via MAVLink:
  gate pursuit  ->  v_des_ned + yaw_rate_des
  geo_control   ->  roll/pitch/yaw rates + thrust  (DEPLOY_GAINS, hover 0.27)
  SimInterface  ->  MAVLink

Gate index comes from sim ``active_gate_index`` (like fly2), not env physics.

    uv run -m rl.fly_jacobian --mode course
    uv run -m rl.fly_jacobian --mode course --seconds 95 --speed 3.5
    uv run -m rl.fly_jacobian --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from rl import spec
from rl.demo_jacobian import pursuit_velocity
from rl.geo_control import GeoGains, velocity_to_action
from rl.sim_interface import GATE_MAP_PATH, SimInterface

DEPLOY_GAINS = GeoGains()  # hover 0.27, live-sim measured defaults
HZ = 100.0


def _quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hover", "course"], default="course")
    ap.add_argument("--seconds", type=float, default=95.0)
    ap.add_argument("--speed", type=float, default=3.5, help="max pursuit speed m/s")
    ap.add_argument("--lookahead", type=float, default=4.0)
    ap.add_argument(
        "--zoff",
        type=float,
        default=-1.0,
        help="NED z offset added to gate aim (negative = fly higher)",
    )
    ap.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="skip ENTER sync at countdown",
    )
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    gate_map = json.load(open(GATE_MAP_PATH))["gates"]
    n = len(gate_map)
    sim = SimInterface()
    if not sim.wait_for_telemetry():
        print("[fly_j] no telemetry", flush=True)
        os._exit(1)
    if args.reset:
        sim.reset_sim()
        time.sleep(3)
    if args.wait and args.mode == "course":
        try:
            input("[fly_j] READY -- press ENTER at countdown 0...")
        except EOFError:
            pass

    sim.arm()
    time.sleep(0.2)
    s0 = sim.snapshot()
    hold_z = s0.pos_ned[2]
    print(
        f"[fly_j] mode={args.mode} gates={n} hover={DEPLOY_GAINS.hover_thrust} "
        f"max_speed={args.speed}",
        flush=True,
    )

    t0 = time.time()
    last_log = 0.0
    last_active = -1
    reason = "timeout"

    while time.time() - t0 < args.seconds:
        snap = sim.snapshot()
        if not snap.has_pose():
            time.sleep(1 / HZ)
            continue

        p = np.asarray(snap.pos_ned, float)
        v = np.asarray(snap.vel_ned, float)
        roll, pitch, yaw = _quat_to_rpy(snap.quat)
        z = p[2]

        if args.mode == "hover":
            v_des = np.zeros(3)
            yaw_rate_des = 0.0
        else:
            active = int(sim.data.get("active_gate_index", 0) or 0)
            if active != last_active:
                print(
                    f"[fly_j] [{time.time() - t0:4.1f}s] ACTIVE GATE -> {active}",
                    flush=True,
                )
                last_active = active
            if active >= n:
                reason = "COURSE COMPLETE"
                break

            g = np.array(gate_map[active]["pos"], float)
            g[2] += args.zoff
            next_g = (
                np.array(gate_map[active + 1]["pos"], float)
                if active + 1 < n
                else None
            )
            if next_g is not None:
                next_g = next_g.copy()
                next_g[2] += args.zoff

            v_des, yaw_rate_des = pursuit_velocity(
                p,
                v,
                snap.quat,
                g,
                lookahead=args.lookahead,
                max_speed=args.speed,
                next_gate_pos=next_g,
            )

        rates_thrust = velocity_to_action(
            v_des, yaw_rate_des, v, roll, pitch, yaw, DEPLOY_GAINS
        )
        sim.send_attitude_rates(*rates_thrust)

        gb_z = (spec.quat_to_R(snap.quat).T @ np.array([0.0, 0, 1.0]))[2]
        if gb_z < 0.0:
            reason = "ABORT flipped"
            break
        if z < hold_z - 30 or z > hold_z + 30:
            reason = "ABORT altitude"
            break

        now = time.time() - t0
        if now - last_log >= 0.5:
            spd = float(np.linalg.norm(v[:2]))
            print(
                f"[fly_j] [{now:4.1f}s] g={last_active if args.mode == 'course' else '-'} "
                f"rpy=({math.degrees(roll):+4.0f},{math.degrees(pitch):+4.0f},"
                f"{math.degrees(yaw):+4.0f}) z={z:+5.1f} spd={spd:.1f} "
                f"thr={rates_thrust[3]:.2f}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / HZ)

    sim.send_attitude_rates(0, 0, 0, DEPLOY_GAINS.hover_thrust)
    print(
        f"[fly_j] === DONE {reason} final={np.round(sim.snapshot().pos_ned, 1)} "
        f"active={sim.data.get('active_gate_index')} ===",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


def _selftest():
    """Offline: pursuit + geo_control chain produces finite live-sim commands."""
    p = np.array([0.0, 0.0, -5.0])
    v = np.zeros(3)
    q = np.array([1.0, 0.0, 0.0, 0.0])
    gate = np.array([10.0, 2.0, -4.0])
    v_des, yaw_rd = pursuit_velocity(p, v, q, gate, max_speed=4.0)
    roll, pitch, yaw = _quat_to_rpy(q)
    cmd = velocity_to_action(v_des, yaw_rd, v, roll, pitch, yaw, DEPLOY_GAINS)
    assert np.all(np.isfinite(cmd)), cmd
    assert cmd[3] > 0, "thrust must be positive"
    print(f"[fly_j] selftest OK cmd={np.round(cmd, 3)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    known, _ = ap.parse_known_args()
    if known.selftest:
        _selftest()
    else:
        main()
