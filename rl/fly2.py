"""Fast raceline pilot -- ported from the 24s AI-GP round-1 qualifier run.

Replaces the old gate-by-gate "yaw to face the gate, then creep forward at
2.8 m/s" controller with the qualifier's approach:

  * a smooth Catmull-Rom raceline through the (hardcoded) gate map,
  * an accel-limited speed profile (lateral / longitudinal / braking + climb),
  * pure-pursuit + cross-track VELOCITY control at up to ~9-13 m/s, and
  * a FIXED yaw (we bank/strafe through gates instead of rotating to face each).

It's the identical sim, control channel (set_attitude_target rates+thrust) and
gate coordinates as the qualifier, so its proven gains/signs transfer directly.
The only adaptation is the live-sim boundary: we read state from
``rl.sim_interface`` instead of owning the raw pymavlink connection.

IMPORTANT: we read attitude from the raw ATTITUDE MAVLink euler
(``sim.data["attitude"]``) -- the SAME source the qualifier used -- NOT the
odometry quaternion. That is what makes the qualifier's roll/pitch/yaw rate
sign conventions valid here.

    uv run -m rl.fly2 --mode course      # fly the full course
    uv run -m rl.fly2 --mode hover --seconds 8
    uv run -m rl.fly2 --mode course --now  # skip fresh-start wait, go now
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

from rl.fastpath import control, course, raceconfig, raceline
from rl.sim_interface import SimInterface

HOVER_THRUST = 0.29


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["course", "hover"], default="course")
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument(
        "--now",
        action="store_true",
        help="skip the fresh-start detection and launch immediately",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="send a sim reset before launching",
    )
    args = ap.parse_args()

    cfg = raceconfig.MEASURED
    print(
        f">>> CONFIG {cfg.name}: v_max={cfg.v_max:.1f} vh_max={cfg.vh_max:.1f} "
        f"climb_max={cfg.climb_max:.1f}",
        flush=True,
    )

    sim = SimInterface(start_vision=False)  # hardcoded gate map -> no camera needed

    # ---- state accessors (MAVLinkRX thread keeps sim.data fresh) -------------
    def pos():
        p = sim.data.get("pos_ned")
        return None if p is None else np.array(p, dtype=float)

    def vel():
        v = sim.data.get("vel_ned")
        return None if v is None else np.array(v, dtype=float)

    def att():
        a = sim.data.get("attitude")
        return None if a is None else np.array([a["roll"], a["pitch"], a["yaw"]], dtype=float)

    def active():
        return int(sim.data.get("active_gate_index", 0) or 0)

    def finished():
        return sim.data.get("race_finish_time_ns", -1) >= 0

    def race_start_ms():
        rs = sim.data.get("race_status")
        return rs["race_start_boot_time_ms"] if rs else 1

    def got_race_status():
        return sim.data.get("race_status") is not None

    def send(rates, thrust: float) -> None:
        sim.send_attitude_rates(rates[0], rates[1], rates[2], thrust)

    def fly_velocity(vn_des, ve_des, target_z, vz_ff, yaw0, affn, affe) -> None:
        v = vel()
        a = att()
        p = pos()
        vn = clamp(vn_des, -cfg.vh_max, cfg.vh_max)
        ve = clamp(ve_des, -cfg.vh_max, cfg.vh_max)
        an = clamp(cfg.kph_vel * (vn - v[0]) + affn, -cfg.ah_max, cfg.ah_max)
        ae = clamp(cfg.kph_vel * (ve - v[1]) + affe, -cfg.ah_max, cfg.ah_max)
        roll_des, pitch_des = control.accel_to_tilt(an, ae, a[2], tilt_max=cfg.tilt_rad)
        rates = control.attitude_rate_command(
            a, [roll_des, pitch_des, yaw0], kp=cfg.kp_att, max_rate=cfg.max_rate
        )
        rates[0] = -rates[0]  # measured roll-rate sign convention (qualifier)
        az = control.vertical_accel(
            target_z, p[2], v[2], vz_ff=vz_ff, kpz_pos=cfg.kpz_pos, kpz_vel=cfg.kpz_vel
        )
        thrust = control.collective_thrust(az, a[0], a[1], hover=HOVER_THRUST)
        send(rates, thrust)

    # ---- wait for telemetry (pos + attitude; camera not needed) -------------
    print("Waiting for telemetry (pos + attitude)...", flush=True)
    t0 = time.monotonic()
    while pos() is None or att() is None or vel() is None:
        if time.monotonic() - t0 > 20.0:
            print("[fly] no telemetry within 20s -- is the sim running?", flush=True)
            sys.stdout.flush()
            os._exit(1)
        time.sleep(0.01)

    if args.reset:
        sim.reset_sim()
        time.sleep(3.0)

    sim.arm()
    print(">>> ARMED.", flush=True)

    if args.mode == "hover":
        yaw0 = float(att()[2])
        hold_z = float(pos()[2])
        print(f">>> HOVER hold_z={hold_z:.1f} yaw0={yaw0:+.2f}", flush=True)
        t_h = time.time()
        while time.time() - t_h < args.seconds:
            fly_velocity(0.0, 0.0, hold_z, 0.0, yaw0, 0.0, 0.0)
            time.sleep(1 / 50)
        send([0.0, 0.0, 0.0], 0.0)
        print(">>> hover done", flush=True)
        sys.stdout.flush()
        os._exit(0)

    # ---- start synchronization (faithful to the qualifier) ------------------
    if not args.now:
        print(">>> Restart the run at the start line, then start the countdown.", flush=True)
        while not (
            got_race_status()
            and race_start_ms() < 0
            and active() == 0
            and abs(pos()[2]) < 10.0
        ):
            send([0.0, 0.0, 0.0], 0.0)
            time.sleep(1 / 50)

        print(">>> Fresh start detected. Start the race.", flush=True)
        while race_start_ms() < 0:
            send([0.0, 0.0, 0.0], 0.0)
            time.sleep(1 / 50)

        countdown = time.time()
        while time.time() - countdown < 3.3:
            send([0.0, 0.0, 0.0], 0.0)
            time.sleep(1 / 50)

    print(">>> GO.", flush=True)
    yaw0 = float(att()[2])
    start_pos = pos().copy()
    pts, cum = course.build_path(start_pos)
    vprof = course.build_profile(pts, cum, cfg)
    total = cum[-1]
    last_log = 0.0
    race_start = time.time()
    reason = "timeout"

    # Per-gate closest-approach tracking (diagnostics for edge clips), measured
    # against the TRUE gate centers (not the tunable aim points).
    # Convention: vert dz = drone_z - gate_z; +dz = larger NED z = physically
    # LOWER (bottom edge). +lat = drone to the RIGHT of the gate center (aligned
    # to the pilot's view).
    true_gates = course.TRUE_GATES
    n_gates = len(true_gates)
    gate_min = [float("inf")] * n_gates
    gate_off: list = [None] * n_gates
    # PLANE crossing: in-plane offset at the moment pos.x passes each gate.x.
    # This is the real clip metric (closest-approach-to-center can mislead).
    gate_plane: list = [None] * n_gates
    prev_p = None
    last_active = -1

    def fmt_gate(gi: int) -> str:
        dz, lat, d3 = gate_off[gi]
        return (
            f"gate {gi}: miss={d3:.2f}m  "
            f"vert={abs(dz):.2f}m {'LOW' if dz > 0 else 'high'}  "
            f"lat={abs(lat):.2f}m {'RIGHT' if lat > 0 else 'left'}"
        )

    def fmt_plane(gi: int) -> str:
        if gate_plane[gi] is None:
            return f"gate {gi}: (never crossed plane)"
        vc, lc, ip = gate_plane[gi]
        return (
            f"gate {gi}: PLANE miss={ip:.2f}m  "
            f"vert={abs(vc):.2f}m {'LOW' if vc > 0 else 'high'}  "
            f"lat={abs(lc):.2f}m {'RIGHT' if lc > 0 else 'left'}"
        )

    while time.time() - race_start < args.seconds:
        p = pos()

        # Track closest approach to every TRUE gate center (6 gates -> cheap).
        for gi in range(n_gates):
            g = true_gates[gi]
            d3 = float(np.linalg.norm(p - g))
            if d3 < gate_min[gi]:
                gate_min[gi] = d3
                dn, de = float(p[0] - g[0]), float(p[1] - g[1])
                lat = dn * math.sin(yaw0) - de * math.cos(yaw0)  # +lat = RIGHT
                gate_off[gi] = (float(p[2] - g[2]), lat, d3)

        # Gate-PLANE crossing: detect pos.x straddling gate.x, interpolate the
        # in-plane (lateral + vertical) offset there -- what actually clips.
        if prev_p is not None:
            for gi in range(n_gates):
                gx = true_gates[gi][0]
                if (prev_p[0] - gx) * (p[0] - gx) <= 0 and prev_p[0] != p[0]:
                    frac = (prev_p[0] - gx) / (prev_p[0] - p[0])
                    yc = prev_p[1] + frac * (p[1] - prev_p[1])
                    zc = prev_p[2] + frac * (p[2] - prev_p[2])
                    g = true_gates[gi]
                    latc = -(yc - g[1]) * math.cos(yaw0)  # dn=0 at crossing
                    vertc = float(zc - g[2])
                    inplane = math.hypot(latc, vertc)
                    if gate_plane[gi] is None or inplane < gate_plane[gi][2]:
                        gate_plane[gi] = (vertc, latc, inplane)
        prev_p = p

        a = active()
        if a != last_active:
            if 0 <= last_active < n_gates and gate_off[last_active] is not None:
                print(f">>> passed {fmt_gate(last_active)}", flush=True)
            last_active = a

        s = raceline.project_arc(pts, cum, p)
        v_set = raceline.speed_at(cum, vprof, s)
        nearest = raceline.point_at_arc(pts, cum, s)
        ahead = raceline.point_at_arc(pts, cum, s + cfg.look_ahead)
        delta = ahead - nearest
        horizontal = delta[:2]
        horizontal_len = float(np.linalg.norm(horizontal))
        tangent = horizontal / horizontal_len if horizontal_len > 1e-6 else np.zeros(2)
        slope = delta[2] / horizontal_len if horizontal_len > 1e-6 else 0.0
        velocity = v_set * tangent + cfg.kph_ct * (nearest[:2] - p[:2])
        speed = float(np.linalg.norm(velocity))
        if speed > cfg.vh_max:
            velocity *= cfg.vh_max / speed
        curv = raceline.path_curvature_vector(pts, cum, s)
        aff = cfg.curv_ff * v_set * v_set * curv
        fly_velocity(
            velocity[0],
            velocity[1],
            nearest[2] - cfg.alt_bias,
            v_set * slope,
            yaw0,
            float(aff[0]),
            float(aff[1]),
        )

        if finished() or active() >= len(course.GATES) or s >= total - 1.0:
            reason = "COURSE COMPLETE" if (finished() or active() >= len(course.GATES)) else "path end"
            break

        now = time.time()
        if now - last_log > 0.75:
            xy_speed = float(np.linalg.norm(vel()[:2]))
            print(
                f"gate={active()} s={s:5.1f}/{total:.0f} speed={xy_speed:.1f} "
                f"t={now - race_start:4.1f}s",
                flush=True,
            )
            last_log = now
        time.sleep(1 / 50)

    elapsed = time.time() - race_start
    send([0.0, 0.0, 0.0], 0.0)
    print(
        f">>> result: {reason} gates={active()}/{len(course.GATES)} "
        f"time={elapsed:.1f}s final={np.round(pos(), 1)}",
        flush=True,
    )
    print(">>> per-gate PLANE crossing (the real clip metric):", flush=True)
    for gi in range(n_gates):
        print(f"    {fmt_plane(gi)}", flush=True)
    print(">>> per-gate closest approach (to center, over whole flight):", flush=True)
    for gi in range(n_gates):
        if gate_off[gi] is not None:
            print(f"    {fmt_gate(gi)}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
