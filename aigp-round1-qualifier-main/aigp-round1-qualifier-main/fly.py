from __future__ import annotations

import math
import time

import numpy as np

from aigp_pilot import control, course, imu_nav, mavlink_io, parsers, raceconfig, raceline

HOVER_THRUST = 0.27
GATE_PASS_MARGIN = 4.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def main() -> None:
    cfg = raceconfig.MEASURED
    print(f">>> CONFIG {cfg.name}: v_max={cfg.v_max:.1f} climb_max={cfg.climb_max:.1f}")

    session = mavlink_io.MavlinkSession()
    nav = imu_nav.ImuNavigator()
    use_mavlink_pose = False
    pose_ready = False

    print("Arming drone...", flush=True)
    session.arm()
    if not session.wait_armed():
        session.close()
        raise RuntimeError("drone did not arm within timeout")
    print(">>> ARMED.", flush=True)

    state = {
        "pos": None,
        "vel": None,
        "att": None,
        "active": 0,
        "finished": False,
        "rstart": 1,
        "got_rs": False,
        "collisions": 0,
    }

    def sync_nav_state() -> None:
        nonlocal pose_ready
        if use_mavlink_pose:
            return
        state["att"] = nav.att.copy()
        pose_ready = nav.ready

    def pump() -> None:
        nonlocal use_mavlink_pose, pose_ready
        while True:
            msg = session.conn.recv_match(blocking=False)
            if msg is None:
                return
            typ = msg.get_type()
            if typ == "HEARTBEAT":
                session.on_heartbeat(msg)
            elif typ == "HIGHRES_IMU":
                nav.update(msg)
                sync_nav_state()
            elif typ == "LOCAL_POSITION_NED":
                use_mavlink_pose = True
                state["pos"] = np.array([msg.x, msg.y, msg.z], dtype=float)
                state["vel"] = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
                pose_ready = state["att"] is not None
            elif typ == "ODOMETRY":
                use_mavlink_pose = True
                state["pos"] = np.array([msg.x, msg.y, msg.z], dtype=float)
                state["vel"] = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
                qw, qx, qy, qz = msg.q[0], msg.q[1], msg.q[2], msg.q[3]
                state["att"] = np.array(mavlink_io.quat_to_rpy(qw, qx, qy, qz), dtype=float)
                pose_ready = True
            elif typ == "ATTITUDE":
                state["att"] = np.array([msg.roll, msg.pitch, msg.yaw], dtype=float)
                if use_mavlink_pose and state["pos"] is not None:
                    pose_ready = True
            elif typ == "COLLISION":
                state["collisions"] += 1
            elif typ == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if raw and raw[0] == parsers.RACE_STATUS_ID:
                    rs = parsers.parse_race_status(raw)
                    state["active"] = rs.active_gate_index
                    state["finished"] = rs.finished
                    state["rstart"] = rs.race_start_boot_time_ms
                    state["got_rs"] = True

    def send_rates(rates, thrust: float) -> None:
        session.send_attitude_rates(rates[0], rates[1], rates[2], thrust)

    def fly_velocity(vn_des: float, ve_des: float, target_z: float, vz_ff: float, yaw_des: float, affn: float, affe: float) -> None:
        vel = np.nan_to_num(np.asarray(state["vel"], dtype=float), nan=0.0)
        att = np.asarray(state["att"], dtype=float)
        vn = clamp(vn_des, -cfg.vh_max, cfg.vh_max)
        ve = clamp(ve_des, -cfg.vh_max, cfg.vh_max)
        an = clamp(cfg.kph_vel * (vn - vel[0]) + affn, -cfg.ah_max, cfg.ah_max)
        ae = clamp(cfg.kph_vel * (ve - vel[1]) + affe, -cfg.ah_max, cfg.ah_max)
        roll_des, pitch_des = control.accel_to_tilt(an, ae, att[2], tilt_max=cfg.tilt_rad)
        rates = control.attitude_rate_command(att, [roll_des, pitch_des, yaw_des], kp=cfg.kp_att, max_rate=cfg.max_rate)
        rates[0] = -rates[0]
        rates[2] = -rates[2]
        az = control.vertical_accel(target_z, state["pos"][2], vel[2], vz_ff=vz_ff, kpz_pos=cfg.kpz_pos, kpz_vel=cfg.kpz_vel)
        thrust = control.collective_thrust(az, att[0], att[1], hover=HOVER_THRUST)
        send_rates(rates, thrust)

    print("Waiting for IMU (VQ2: no ODOMETRY; start the race in the sim)...", flush=True)
    last_log = time.time()
    while not pose_ready:
        pump()
        session.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
        now = time.time()
        if now - last_log >= 2.0:
            print("  still waiting for HIGHRES_IMU...", flush=True)
            last_log = now
        time.sleep(1 / 50)

    pose_src = "MAVLink pose" if use_mavlink_pose else "IMU attitude + gate-synced path"
    print(f">>> Pose ready ({pose_src}). Restart at start line, then start countdown.", flush=True)

    while not (state["got_rs"] and state["rstart"] < 0 and state["active"] == 0 and (use_mavlink_pose or state["att"] is not None)):
        pump()
        send_rates([0.0, 0.0, 0.0], 0.0)
        time.sleep(1 / 50)

    print(">>> Fresh start detected. Start the race.", flush=True)
    while state["rstart"] < 0:
        pump()
        send_rates([0.0, 0.0, 0.0], 0.0)
        time.sleep(1 / 50)

    countdown = time.time()
    while time.time() - countdown < 3.3:
        pump()
        send_rates([0.0, 0.0, 0.0], 0.0)
        time.sleep(1 / 50)

    print(">>> GO.", flush=True)
    start_pos = course.SPAWN if not use_mavlink_pose else state["pos"].copy()
    pts, cum = course.build_path(start_pos)
    gate_arcs = course.gate_arcs(pts, cum)
    vprof = course.build_profile(pts, cum, cfg)
    total = cum[-1]
    last_log = 0.0
    race_start = time.time()
    last_tick = race_start
    arc_s = 0.0
    last_active = 0

    while time.time() - race_start < 150:
        pump()
        now = time.time()
        dt = max(1e-4, now - last_tick)
        last_tick = now

        active = int(state["active"])
        if active != last_active:
            print(f">>> gate advance {last_active} -> {active}", flush=True)
            if active > last_active and active > 0:
                arc_s = max(arc_s, gate_arcs[active - 1])
            last_active = active

        if use_mavlink_pose:
            pos = state["pos"]
            s = raceline.project_arc(pts, cum, pos)
            v_set = raceline.speed_at(cum, vprof, s)
            max_s = total
        else:
            v_set = raceline.speed_at(cum, vprof, arc_s)
            if active < len(course.GATES):
                max_s = gate_arcs[active] + GATE_PASS_MARGIN
            else:
                max_s = total
            arc_s = min(arc_s + v_set * dt, max_s, total)
            s = arc_s
            pos = raceline.point_at_arc(pts, cum, s)
            state["pos"] = pos
            tangent_pt = raceline.point_at_arc(pts, cum, min(s + 0.5, total))
            back_pt = raceline.point_at_arc(pts, cum, max(s - 0.5, 0.0))
            tangent = tangent_pt - back_pt
            t_len = float(np.linalg.norm(tangent))
            if t_len > 1e-6:
                state["vel"] = tangent * (v_set / t_len)
            else:
                state["vel"] = np.zeros(3, dtype=float)
        nearest = raceline.point_at_arc(pts, cum, s)
        ahead = raceline.point_at_arc(pts, cum, s + cfg.look_ahead)
        delta = ahead - nearest
        horizontal = delta[:2]
        horizontal_len = float(np.linalg.norm(horizontal))
        path_tangent = horizontal / horizontal_len if horizontal_len > 1e-6 else np.zeros(2)
        slope = delta[2] / horizontal_len if horizontal_len > 1e-6 else 0.0
        velocity = v_set * path_tangent + cfg.kph_ct * (nearest[:2] - pos[:2])
        speed = float(np.linalg.norm(velocity))
        if speed > cfg.vh_max:
            velocity *= cfg.vh_max / speed
        curv = raceline.path_curvature_vector(pts, cum, s)
        aff = cfg.curv_ff * v_set * v_set * curv
        track_tangent = raceline.point_at_arc(pts, cum, min(s + 1.0, total)) - raceline.point_at_arc(pts, cum, max(s - 1.0, 0.0))
        yaw_des = math.atan2(float(track_tangent[1]), float(track_tangent[0]))
        fly_velocity(velocity[0], velocity[1], nearest[2] - cfg.alt_bias, v_set * slope, yaw_des, float(aff[0]), float(aff[1]))

        if state["finished"] or state["active"] >= len(course.GATES) or s >= total - 1.0:
            break

        now = time.time()
        if now - last_log > 0.75:
            cmd_speed = float(np.linalg.norm(velocity[:2]))
            print(
                f"gate={active} s={s:5.1f}/{total:.0f} max_s={max_s:.0f} speed={cmd_speed:.1f}",
                flush=True,
            )
            last_log = now
        time.sleep(1 / 50)

    elapsed = time.time() - race_start
    session.close()
    send_rates([0.0, 0.0, 0.0], 0.0)
    print(f">>> result: gates={state['active']}/{len(course.GATES)} time={elapsed:.1f}s collisions={state['collisions']}", flush=True)


if __name__ == "__main__":
    main()
