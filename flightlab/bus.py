"""MAVLink bus: connect, heartbeat, telemetry, actuation at 90 Hz."""

from __future__ import annotations

import os
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from simulator.controller import (  # noqa: E402
    CONTROL_HZ,
    MAVLINK_CMD_SIM_RESET,
    _send_attitude_rates,
)
from simulator.mavlink_client import (  # noqa: E402
    GcsHeartbeat,
    request_flight_streams,
    send_gcs_heartbeat,
)
from simulator.mavlink_rx import MAVLinkRX  # noqa: E402
from simulator.preflight import vision_ready  # noqa: E402
from simulator.state_estimator import quat_from_rpy  # noqa: E402
from simulator.transforms import quat_to_yaw  # noqa: E402
from simulator.vision_rx import VisionRX  # noqa: E402

from flightlab.state import Cmd, State  # noqa: E402

CONTROL_DT = 1.0 / CONTROL_HZ
LISTEN_ADDR = "127.0.0.1"
LISTEN_PORT = 14550

# Sim-ground-truth pose for attitude tests (not EKF).
POSE_SOURCES_OK = frozenset({"odometry", "attitude"})
VQ2_FALLBACK_S = float(os.environ.get("VQ2_FALLBACK_S", "2.0"))


def _local_ned_from_data(
    data: dict,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """LOCAL_POSITION_NED dict or legacy pos_ned/vel_ned from mavlink_rx."""
    lpos = data.get("local_position_ned")
    if lpos is not None:
        return (
            (float(lpos["x"]), float(lpos["y"]), float(lpos["z"])),
            (
                float(lpos.get("vx", 0.0)),
                float(lpos.get("vy", 0.0)),
                float(lpos.get("vz", 0.0)),
            ),
        )
    if data.get("has_position") and data.get("pos_ned") is not None:
        pos = data["pos_ned"]
        vel = data.get("vel_ned") or (0.0, 0.0, 0.0)
        return (
            (float(pos[0]), float(pos[1]), float(pos[2])),
            (float(vel[0]), float(vel[1]), float(vel[2])),
        )
    return None


def _rpy_from_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    import math

    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = quat_to_yaw(w, x, y, z)
    return roll, pitch, yaw


def _rpy_from_imu(imu: dict, accel_sign: float = 1.0) -> tuple[float, float]:
    """Gravity tilt from accelerometer — matches StateEstimator boot init."""
    import math

    ax = accel_sign * imu["ax"]
    ay = accel_sign * imu["ay"]
    az = accel_sign * imu["az"]
    roll = math.atan2(-ay, -az)
    pitch = math.atan2(ax, math.hypot(ay, az))
    return roll, pitch


class Bus:
    def __init__(self) -> None:
        self.data: dict = {}
        self.system_boot_ms = int(time.time() * 1000)
        self._pose_mono: float | None = None
        self._last_stamp: object = None
        self._gcs_hb: GcsHeartbeat | None = None
        self._seen: dict[str, bool] = {
            "imu": False,
            "attitude": False,
            "odometry": False,
            "local_position": False,
            "estimator": False,
        }
        self.vq2_mode = False
        self.pose_mode = ""

        os.environ.setdefault("FLIGHTLAB_QUIET_VISION", "1")

        print(
            f"[bus] connecting udpin:{LISTEN_ADDR}:{LISTEN_PORT} ...",
            flush=True,
        )
        self.conn = mavutil.mavlink_connection(f"udpin:{LISTEN_ADDR}:{LISTEN_PORT}")
        self.conn.wait_heartbeat()
        send_gcs_heartbeat(self.conn)
        print(f"[bus] heartbeat from system {self.conn.target_system}", flush=True)
        print(
            "[bus] no ODOMETRY/ATTITUDE yet — will use EKF pose if blocked (VQ2 profile)",
            flush=True,
        )

        # Feed IMU into estimator for pose fallback when ODOMETRY is blocked.
        from simulator.state_estimator import StateEstimator

        self.estimator = StateEstimator()
        self.mavlink_rx = MAVLinkRX.create_mavlink_rx(
            self.conn, self.data, estimator=self.estimator
        )
        self.vision_rx = VisionRX(self.data)
        self._request_streams()
        self._gcs_hb = GcsHeartbeat(self.conn)
        self._gcs_hb.start()

    def _request_streams(self) -> None:
        request_flight_streams(self.conn)

    def _mark_seen(self) -> None:
        self._seen["imu"] = self.data.get("imu") is not None
        self._seen["attitude"] = self.data.get("attitude") is not None
        self._seen["odometry"] = self.data.get("odometry") is not None
        self._seen["local_position"] = self.data.get(
            "local_position_ned"
        ) is not None or bool(self.data.get("has_position"))
        self._seen["estimator"] = self.estimator.ready

    def has_pose(self) -> bool:
        self._mark_seen()
        if self.data.get("odometry") is not None:
            return True
        if self.data.get("attitude") is not None:
            return True
        if self.estimator.ready:
            return True
        return False

    def has_flight_pose(self) -> bool:
        self._mark_seen()
        if self.data.get("odometry") is not None:
            return True
        if self.data.get("attitude") is not None:
            return True
        return False

    @staticmethod
    def pose_ok(s: State | None) -> bool:
        return s is not None and s.pose_source in POSE_SOURCES_OK

    def pose_usable(self, s: State | None) -> bool:
        if s is None:
            return False
        if s.pose_source in POSE_SOURCES_OK:
            return True
        return self.vq2_mode and s.pose_source in ("ekf", "imu_tilt")

    def _race_started(self) -> bool:
        race = self.data.get("race_status") or {}
        return race.get("race_start_boot_time_ms", -1) >= 0

    def _try_vq2_fallback(self) -> bool:
        self._mark_seen()
        if not self._seen["estimator"]:
            return False
        if not vision_ready(self.data):
            return False
        if not self._race_started():
            return False
        self.vq2_mode = True
        self.pose_mode = "vq2"
        print(
            "[bus] VQ2 telemetry block — using EKF pose "
            "(IMU+vision; no ODOMETRY/ATTITUDE)",
            flush=True,
        )
        return True

    def wait_for_odometry(self, timeout_s: float = 90.0) -> bool:
        mode = self.wait_for_flight_ready(timeout_s=timeout_s)
        return mode is not None

    def wait_for_flight_ready(self, timeout_s: float = 90.0) -> str | None:
        """Wait for ODOMETRY/ATTITUDE, or fall back to EKF in VQ2 sessions."""
        if not self.wait_for_link():
            print("[bus] no HIGHRES_IMU — is FlightSim running?", flush=True)
            return None

        print(
            "[bus] MAVLink link OK (IMU). Enter TRAINING/SUBMISSION session, click Race.",
            flush=True,
        )
        t0 = time.monotonic()
        last_log = 0.0
        last_arm = 0.0
        last_req = 0.0

        while time.monotonic() - t0 < timeout_s:
            self._mark_seen()
            if self.data.get("odometry") is not None:
                self.pose_mode = "odometry"
                print("[bus] pose source: ODOMETRY", flush=True)
                return "odometry"
            if self.data.get("attitude") is not None:
                self.pose_mode = "attitude"
                print("[bus] pose source: ATTITUDE", flush=True)
                return "attitude"

            now = time.monotonic()
            if now - t0 >= VQ2_FALLBACK_S and self._try_vq2_fallback():
                return "vq2"

            if now - last_arm >= 1.0:
                self.arm()
                last_arm = now
            if now - last_req >= 5.0:
                self._request_streams()
                last_req = now

            if now - last_log >= 2.0:
                race = self.data.get("race_status") or {}
                vision = vision_ready(self.data)
                hint = ""
                if not vision:
                    hint = " → enter TRAINING/SUBMISSION flight session"
                elif not self._race_started():
                    hint = " → click Race"
                elif now - t0 >= VQ2_FALLBACK_S:
                    hint = " → switching to EKF pose"
                print(
                    "[bus] waiting for pose... "
                    f"vision={vision} odo={self._seen['odometry']} "
                    f"att={self._seen['attitude']} ekf={self._seen['estimator']} "
                    f"race_start={race.get('race_start_boot_time_ms', -1)}"
                    f"{hint}",
                    flush=True,
                )
                last_log = now
            time.sleep(0.05)

        if self._try_vq2_fallback():
            return "vq2"
        return None

    def wait_for_link(self, timeout_s: float = 15.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self._mark_seen()
            if self._seen["imu"]:
                return True
            time.sleep(0.05)
        return False

    def pose_diagnostic(self) -> str:
        self._mark_seen()
        race = self.data.get("race_status") or {}
        lines = [
            f"  vision:      {'yes' if vision_ready(self.data) else 'NO'}",
            f"  HIGHRES_IMU: {'yes' if self._seen['imu'] else 'NO'}",
            f"  ATTITUDE:    {'yes' if self._seen['attitude'] else 'NO'}",
            f"  ODOMETRY:    {'yes' if self._seen['odometry'] else 'NO'}",
            f"  LOCAL_POS:   {'yes' if self._seen['local_position'] else 'NO'}",
            f"  race_start:  {race.get('race_start_boot_time_ms', 'n/a')}",
            f"  EKF ready:   {'yes' if self._seen['estimator'] else 'NO'}",
            f"  vq2_mode:    {self.vq2_mode}",
        ]
        return "\n".join(lines)

    def _read_state(self) -> State | None:
        odo = self.data.get("odometry")
        att = self.data.get("attitude")
        imu = self.data.get("imu")

        roll = pitch = yaw = 0.0
        quat = (1.0, 0.0, 0.0, 0.0)
        pos = (0.0, 0.0, 0.0)
        vel = (0.0, 0.0, 0.0)
        ang = (0.0, 0.0, 0.0)
        pose_source = "unknown"
        alt_trusted = False

        if odo is not None:
            pose_source = "odometry"
            alt_trusted = True
            quat = (odo["qw"], odo["qx"], odo["qy"], odo["qz"])
            roll, pitch, yaw = _rpy_from_quat(quat)
            pos = (odo["x"], odo["y"], odo["z"])
            vel = (odo["vx"], odo["vy"], odo["vz"])
            ang = (
                odo.get("roll_speed", 0.0),
                odo.get("pitch_speed", 0.0),
                odo.get("yaw_speed", 0.0),
            )
        elif att is not None:
            pose_source = "attitude"
            roll = att["roll"]
            pitch = att["pitch"]
            yaw = att["yaw"]
            q = quat_from_rpy(roll, pitch, yaw)
            quat = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            ang = (att["roll_speed"], att["pitch_speed"], att["yaw_speed"])
            local = _local_ned_from_data(self.data)
            if local is not None:
                pos, vel = local
                alt_trusted = True
        elif self.estimator.ready:
            pose_source = "ekf"
            p_e, v_e, q_e = self.estimator.pose()
            quat = tuple(float(x) for x in q_e)  # type: ignore[assignment]
            roll, pitch, yaw = _rpy_from_quat(quat)
            pos = tuple(float(x) for x in p_e)  # type: ignore[assignment]
            vel = tuple(float(x) for x in v_e)  # type: ignore[assignment]
            if imu is not None:
                ang = (imu["gx"], imu["gy"], imu["gz"])
            alt_trusted = self.vq2_mode
        else:
            return None

        if imu is not None:
            gyro = (imu["gx"], imu["gy"], imu["gz"])
            g_body = None
            ax, ay, az = imu["ax"], imu["ay"], imu["az"]
            gn = (ax * ax + ay * ay + az * az) ** 0.5
            if gn > 1e-3:
                g_body = (ax / gn, ay / gn, az / gn)
        else:
            gyro = ang
            g_body = None

        now = time.monotonic()
        if odo is not None:
            stamp: object = ("odo", id(odo))
        elif att is not None:
            stamp = ("att", self.data.get("att_time_ms"))
        else:
            stamp = ("ekf", round(now, 2))
        if stamp != self._last_stamp:
            self._last_stamp = stamp
            self._pose_mono = now
        pose_age = now - self._pose_mono if self._pose_mono is not None else 999.0

        return State(
            t_mono=now,
            armed=bool(self.data.get("armed", False)),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            roll_rate=ang[0],
            pitch_rate=ang[1],
            yaw_rate=ang[2],
            pos_ned=pos,
            vel_ned=vel,
            gyro=gyro,
            quat=quat,
            pose_age_s=pose_age,
            gravity_body=g_body,
            pose_source=pose_source,
            alt_trusted=alt_trusted,
        )

    def snapshot(self) -> State | None:
        return self._read_state()

    def send_cmd(self, cmd: Cmd) -> None:
        self.estimator.thrust_cmd = float(cmd.thrust)
        _send_attitude_rates(
            self.conn,
            self.system_boot_ms,
            roll_rate=cmd.roll_rate,
            pitch_rate=cmd.pitch_rate,
            yaw_rate=cmd.yaw_rate,
            thrust=cmd.thrust,
        )

    def arm(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def reset_sim(self) -> None:
        self.estimator.reset()
        self._pose_mono = None
        self._last_stamp = None
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def ensure_armed(self, timeout_s: float = 30.0) -> bool:
        t0 = time.monotonic()
        last_arm = 0.0
        while time.monotonic() - t0 < timeout_s:
            if self.data.get("armed"):
                return True
            if time.monotonic() - last_arm >= 1.0:
                self.arm()
                last_arm = time.monotonic()
            time.sleep(0.05)
        return False

    def close(self) -> None:
        if self._gcs_hb is not None:
            self._gcs_hb.stop()
        for rx in (self.mavlink_rx, self.vision_rx):
            thread = rx.get_thread_for_join()
            if thread is not None:
                thread.join(timeout=2.0)
