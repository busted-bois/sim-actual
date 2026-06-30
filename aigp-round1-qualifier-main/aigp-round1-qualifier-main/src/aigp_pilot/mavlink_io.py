from __future__ import annotations

import math
import threading
import time

from pymavlink import mavutil

TIMESYNC_HZ = 10
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def quat_to_rpy(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class MavlinkSession:
    """MAVLink link setup + arming (PyAIPilotExample-v2)."""

    def __init__(self, ip: str = "0.0.0.0", port: int = 14550):
        self.system_boot_ms = int(time.time() * 1000)
        self.conn = mavutil.mavlink_connection(f"udpin:{ip}:{port}")
        print("Waiting for heartbeat...", flush=True)
        self.conn.wait_heartbeat()
        print(f"Connected to system {self.conn.target_system}.", flush=True)
        self.armed = False
        self._stop = threading.Event()
        threading.Thread(target=self._timesync_loop, daemon=True).start()

    def _timesync_loop(self) -> None:
        while not self._stop.is_set():
            self.conn.mav.timesync_send(int(time.time_ns()), 0)
            self._stop.wait(1.0 / TIMESYNC_HZ)

    def on_heartbeat(self, msg) -> None:
        self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def arm(self) -> None:
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def wait_armed(self, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg = self.conn.recv_match(type="HEARTBEAT", blocking=False)
            if msg is not None:
                self.on_heartbeat(msg)
            if self.armed:
                return True
            time.sleep(0.01)
        return self.armed

    def send_attitude_rates(
        self,
        roll_rate: float,
        pitch_rate: float,
        yaw_rate: float,
        thrust: float,
    ) -> None:
        now_ms = int(time.time() * 1000)
        self.conn.mav.set_attitude_target_send(
            now_ms - self.system_boot_ms,
            self.conn.target_system,
            self.conn.target_component,
            RATES_ATTITUDE_MASK,
            [1.0, 0.0, 0.0, 0.0],
            float(roll_rate),
            float(pitch_rate),
            float(yaw_rate),
            float(thrust),
        )

    def close(self) -> None:
        self._stop.set()
