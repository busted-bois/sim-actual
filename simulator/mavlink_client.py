"""Shared MAVLink GCS registration + telemetry stream requests."""

from __future__ import annotations

import threading

from pymavlink import mavutil

HEARTBEAT_HZ = 1.0
IMU_INTERVAL_US = 10_000
ODO_INTERVAL_US = 20_000
ATT_INTERVAL_US = 20_000
LPOS_INTERVAL_US = 20_000


def send_gcs_heartbeat(conn) -> None:
    conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        0,
    )


def request_message_interval(conn, message_id: int, interval_us: int) -> None:
    conn.mav.command_long_send(
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def request_flight_streams(conn) -> None:
    """Best-effort request for pose + IMU streams (Training and VQ2)."""
    for msg_id, interval in (
        (mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, IMU_INTERVAL_US),
        (mavutil.mavlink.MAVLINK_MSG_ID_ODOMETRY, ODO_INTERVAL_US),
        (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, ATT_INTERVAL_US),
        (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, LPOS_INTERVAL_US),
    ):
        request_message_interval(conn, msg_id, interval)


class GcsHeartbeat:
    """1 Hz GCS heartbeat so FlightSim registers this client for setpoints."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        send_gcs_heartbeat(self._conn)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            send_gcs_heartbeat(self._conn)
            self._stop.wait(1.0 / HEARTBEAT_HZ)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
