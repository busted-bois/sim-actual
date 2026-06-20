import time

from pymavlink import mavutil

GCS_HEARTBEAT_INTERVAL_S = 0.5


def send_gcs_heartbeat(sim_conn):
    sim_conn.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )


class GCSHeartbeat:
    """Send client HEARTBEAT at >= 2 Hz (AGP spec §4.4)."""

    def __init__(self, sim_conn, interval_s=GCS_HEARTBEAT_INTERVAL_S):
        self.sim_conn = sim_conn
        self.interval_s = interval_s
        self._next_tx = 0.0

    def tick(self):
        now = time.monotonic()
        if now < self._next_tx:
            return
        send_gcs_heartbeat(self.sim_conn)
        self._next_tx = now + self.interval_s
