"""Diagnostic: listen for MAVLink from FlightSim and print message types.

Confirms what the sim actually streams to us (with a GCS heartbeat sent, bound to
0.0.0.0). If ODOMETRY / ATTITUDE appear here, telemetry works and manual flight
can close the loop. If only HEARTBEAT appears, the sim is not streaming to us.

Run:  make probe   (while a FlightSim session is active)
"""

import os

os.environ.setdefault("MAVLINK20", "1")  # MAVLink 2 so ODOMETRY (id 331) is visible

import sys
import time
from collections import Counter

from pymavlink import mavutil

from simulator.setup import HEARTBEAT_TIMEOUT_S, _send_gcs_heartbeat

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 14550
REPORT_INTERVAL_S = 2.0


def main() -> int:
    print(f"Probing udpin:{LISTEN_IP}:{LISTEN_PORT}", flush=True)
    print(
        "Start FlightSim, begin a flight session, then leave this running.", flush=True
    )
    conn = mavutil.mavlink_connection(f"udpin:{LISTEN_IP}:{LISTEN_PORT}")

    deadline = time.monotonic() + HEARTBEAT_TIMEOUT_S
    next_tx = 0.0
    next_report = time.monotonic() + REPORT_INTERVAL_S
    counts = Counter()
    connected = False

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_tx:
            _send_gcs_heartbeat(conn)
            next_tx = now + 1.0

        msg = conn.recv_match(blocking=False)
        if msg is not None:
            msg_type = msg.get_type()
            counts[msg_type] += 1
            if msg_type == "HEARTBEAT" and not connected:
                connected = True
                print(
                    f"Connected — system {conn.target_system}. Watching streams...",
                    flush=True,
                )

        if now >= next_report:
            next_report = now + REPORT_INTERVAL_S
            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            print(f"[{int(now)}] {summary or '(nothing yet)'}", flush=True)
            has_pose = any(
                k in counts for k in ("ODOMETRY", "ATTITUDE", "LOCAL_POSITION_NED")
            )
            if has_pose:
                print(
                    "  -> POSE TELEMETRY PRESENT (ODOMETRY/ATTITUDE) — closed-loop OK.",
                    flush=True,
                )

        if msg is None:
            time.sleep(0.005)

    print("Probe finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
