"""Listen for MAVLink from FlightSim and print the first messages."""

import sys
import time

from pymavlink import mavutil

from simulator.setup import (
    HEARTBEAT_TIMEOUT_S,
    _send_gcs_heartbeat,
)

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 14550


def main() -> int:
    print(f"Probing udpin:{LISTEN_IP}:{LISTEN_PORT}", flush=True)
    print(
        "Start FlightSim, begin a flight session, then leave this running.",
        flush=True,
    )
    conn = mavutil.mavlink_connection(f"udpin:{LISTEN_IP}:{LISTEN_PORT}")
    deadline = time.monotonic() + HEARTBEAT_TIMEOUT_S
    next_tx = 0.0
    seen = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_tx:
            _send_gcs_heartbeat(conn)
            next_tx = now + 1.0
        msg = conn.recv_match(blocking=False)
        if msg is None:
            time.sleep(0.02)
            continue
        seen += 1
        print(f"[{seen}] {msg.get_type()}", flush=True)
        if msg.get_type() == "HEARTBEAT" and seen >= 5:
            print("Heartbeat OK — safe to run `make sim`.", flush=True)
            return 0
    print(
        f"No MAVLink traffic in {HEARTBEAT_TIMEOUT_S}s. "
        "Confirm FlightSim session is active.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
