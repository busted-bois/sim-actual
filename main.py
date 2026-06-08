#
# Sample Python client for the AI GP controller
#

import argparse
import sys
import time

from simulator.preflight import wait_for_race_go, wait_for_track
from simulator.setup import setup_components

SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550


def _parse_args():
    parser = argparse.ArgumentParser(description="AI GP autonomous drone pilot")
    parser.add_argument(
        "--collision-reset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "After collision hold, call reset_sim on high-threat hits "
            "(default: off, or set AUTO_RESET_ON_COLLISION=1)"
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    system_boot_ms = int(time.time() * 1000)
    shared_data = {}

    components = setup_components(
        shared_data,
        system_boot_ms,
        SIM_SERVER_UDP_IP,
        SIM_SERVER_UDP_PORT,
        auto_reset_on_collision=args.collision_reset,
    )
    controller = components["controller"]
    ts_loop = components["ts_loop"]
    mavlink_rx = components["mavlink_rx"]
    vision_rx = components["vision_rx"]
    local_tracker = shared_data.get("_local_tracker")

    if controller.pilot.auto_reset_on_collision:
        print("Collision auto-reset: ON", flush=True)
    else:
        print("Collision auto-reset: OFF", flush=True)

    if not wait_for_track(shared_data):
        sys.exit(1)

    race = shared_data.get("race_status") or {}
    armed_sim_boot_ms = race.get("sim_boot_time_ms", 0)

    print("Arming drone...", flush=True)
    controller.arm()

    if not wait_for_race_go(shared_data, armed_sim_boot_ms=armed_sim_boot_ms):
        sys.exit(1)

    print("Starting control loop...", flush=True)
    try:
        while True:
            controller.update()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)
    finally:
        controller.disarm()
        if local_tracker is not None:
            local_tracker.flush_log()
        ts_loop.get_thread_for_join().join(timeout=1.0)
        mavlink_rx.get_thread_for_join().join(timeout=1.0)
        vision_rx.get_thread_for_join().join(timeout=1.0)

    print("Client exited!", flush=True)


if __name__ == "__main__":
    main()
