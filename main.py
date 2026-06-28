#
# Sample Python client for the AI GP controller
#

import sys
import time

from simulator import display
from simulator.preflight import wait_for_race_go, wait_for_track
from simulator.setup import setup_components

SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

system_boot_ms = int(time.time() * 1000)
shared_data = {}

components = setup_components(
    shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
)
controller = components["controller"]
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]

print("Waiting for telemetry...", flush=True)
t0 = time.time()
while shared_data.get("odometry") is None and time.time() - t0 < 30:
    time.sleep(0.05)
if shared_data.get("odometry") is None:
    print("WARNING: no odometry after 30s — continuing anyway", flush=True)

if not wait_for_track(shared_data):
    print("ERROR: start a race in the sim to load track gates", flush=True)
    sys.exit(1)

race = shared_data.get("race_status") or {}
armed_sim_boot_ms = race.get("sim_boot_time_ms", 0)

print("Arming drone...", flush=True)
controller.arm()

if not wait_for_race_go(shared_data, armed_sim_boot_ms=armed_sim_boot_ms):
    print("ERROR: countdown never reached GO", flush=True)
    sys.exit(1)

print("Starting control loop...", flush=True)
display.start()
t0_display = time.monotonic()
last_shown_frame = -1
try:
    while True:
        controller.update()
        frame = shared_data.get("frame")
        if frame is not None and frame["frame_id"] != last_shown_frame:
            last_shown_frame = frame["frame_id"]
            display.tick(
                frame.get("annotated", frame["img"]),
                time.monotonic() - t0_display,
            )
finally:
    display.close()

ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
