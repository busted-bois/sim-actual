#
# Sample Python client for the AI GP controller
#

import sys
import time

from simulator import display
from simulator.auto_flight import auto_flight_enabled, run_auto_flight_loop
from simulator.preflight import wait_for_connect, wait_for_race_go, wait_for_track
from simulator.setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {}

# setup components
try:
    components = setup_components(
        shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    )
except TimeoutError as exc:
    print(f"ERROR: {exc}", flush=True)
    sys.exit(1)
controller = components["controller"]
pilot = controller.pilot
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]

if auto_flight_enabled():
    print("VQ2 mode — skipping odometry wait", flush=True)
    if not wait_for_connect(shared_data, timeout_s=5.0):
        print("WARNING: no IMU/vision after 5s — continuing anyway", flush=True)
else:
    print("Waiting for telemetry...", flush=True)
    t0 = time.time()
    while shared_data.get("odometry") is None and time.time() - t0 < 30:
        time.sleep(0.05)
    if shared_data.get("odometry") is None:
        print("WARNING: no odometry after 30s — continuing anyway", flush=True)

if auto_flight_enabled():
    outcome, was_flying = run_auto_flight_loop(controller, pilot, shared_data)
    if outcome == "preflight_fail":
        sys.exit(1)
    if was_flying:
        try:
            while True:
                controller.update()
        except KeyboardInterrupt:
            print("Exiting.", flush=True)
            sys.exit(0)

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
display.start()  # live vision window (what the drone's camera sees)
t0 = time.monotonic()
last_shown_tag = None
try:
    while True:
        controller.update()
        # Pump the vision window whenever a new (detected) frame is ready.
        img, tag = display.pick(shared_data)
        if tag is not None and tag != last_shown_tag:
            last_shown_tag = tag
            display.tick(img, time.monotonic() - t0)
except KeyboardInterrupt:
    print("Exiting.", flush=True)
finally:
    display.close()
    ts_loop.get_thread_for_join().join(timeout=1.0)
    mavlink_rx.get_thread_for_join().join(timeout=1.0)
    vision_rx.get_thread_for_join().join(timeout=1.0)
