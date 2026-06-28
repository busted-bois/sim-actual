#
# Sample Python client for the AI GP controller
#

import sys
import time

from simulator.auto_flight import auto_flight_enabled, run_auto_flight_loop
from simulator.preflight import wait_for_race_go, wait_for_track
from simulator.setup import setup_components

SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

system_boot_ms = int(time.time() * 1000)
shared_data = {}

try:
    components = setup_components(
        shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
    )
except TimeoutError as exc:
    print(f"ERROR: {exc}", flush=True)
    sys.exit(1)
controller = components["controller"]
pilot = controller.pilot

print("Waiting for telemetry...", flush=True)
t0 = time.time()
while shared_data.get("odometry") is None and time.time() - t0 < 30:
    time.sleep(0.05)
if shared_data.get("odometry") is None:
    print("WARNING: no odometry after 30s — continuing anyway", flush=True)

if auto_flight_enabled():
    outcome, was_flying = run_auto_flight_loop(controller, pilot, shared_data)
    if outcome == "success":
        sys.exit(0)
    if outcome == "preflight_fail":
        sys.exit(1)
    print("[AUTO] cancelled — resuming normal sim mode", flush=True)
    if was_flying:
        while True:
            controller.update()

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

if hasattr(pilot, "on_attempt_start"):
    pilot.on_attempt_start()

print("Starting control loop...", flush=True)
while True:
    controller.update()
