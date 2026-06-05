#
# Sample Python client for the AI GP controller
#

import sys
import time

from simulator.preflight import wait_for_ready
from simulator.setup import setup_components

SIM_SERVER_UDP_IP = "0.0.0.0"
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

print("Arming drone...", flush=True)
controller.arm()

if not wait_for_ready(shared_data):
    sys.exit(1)

print("Starting control loop...", flush=True)
try:
    while True:
        controller.update()
except KeyboardInterrupt:
    print("Shutting down...", flush=True)
finally:
    controller.disarm()
    ts_loop.get_thread_for_join().join(timeout=1.0)
    mavlink_rx.get_thread_for_join().join(timeout=1.0)
    vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
