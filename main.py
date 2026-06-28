#
# Sample Python client for the AI GP controller
#

import time

from simulator import display
from simulator.setup import setup_components

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

# time since sim started ms
system_boot_ms = int(time.time() * 1000)

# arbitrary shared data between the various components
shared_data = {}

# setup components
components = setup_components(
    shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT
)
controller = components["controller"]
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]

print("Arming drone...", flush=True)
controller.arm()
print("Starting control loop...", flush=True)
display.start()  # live vision window (what the drone's camera sees)
t0 = time.monotonic()
last_shown_frame = -1
is_running = True
try:
    while is_running:
        controller.update()
        # Pump the vision window on each new camera frame.
        frame = shared_data.get("frame")
        if frame is not None and frame["frame_id"] != last_shown_frame:
            last_shown_frame = frame["frame_id"]
            display.tick(frame.get("annotated", frame["img"]), time.monotonic() - t0)
finally:
    display.close()

# exit
ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

print("Client exited!", flush=True)
