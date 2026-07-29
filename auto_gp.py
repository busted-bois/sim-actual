#
# make control-flight — smooth GP vision flight (YOLO/PnP -> GPPilot guidance).
# Arm, then run the GP control loop until Ctrl+C. No overnight AUTO_FLIGHT.
#

import os
import sys
import time
import traceback

# Select GP pilot before Controller is constructed (via setup → main path).
os.environ["AUTO_PILOT"] = "gp"

from simulator import display
from simulator.setup import setup_components

# Live vision window (YOLO-annotated camera feed). GP_DISPLAY=0 to disable.
SHOW_VISION = os.environ.get("GP_DISPLAY", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

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
ts_loop = components["ts_loop"]
mavlink_rx = components["mavlink_rx"]
vision_rx = components["vision_rx"]

print("Arming drone...", flush=True)
controller.arm()

print("Starting control loop...", flush=True)
if SHOW_VISION:
    # imshow/waitKey must run on the thread that created the window — this
    # main loop, never the VisionRX receiver thread.
    display.start()
_last_tb = 0.0
_t0 = time.time()
_last_shown_tag = None
try:
    while True:
        try:
            controller.update()
        except Exception:
            # A transient tick error (bad packet, momentary NaN) must not
            # drop the aircraft mid-flight; log rate-limited and keep going.
            now = time.monotonic()
            if now - _last_tb >= 5.0:
                traceback.print_exc()
                _last_tb = now
            time.sleep(1.0 / 90.0)  # keep loop cadence if update() bailed early
        if SHOW_VISION:
            img, tag = display.pick(shared_data)
            if tag is not None and tag != _last_shown_tag:
                _last_shown_tag = tag
                display.tick(img, time.time() - _t0)
            else:
                display.tick(None, 0.0)  # keep the window pumping OS events
except KeyboardInterrupt:
    print("Exiting.", flush=True)
finally:
    if SHOW_VISION:
        display.close()
    pilot = getattr(controller, "pilot", None)
    if pilot is not None and hasattr(pilot, "shutdown"):
        pilot.shutdown()
    print("Shutting down background threads...", flush=True)
    ts_loop.get_thread_for_join().join(timeout=2.0)
    mavlink_rx.get_thread_for_join().join(timeout=2.0)
    vision_rx.get_thread_for_join().join(timeout=2.0)
    print("Client exited.", flush=True)
