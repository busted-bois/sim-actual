#
# Manual keyboard flight client for the AI GP simulator.
# Fly the drone by hand to explore the map (make manual).
#

# Force MAVLink 2.0 BEFORE pymavlink is imported (any simulator import pulls it in).
# The sim's pose telemetry (ODOMETRY, id 331) is a MAVLink-2 message; as a MAVLink-1
# client we'd never receive it and couldn't request it.
import os

os.environ.setdefault("MAVLINK20", "1")

import time

from simulator.controller import CONTROL_HZ
from simulator.manual_control import ManualControl
from simulator.setup import setup_components

# Bind on 0.0.0.0 (all interfaces), matching the working branch — a loopback-only
# bind can stop our GCS heartbeat / setpoints from routing back to the sim, so it
# never registers us (no telemetry, commands ignored, drone runs its default launch).
SIM_SERVER_UDP_IP = "0.0.0.0"
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

# Fly via the focused Tk control window (owns the keyboard so the sim can't steal
# it) with a live HUD. Falls back to the global keyboard-hook loop if Tk is missing.
# The control loop starts streaming hover setpoints immediately after arming (no
# gap), so the sim can't run its default fast-launch; the HUD shows telemetry live.
print(
    "Manual flight ready — fly from the control window, close it to exit.", flush=True
)
try:
    from simulator.manual_ui import run_manual_ui

    run_manual_ui(controller, shared_data)
except ImportError as exc:
    print(f"[manual] Tk UI unavailable ({exc}); using keyboard-hook loop.", flush=True)
    manual = ManualControl(controller, shared_data)
    try:
        while True:
            manual.tick()
            time.sleep(1.0 / CONTROL_HZ)
    except KeyboardInterrupt:
        print("Exiting manual flight...", flush=True)

# exit — join whatever threads exist (guard against components without one)
for _component in (ts_loop, mavlink_rx, vision_rx):
    _thread = _component.get_thread_for_join()
    if _thread is not None:
        _thread.join(timeout=1.0)

print("Client exited!", flush=True)
