#
# make blueline / make sim — blue-line corridor flight. HSV dual-cyan primary
# + YOLO gate assist (SKIP_YOLO=1 disables the detector, BL_GATE_ASSIST=0
# keeps it running but ignores it). Arm, then run BlueLinePilot until Ctrl+C.
# Race-gated: the pilot holds zero thrust until a fresh race start appears
# (start/Restart Race in the sim), then flies immediately — no countdown
# hold. Per-attempt telemetry -> rl/data/bl_log_*.csv. Cruise speed via
# BL_CRUISE_KMH (default 12). No overnight AUTO_FLIGHT.
#

import os
import sys
import time
import traceback

os.environ["AUTO_PILOT"] = "blueline"

from simulator import display
from simulator.setup import setup_components

SHOW_VISION = os.environ.get("BL_DISPLAY", "1").strip().lower() not in (
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

print("Starting blue-line control loop...", flush=True)
if SHOW_VISION:
    display.start()
_last_tb = 0.0
_t0 = time.time()
_last_shown_tag = None
try:
    while True:
        try:
            controller.update()
        except Exception:
            now = time.monotonic()
            if now - _last_tb >= 5.0:
                traceback.print_exc()
                _last_tb = now
            time.sleep(1.0 / 90.0)
        if SHOW_VISION:
            img, tag = display.pick(shared_data)
            if tag is not None and tag != _last_shown_tag:
                _last_shown_tag = tag
                ga = shared_data.get("bl_gate_assist") or {}
                hud = None
                if ga.get("bearing_deg") is not None:
                    inf = ga.get("infer_ms") or 0.0
                    lag = ga.get("lag_fr")
                    hud = (
                        f"GATE {ga.get('mode')} brg={ga['bearing_deg']:+.0f}d "
                        f"rng={ga['range_m']:.1f}m yolo={float(inf):.0f}ms "
                        f"lag={'?' if lag is None else lag}f"
                    )
                elif ga:
                    hud = f"GATE none ({ga.get('mode')})"
                display.tick(img, time.time() - _t0, hud=hud)
            else:
                display.tick(None, 0.0)
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
