"""Passive live vision viewer — camera + YOLO gate detection, no MAVLink.

Pops up the drone-camera window with the YOLO-pose overlay (boxes + corner
keypoints, drawn by GatePoseRunner) and records runs/vision.mp4. Reads ONLY the
camera stream (UDP 5600) -- no odometry/attitude/heartbeat -- so it works in the
VQ2 Qualification event block where pose telemetry is blocked (spec
VADR-TS-003 s9.3). It never arms or commands the drone; it just watches, so it's
safe to run during a live qualification race purely to eyeball detections.

    uv run -m simulator.vision_view        (or: make view)

Start the sim/race first (or after -- it waits for frames). Ctrl+C to quit.
"""

import os
import time

from simulator import display
from simulator.vision_rx import VisionRX


def main():
    data: dict = {}
    VisionRX(data)  # binds UDP 5600, starts the camera + YOLO threads
    print("[view] camera + YOLO up -- start the sim/race to see detections.", flush=True)

    display.start()
    t0 = time.monotonic()
    last_tag = None
    try:
        while True:
            img, tag = display.pick(data)
            if tag is not None and tag != last_tag:
                last_tag = tag
                display.tick(img, time.monotonic() - t0)
            else:
                display.tick(None, time.monotonic() - t0)  # pump waitKey, stay responsive
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n[view] exiting", flush=True)
    finally:
        display.close()
    os._exit(0)  # hard-exit past the non-daemon VisionRX receiver thread


if __name__ == "__main__":
    main()
