"""Vision smoke test: synthetic detection + optional live UDP probe."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from simulator.vision_processing import detect_gate_target
from simulator.vision_rx import SIM_SERVER_UDP_PORT, VisionRX


def synthetic_smoke() -> bool:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(image, (420, 180), 45, (255, 120, 40), thickness=10)
    result = detect_gate_target(image)
    ok = result["detected"] and result["nx"] > 0.0
    print(
        f"[synthetic] detected={result['detected']} "
        f"nx={result['nx']:.3f} ny={result['ny']:.3f} r_frac={result['r_frac']:.3f}"
    )
    return ok


def live_smoke(timeout_s: float) -> bool:
    shared_data: dict = {}
    rx = VisionRX(shared_data)
    deadline = time.time() + timeout_s
    print(
        f"[live] listening on UDP :{SIM_SERVER_UDP_PORT} for up to {timeout_s:.0f}s..."
    )

    saw_frame = False
    saw_detection = False
    while time.time() < deadline:
        camera = shared_data.get("camera")
        if camera and camera.get("frame_id") is not None:
            saw_frame = True
            gate = shared_data.get("gate_target", {})
            if gate.get("detected"):
                saw_detection = True
            if saw_frame:
                print(
                    f"[live] frame_id={camera['frame_id']} "
                    f"{camera['width']}x{camera['height']} "
                    f"gate_detected={gate.get('detected', False)} "
                    f"nx={gate.get('nx', 0.0):.3f}"
                )
                break
        time.sleep(0.05)

    rx.is_running = False
    rx.thread.join(timeout=1.0)

    if not saw_frame:
        print("[live] no frames received (is FlightSim running with a session active?)")
        return False
    print(f"[live] frames OK, gate_detected={saw_detection}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision pipeline smoke test")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also probe live UDP vision stream from FlightSim",
    )
    parser.add_argument("--timeout", type=float, default=8.0, help="Live probe seconds")
    args = parser.parse_args()

    ok = synthetic_smoke()
    if args.live:
        try:
            ok = live_smoke(args.timeout) and ok
        except OSError as exc:
            print(f"[live] failed to bind UDP port: {exc}")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
