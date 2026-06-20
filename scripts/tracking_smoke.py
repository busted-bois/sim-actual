"""Validate tracking snapshot from synthetic or live MAVLink state."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.tracking.local_tracker import LocalTracker
from simulator.vision_rx import SIM_SERVER_UDP_PORT, VisionRX


def synthetic_smoke() -> bool:
    data = {
        "armed": True,
        "highres_imu": {
            "xacc": 0.0,
            "yacc": 0.0,
            "zacc": -9.81,
            "xgyro": 0.0,
            "ygyro": 0.0,
            "zgyro": 0.0,
            "time_boot_us": 10_000,
        },
        "local_position_ned": {
            "x": 1.0,
            "y": 0.5,
            "z": -2.0,
            "vx": 0.2,
            "vy": 0.0,
            "vz": 0.0,
        },
        "attitude": {"roll": 0.0, "pitch": 0.1, "yaw": 0.2},
        "camera": {"sim_time_ns": 10_000_000},
        "gate_target": {"detected": True, "nx": 0.1, "ny": 0.0, "r_frac": 0.05},
    }

    tracker = LocalTracker(log_csv=False)
    tracker.tick(data)
    for us in (20_000, 30_000):
        data["highres_imu"]["time_boot_us"] = us
        data["camera"]["sim_time_ns"] = us * 1000
        tracker.tick(data)

    snapshot = data.get("tracking_snapshot")
    if snapshot is None:
        print("[synthetic] FAIL: no tracking_snapshot")
        return False

    ok = snapshot.status == "tracking" and snapshot.healthy
    print(
        f"[synthetic] status={snapshot.status} healthy={snapshot.healthy} "
        f"imu_samples={snapshot.imu_samples} "
        f"pos=({snapshot.x:.2f},{snapshot.y:.2f},{snapshot.z:.2f})"
    )
    return ok


def live_smoke(timeout_s: float) -> bool:
    shared_data: dict = {}
    vision_rx = VisionRX(shared_data)
    deadline = time.time() + timeout_s
    imu_times: list[int] = []
    last_imu_us = None
    saw_frame = False
    saw_tracking = False
    saw_healthy = False

    print(
        f"[live] listening UDP :{SIM_SERVER_UDP_PORT} + shared_data for "
        f"{timeout_s:.0f}s (FlightSim session required)...",
        flush=True,
    )

    while time.time() < deadline:
        imu = shared_data.get("highres_imu")
        if imu is not None:
            time_us = int(imu.get("time_boot_us", 0))
            if time_us != last_imu_us:
                imu_times.append(time.time())
                last_imu_us = time_us
                if len(imu_times) > 240:
                    imu_times.pop(0)

        camera = shared_data.get("camera")
        if camera and camera.get("frame_id") is not None:
            saw_frame = True

        snapshot = shared_data.get("tracking_snapshot")
        if snapshot is not None:
            if snapshot.status == "tracking":
                saw_tracking = True
            if getattr(snapshot, "healthy", False):
                saw_healthy = True

        tracker = shared_data.get("_local_tracker")
        if tracker is not None and shared_data.get("armed"):
            tracker.tick(shared_data)

        if saw_frame and saw_tracking and len(imu_times) >= 10:
            break
        time.sleep(0.02)

    vision_rx.is_running = False
    vision_rx.get_thread_for_join().join(timeout=1.0)

    imu_hz = 0.0
    if len(imu_times) >= 2:
        span = imu_times[-1] - imu_times[0]
        if span > 0:
            imu_hz = (len(imu_times) - 1) / span

    print(
        f"[live] frames={saw_frame} tracking={saw_tracking} "
        f"healthy={saw_healthy} imu_hz~={imu_hz:.1f}",
        flush=True,
    )
    return saw_frame and saw_tracking


def main() -> int:
    parser = argparse.ArgumentParser(description="Tracking pipeline smoke test")
    parser.add_argument(
        "--live", action="store_true", help="Probe live FlightSim feeds"
    )
    parser.add_argument(
        "--timeout", type=float, default=12.0, help="Live probe seconds"
    )
    args = parser.parse_args()

    ok = synthetic_smoke()
    if args.live:
        try:
            ok = live_smoke(args.timeout) and ok
        except OSError as exc:
            print(f"[live] FAIL: {exc}", flush=True)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
