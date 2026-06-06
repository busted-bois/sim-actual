"""Validate tracking snapshot from synthetic MAVLink state."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.tracking.local_tracker import LocalTracker


def main() -> int:
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
    tracker.tick(
        {
            **data,
            "highres_imu": {
                **data["highres_imu"],
                "time_boot_us": 20_000,
            },
            "camera": {"sim_time_ns": 20_000_000},
        }
    )

    snapshot = data.get("tracking_snapshot")
    if snapshot is None:
        print("[tracking-smoke] FAIL: no tracking_snapshot")
        return 1

    print(
        f"[tracking-smoke] OK status={snapshot.status} "
        f"pos=({snapshot.x:.2f},{snapshot.y:.2f},{snapshot.z:.2f}) "
        f"yaw={snapshot.yaw:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
