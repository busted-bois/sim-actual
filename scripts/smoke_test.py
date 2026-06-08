"""Synthetic smoke test for hackathon pilot stack (no FlightSim required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.config import DroneState, TrackGate
from simulator.pilot import ControlSetpoint
from simulator.gate_detector import detect_gate
from simulator.gate_estimator import GateEstimator
from simulator.pilot import Pilot
from simulator.state_machine import PilotState, transition
from simulator.transforms import body_to_ned_velocity, quat_to_yaw


def _synthetic_gate_frame() -> np.ndarray:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.rectangle(img, (220, 140), (420, 340), (15, 57, 243), thickness=-1)
    return img


def main() -> int:
    ok = True

    vn, ve = body_to_ned_velocity(2.0, 0.5, 0.0)
    if not (vn > 1.5 and abs(ve - 0.5) < 0.01):
        print("[fail] body_to_ned_velocity")
        ok = False
    else:
        print("[ok] transforms")

    yaw = quat_to_yaw(1.0, 0.0, 0.0, 0.0)
    if abs(yaw) > 1e-6:
        print("[fail] quat_to_yaw identity")
        ok = False

    det = detect_gate(_synthetic_gate_frame(), frame_id=1, sim_time_ns=1_000_000)
    if det is None or det.area_px < 500:
        print("[fail] gate_detector")
        ok = False
    else:
        print(f"[ok] gate_detector area={det.area_px:.0f}px")

    drone = DroneState((0.0, 0.0, -3.0), (0.0, 0.0, 0.0), 0.0, 0.0, 1000, True)
    next_state = transition(PilotState.TAKEOFF, None, None, drone, False, 11.0)
    if next_state != PilotState.CHASE:
        print("[fail] state_machine takeoff timeout")
        ok = False
    else:
        print("[ok] state_machine")

    gate = TrackGate(0, (10.0, 0.0, -3.0), (1.0, 0.0, 0.0, 0.0), 1.5, 1.5)
    est = GateEstimator().update(det, drone, [gate], 0)
    if est.confidence <= 0:
        print("[fail] gate_estimator")
        ok = False
    else:
        print(f"[ok] gate_estimator source={est.source}")

    shared: dict = {
        "pos_ned": (0.0, 0.0, -3.0),
        "vel_ned": (0.0, 0.0, 0.0),
        "yaw_rad": 0.0,
        "yaw_rate": 0.0,
        "att_time_ms": 1000,
        "has_position": True,
        "active_gate_index": 0,
        "gates": [gate],
    }
    pilot = Pilot(shared)
    pilot.on_frame(det)
    pilot._state = PilotState.CHASE
    cs = pilot.update(1.0 / 250.0)
    if cs.mode != "velocity" or cs.vel_ned is None:
        print("[fail] pilot chase control")
        ok = False
    else:
        print(f"[ok] pilot mode={cs.mode} vel={cs.vel_ned}")

    if not isinstance(cs, ControlSetpoint):
        print("[fail] control setpoint type")
        ok = False

    print("smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
