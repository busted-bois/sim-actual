"""Pilot orchestrator — produces control setpoints per cycle.

Integrates gate estimator, expanding search, and state machine to drive the
drone through a sequence of gates. Called at ~250 Hz by the controller loop.
Reads shared_data (populated by telemetry / vision) and outputs a single
ControlSetpoint per cycle — either velocity or attitude, never both.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

from simulator.config import (
    DEADBAND_PX,
    DETECTION_AGE_OUT_MS,
    FORWARD_BASE_SPEED_MPS,
    FORWARD_GAIN_PER_AREA,
    LATERAL_KP,
    MAX_YAW_RATE,
    TAKEOFF_THRUST,
    YAW_KP,
    DroneState,
    GateDetection,
)
from simulator.gate_estimator import GateEstimator
from simulator.search import ExpandingSearch
from simulator.state_machine import PilotState, transition
from simulator.transforms import body_to_ned_velocity

_IMG_W = 640.0
_IMG_H = 480.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass
class ControlSetpoint:
    mode: str  # "velocity" | "attitude"
    vel_ned: tuple[float, float, float] | None
    yaw_rate: float | None
    thrust: float | None


class Pilot:
    """Gate-traversal pilot. Read-only consumer of shared_data."""

    def __init__(self, shared_data: dict):  # type: ignore[type-arg]
        self._shared_data = shared_data
        self._estimator = GateEstimator()
        self._search = ExpandingSearch()
        self._state = PilotState.TAKEOFF
        self._last_detection: GateDetection | None = None
        self._last_det_time_ns: int = 0
        self._last_gate_idx: int = -1
        self._elapsed: float = 0.0

    def on_frame(self, detection: GateDetection | None) -> None:
        self._last_detection = detection
        self._last_det_time_ns = detection.sim_time_ns if detection else 0
        if detection is not None:
            self._search.reset()

    def update(self, dt_s: float) -> ControlSetpoint:
        # 1. Read drone state from shared_data
        _d = self._shared_data
        pos_ned = _d.get("pos_ned", (0, 0, 0))  # type: ignore[assignment]
        vel_ned = _d.get("vel_ned", (0, 0, 0))  # type: ignore[assignment]
        yaw_rad = _d.get("yaw_rad", 0.0)  # type: ignore[assignment]
        yaw_rate = _d.get("yaw_rate", 0.0)  # type: ignore[assignment]
        att_time_ms = _d.get("att_time_ms", 0)  # type: ignore[assignment]
        has_position = _d.get("has_position", False)  # type: ignore[assignment]
        active_gate_idx = _d.get("active_gate_index", 0)  # type: ignore[assignment]
        gates = _d.get("gates", [])  # type: ignore[assignment]

        # 2. Build DroneState
        drone_state = DroneState(
            pos_ned=pos_ned,
            vel_ned=vel_ned,
            yaw_rad=yaw_rad,
            yaw_rate=yaw_rate,
            time_boot_ms=att_time_ms,
            has_position=has_position,
        )

        # 3. Frame-reuse check — age out stale detections
        effective_det: GateDetection | None = None
        if self._last_detection is not None:
            now_ns = int(_time.time() * 1e9)
            age_ms = (now_ns - self._last_det_time_ns) / 1e6
            effective_det = (
                self._last_detection if age_ms < DETECTION_AGE_OUT_MS else None
            )

        # 4. Gate estimate
        estimate = self._estimator.update(
            effective_det, drone_state, gates, active_gate_idx
        )

        # 5. Track active_gate_index changes
        idx_changed = (
            active_gate_idx != self._last_gate_idx and self._last_gate_idx >= 0
        )
        self._last_gate_idx = active_gate_idx

        # 6. State transition
        prev_state = self._state
        self._elapsed += dt_s
        new_state = transition(
            self._state,
            effective_det,
            estimate,
            drone_state,
            idx_changed,
            self._elapsed,
        )
        if new_state != prev_state:
            self._state = new_state
            self._elapsed = 0.0

        # 7. Produce ControlSetpoint
        return self._control_for_state(effective_det, estimate, drone_state, dt_s)

    # ------------------------------------------------------------------
    # Per-state control logic
    # ------------------------------------------------------------------

    def _control_for_state(
        self,
        detection: GateDetection | None,
        estimate,
        drone_state: DroneState,
        dt_s: float,
    ) -> ControlSetpoint:
        if self._state == PilotState.TAKEOFF:
            return ControlSetpoint(
                mode="attitude", vel_ned=None, yaw_rate=0.0, thrust=TAKEOFF_THRUST
            )

        if self._state == PilotState.CHASE:
            return self._chase_control(detection, estimate, drone_state)

        if self._state == PilotState.ADVANCE:
            return self._advance_control(detection, drone_state)

        if self._state == PilotState.SEARCH:
            return self._search_control(detection, drone_state, dt_s)

        # Fallback: hover
        return ControlSetpoint(
            mode="velocity", vel_ned=(0.0, 0.0, 0.0), yaw_rate=0.0, thrust=None
        )

    def _chase_control(
        self,
        detection: GateDetection | None,
        estimate,
        drone_state: DroneState,
    ) -> ControlSetpoint:
        # Gate lost in chase — notify search module
        if detection is None:
            self._search.on_gate_lost()
            return ControlSetpoint(
                mode="velocity", vel_ned=(0.0, 0.0, 0.0), yaw_rate=0.0, thrust=None
            )

        # Yaw: center the gate in the image
        offset = detection.centroid_x_px - _IMG_W / 2.0
        if abs(offset) < DEADBAND_PX:
            yaw_rate = 0.0
        else:
            yaw_rate = _clamp(YAW_KP * offset, -MAX_YAW_RATE, MAX_YAW_RATE)

        # Forward speed — ramp up as gate gets closer (area grows)
        area_norm = detection.area_px / (_IMG_W * _IMG_H)
        forward = FORWARD_BASE_SPEED_MPS * (
            1.0 + FORWARD_GAIN_PER_AREA * min(area_norm, 1.0)
        )

        # Lateral correction
        if estimate.lateral_offset_m is not None:
            lateral = LATERAL_KP * estimate.lateral_offset_m
        else:
            # Pixel-ratio mode: bearing_rad is normalized offset, NOT radians
            lateral = LATERAL_KP * estimate.bearing_rad * FORWARD_BASE_SPEED_MPS

        vn, ve = body_to_ned_velocity(forward, lateral, drone_state.yaw_rad)
        return ControlSetpoint(
            mode="velocity", vel_ned=(vn, ve, 0.0), yaw_rate=yaw_rate, thrust=None
        )

    def _advance_control(
        self,
        detection: GateDetection | None,
        drone_state: DroneState,
    ) -> ControlSetpoint:
        forward = FORWARD_BASE_SPEED_MPS * 1.5
        yaw_rate = 0.0
        if detection is not None:
            offset = detection.centroid_x_px - _IMG_W / 2.0
            if abs(offset) > DEADBAND_PX:
                yaw_rate = _clamp(YAW_KP * offset * 0.5, -MAX_YAW_RATE, MAX_YAW_RATE)
        vn, ve = body_to_ned_velocity(forward, 0.0, drone_state.yaw_rad)
        return ControlSetpoint(
            mode="velocity", vel_ned=(vn, ve, 0.0), yaw_rate=yaw_rate, thrust=None
        )

    def _search_control(
        self,
        detection: GateDetection | None,
        drone_state: DroneState,
        dt_s: float,
    ) -> ControlSetpoint:
        if detection is None:
            self._search.on_gate_lost()
        cmd = self._search.next_command(drone_state, dt_s)
        if cmd.phase == "SWEEP":
            return ControlSetpoint(
                mode="attitude", vel_ned=None, yaw_rate=cmd.yaw_rate_cmd, thrust=None
            )
        # EXPAND phase — velocity mode, no yaw
        vn, ve = body_to_ned_velocity(
            cmd.forward_vel_mps, cmd.lateral_vel_mps, drone_state.yaw_rad
        )
        return ControlSetpoint(
            mode="velocity", vel_ned=(vn, ve, 0.0), yaw_rate=0.0, thrust=None
        )
