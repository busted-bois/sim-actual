from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from simulator.config import (
    ALTITUDE_TARGET_M,
    LOST_FRAMES_THRESHOLD,
    PASS_AREA_FRAC,
    PASS_RANGE_M,
    TAKEOFF_TIMEOUT_S,
    GateDetection,
    DroneState,
)

if TYPE_CHECKING:
    from simulator.gate_estimator import GateEstimate


class PilotState(Enum):
    TAKEOFF = "TAKEOFF"
    CHASE = "CHASE"
    ADVANCE = "ADVANCE"
    SEARCH = "SEARCH"


def transition(
    current: PilotState,
    detection: GateDetection | None,
    estimate: GateEstimate | None,
    drone_state: DroneState,
    active_gate_index_changed: bool,
    elapsed_in_state_s: float,
) -> PilotState:
    if current == PilotState.TAKEOFF:
        altitude_reached = (
            drone_state.has_position
            and drone_state.pos_ned[2] <= -ALTITUDE_TARGET_M * 0.9
        )
        if altitude_reached or elapsed_in_state_s >= TAKEOFF_TIMEOUT_S:
            return PilotState.CHASE
        return current

    if current == PilotState.CHASE:
        if detection is not None:
            if estimate is not None and estimate.range_m is not None:
                if estimate.range_m < PASS_RANGE_M:
                    return PilotState.ADVANCE
            area_threshold = PASS_AREA_FRAC * 640 * 480
            if detection.area_px > area_threshold:
                return PilotState.ADVANCE

        lost_timeout = LOST_FRAMES_THRESHOLD / 20.0
        if detection is None and elapsed_in_state_s > lost_timeout:
            return PilotState.SEARCH
        return current

    if current == PilotState.ADVANCE:
        if active_gate_index_changed:
            return PilotState.CHASE
        if detection is None and elapsed_in_state_s > 0.5:
            return PilotState.CHASE
        return current

    if current == PilotState.SEARCH:
        if detection is not None and estimate is not None and estimate.confidence > 0:
            return PilotState.CHASE
        return current

    return current
