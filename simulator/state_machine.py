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
    """Pure transition function for pilot state machine.

    Args:
        current: Current pilot state
        detection: Latest gate detection from vision, or None if no gate detected
        estimate: Latest gate estimate (range + confidence), or None if not available
        drone_state: Current drone state (position, velocity, yaw, etc.)
        active_gate_index_changed: True if the active gate index just changed
        elapsed_in_state_s: Time spent in current state, in seconds

    Returns:
        Next pilot state (may be same as current)
    """
    # TAKEOFF → CHASE: altitude reached OR timeout
    if current == PilotState.TAKEOFF:
        altitude_reached = (
            drone_state.has_position
            and drone_state.pos_ned[2] <= -ALTITUDE_TARGET_M * 0.9
        )
        timeout = elapsed_in_state_s >= TAKEOFF_TIMEOUT_S
        if altitude_reached or timeout:
            return PilotState.CHASE
        return current

    # CHASE transitions
    if current == PilotState.CHASE:
        # CHASE → ADVANCE: gate centered AND close
        if detection is not None:
            # If range estimate available, use range criterion
            if estimate is not None and estimate.range_m is not None:
                if estimate.range_m < PASS_RANGE_M:
                    return PilotState.ADVANCE
            # If no range, gate must be large (close) to advance
            area_threshold = PASS_AREA_FRAC * 640 * 480
            if detection.area_px > area_threshold:
                return PilotState.ADVANCE

        # CHASE → SEARCH: gate lost for sustained cycles
        lost_timeout = LOST_FRAMES_THRESHOLD / 20.0  # frames at ~20fps
        if detection is None and elapsed_in_state_s > lost_timeout:
            return PilotState.SEARCH

        return current

    # ADVANCE transitions
    if current == PilotState.ADVANCE:
        # ADVANCE → CHASE: next gate to pursue
        if active_gate_index_changed:
            return PilotState.CHASE

        # ADVANCE → CHASE: gate lost after pass (will go to SEARCH if sustained)
        if detection is None and elapsed_in_state_s > 0.5:
            return PilotState.CHASE

        return current

    # SEARCH → CHASE: gate re-acquired
    if current == PilotState.SEARCH:
        if detection is not None and estimate is not None and estimate.confidence > 0:
            return PilotState.CHASE
        return current

    # Default: no transition
    return current
