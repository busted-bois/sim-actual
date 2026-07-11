from dataclasses import dataclass
from simulator.config import (
    SEARCH_SWEEP_YAW_RATE,
    SEARCH_SWEEP_PERIOD_S,
    SEARCH_EXPAND_STEP_M,
    SEARCH_MAX_EXPAND_M,
    SEARCH_FORWARD_MPS,
    SEARCH_SWEEPS_BEFORE_EXPAND,
    DroneState,
)


@dataclass
class SearchCommand:
    yaw_rate_cmd: float
    lateral_vel_mps: float  # body frame
    forward_vel_mps: float  # body frame
    phase: str  # "SWEEP", "EXPAND", "RELOCATE"


class ExpandingSearch:
    def __init__(self):
        self._sweep_count: int = 0
        self._sweep_timer: float = 0.0
        self._sweep_direction: float = 1.0  # alternating ±1
        self._level: int = 0  # expand level
        self._expand_side: float = 1.0  # alternating ±1
        self._phase: str = "SWEEP"

    def reset(self):
        self._sweep_count = 0
        self._sweep_timer = 0.0
        self._sweep_direction = 1.0
        self._level = 0
        self._expand_side = 1.0
        self._phase = "SWEEP"

    def on_gate_lost(self):
        self._sweep_count += 1

    def next_command(self, drone_state: DroneState, dt_s: float) -> SearchCommand:
        self._sweep_timer += dt_s

        if self._sweep_count < SEARCH_SWEEPS_BEFORE_EXPAND:
            self._phase = "SWEEP"
            yaw_rate = SEARCH_SWEEP_YAW_RATE * self._sweep_direction

            if self._sweep_timer >= SEARCH_SWEEP_PERIOD_S:
                self._sweep_timer = 0.0
                self._sweep_direction *= -1

            return SearchCommand(
                yaw_rate_cmd=yaw_rate,
                lateral_vel_mps=0.0,
                forward_vel_mps=0.0,
                phase="SWEEP",
            )
        else:
            self._phase = "EXPAND"
            lateral_offset = SEARCH_EXPAND_STEP_M * self._level * self._expand_side

            if abs(lateral_offset) >= SEARCH_MAX_EXPAND_M:
                self._level = 0
                self._expand_side *= -1
                lateral_offset = SEARCH_EXPAND_STEP_M * self._level * self._expand_side

            if self._sweep_timer >= SEARCH_SWEEP_PERIOD_S:
                self._sweep_timer = 0.0
                self._level += 1
                self._expand_side *= -1

            return SearchCommand(
                yaw_rate_cmd=0.0,
                lateral_vel_mps=lateral_offset,
                forward_vel_mps=SEARCH_FORWARD_MPS,
                phase="EXPAND",
            )
