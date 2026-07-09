"""Gate transition state machine.

Pure logic - no MAVLink or vision imports. Decides when to commit to the
next gate vs keep hovering on the current one.

Phases:
    APPROACH    - flying toward / centering on current gate
    HOVERING    - inside acceptance radius, slow; accumulating dwell time
    COMMITTED   - dwell exceeded t_min; one-tick signal to start dead reckon
    DEAD_RECKON - flying forward blind until the next gate is visible
"""

import math
from dataclasses import dataclass
from enum import Enum


class GatePhase(Enum):
    APPROACH = "approach"
    HOVERING = "hovering"
    COMMITTED = "committed"
    DEAD_RECKON = "dead_reckon"


@dataclass
class GateTransitionConfig:
    r_acceptance: float = 0.5  # m - lateral distance to count as "on gate"
    t_min: float = 0.5  # s - sole dwell threshold, no t_max
    v_max: float = 1.0  # m/s - above this speed, no hover accumulation


class GateTransitionTracker:
    def __init__(self, config=None):
        self.config = config or GateTransitionConfig()
        self.phase = GatePhase.APPROACH
        self.thover = 0.0
        self.d = math.inf
        self._last_update_time = None

    def update(self, d, speed, now):
        """Advance the state machine. d in metres (inf if gate not visible),
        speed in m/s, now in seconds. Returns the current GatePhase."""
        if self._last_update_time is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_update_time)
        self._last_update_time = now
        self.d = d

        if self.phase == GatePhase.COMMITTED:
            # one-tick signal consumed; move into dead reckoning
            self.phase = GatePhase.DEAD_RECKON
            return self.phase

        if self.phase == GatePhase.DEAD_RECKON:
            # exit only via on_next_gate_visible() or on_gate_passed()
            return self.phase

        hovering = (d < self.config.r_acceptance) and (speed < self.config.v_max)
        if hovering:
            self.thover += dt
            if self.thover > self.config.t_min:
                self.phase = GatePhase.COMMITTED
            else:
                self.phase = GatePhase.HOVERING
        else:
            # lost gate, drifted out, or too fast - reset dwell
            self.thover = 0.0
            self.phase = GatePhase.APPROACH

        return self.phase

    def should_commit(self):
        return self.phase == GatePhase.COMMITTED

    def on_gate_passed(self):
        """Server advanced active_gate_index - reset for the new gate."""
        self.phase = GatePhase.APPROACH
        self.thover = 0.0
        self.d = math.inf

    def on_next_gate_visible(self):
        """Perception sees the next gate while dead reckoning."""
        if self.phase == GatePhase.DEAD_RECKON:
            self.phase = GatePhase.APPROACH
            self.thover = 0.0
