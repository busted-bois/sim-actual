"""
Autonomous gate-following navigator.

Consumes the latest :class:`~simulator.vision_processing.FrameAnalysis` and emits
a body-frame velocity + yaw-rate :class:`Command` each control tick. The strategy
is a small state machine:

    SEARCHING      - nothing useful in view: rotate slowly to scan (and optionally
                     creep forward) until a gate or the path appears.
    APPROACHING    - gate visible: yaw to center it, adjust altitude, ease forward.
    PASSING        - gate is big and centered: drive straight through for a fixed
                     short burst (the gate leaves the FOV as we get close, so we
                     commit blind for pass_through_seconds).
    FOLLOWING_PATH - no gate but blue path visible: steer along the path.
    COMPLETE       - safety stop: course finished (or a hard cap tripped). Hover.

Safety / end-of-course: once we've seen *anything* (gate or path) at least once,
if we then see *nothing* for ``end_of_course_seconds`` we declare the course
complete and stop. ``require_detection_before_end`` prevents a premature "finish"
during the second or two before the first camera frame arrives.
"""

import time
from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    PASSING = "PASSING"
    FOLLOWING_PATH = "FOLLOWING_PATH"
    COMPLETE = "COMPLETE"


@dataclass
class Command:
    """Body-frame setpoint for one control tick (NED body: +vx fwd, +vy right, +vz down)."""

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    phase: Phase = Phase.SEARCHING
    complete: bool = False
    gates_passed: int = 0


def _clamp(value, limit):
    return max(-limit, min(limit, value))


class GateNavigator:
    def __init__(self, config, clock=time.time):
        nav = config["navigation"]
        safety = config["safety"]

        self.cruise_speed = nav["cruise_speed"]
        self.approach_speed = nav["approach_speed"]
        self.pass_through_speed = nav["pass_through_speed"]
        self.pass_through_seconds = nav["pass_through_seconds"]
        self.path_speed_frac = nav["path_speed_frac"]
        self.yaw_gain = nav["yaw_gain"]
        self.max_yaw_rate = nav["max_yaw_rate"]
        self.vertical_gain = nav["vertical_gain"]
        self.max_vertical_speed = nav["max_vertical_speed"]
        self.gate_pass_area_frac = nav["gate_pass_area_frac"]
        self.gate_pass_center_tol = nav["gate_pass_center_tol"]
        self.search_yaw_rate = nav["search_yaw_rate"]
        self.search_creep_speed = nav["search_creep_speed"]

        self.require_detection_before_end = safety["require_detection_before_end"]
        self.end_of_course_seconds = safety["end_of_course_seconds"]
        self.max_run_seconds = safety["max_run_seconds"]

        self._clock = clock
        self.phase = Phase.SEARCHING
        self.gates_passed = 0

        self.start_time = None
        self.last_seen_time = None  # last tick at which a gate or path was visible
        self.has_seen_anything = False
        self.pass_start_time = None
        self._complete_reason = None

    # -- logging helper -----------------------------------------------------------------
    def _set_phase(self, phase):
        if phase != self.phase:
            print(f"[nav] {self.phase.value} -> {phase.value}", flush=True)
            self.phase = phase

    def _complete(self, reason):
        if self.phase != Phase.COMPLETE:
            print(
                f"[nav] COURSE COMPLETE: {reason} (gates passed: {self.gates_passed})",
                flush=True,
            )
        self._set_phase(Phase.COMPLETE)
        self._complete_reason = reason
        return Command(
            phase=Phase.COMPLETE, complete=True, gates_passed=self.gates_passed
        )

    # -- main entry point ---------------------------------------------------------------
    def compute(self, analysis, now=None):
        """
        Decide the next command from the latest frame analysis.

        ``analysis`` may be None (no frame received yet). ``now`` defaults to the
        injected clock; tests pass it explicitly to drive timing deterministically.
        """
        now = self._clock() if now is None else now
        if self.start_time is None:
            self.start_time = now

        gate = analysis.gate if analysis is not None else None
        path = analysis.path if analysis is not None else None
        gate_found = bool(gate and gate.found)
        path_found = bool(path and path.found)

        if gate_found or path_found:
            self.last_seen_time = now
            self.has_seen_anything = True

        # --- hard safety: absolute run-time cap ---
        if self.max_run_seconds and (now - self.start_time) > self.max_run_seconds:
            return self._complete(f"max_run_seconds ({self.max_run_seconds}s) reached")

        # --- already finished: stay stopped ---
        if self.phase == Phase.COMPLETE:
            return Command(
                phase=Phase.COMPLETE, complete=True, gates_passed=self.gates_passed
            )

        # --- end-of-course: nothing in view for too long after we'd seen something ---
        can_end = self.has_seen_anything or not self.require_detection_before_end
        if can_end and self.last_seen_time is not None:
            if (now - self.last_seen_time) > self.end_of_course_seconds:
                return self._complete(
                    f"no gate/path for {self.end_of_course_seconds:.1f}s"
                )

        # --- committed pass-through burst (fly blind through the gate) ---
        if self.phase == Phase.PASSING:
            if (now - self.pass_start_time) < self.pass_through_seconds:
                return Command(
                    vx=self.pass_through_speed,
                    phase=Phase.PASSING,
                    gates_passed=self.gates_passed,
                )
            self._set_phase(Phase.SEARCHING)  # burst done, look for the next gate

        # --- gate in view: approach, or commit to passing ---
        if gate_found:
            centered = abs(gate.cx_norm) <= self.gate_pass_center_tol
            if gate.area_frac >= self.gate_pass_area_frac and centered:
                self.gates_passed += 1
                self.pass_start_time = now
                self._set_phase(Phase.PASSING)
                print(f"[nav] passing gate #{self.gates_passed}", flush=True)
                return Command(
                    vx=self.pass_through_speed,
                    phase=Phase.PASSING,
                    gates_passed=self.gates_passed,
                )

            self._set_phase(Phase.APPROACHING)
            yaw_rate = _clamp(self.yaw_gain * gate.cx_norm, self.max_yaw_rate)
            vz = _clamp(self.vertical_gain * gate.cy_norm, self.max_vertical_speed)
            # ease off forward speed when the gate is off to the side so we can turn onto it
            centering = max(0.0, 1.0 - abs(gate.cx_norm))
            vx = self.approach_speed * (0.4 + 0.6 * centering)
            return Command(
                vx=vx,
                vz=vz,
                yaw_rate=yaw_rate,
                phase=Phase.APPROACHING,
                gates_passed=self.gates_passed,
            )

        # --- no gate, but path visible: follow the path ---
        if path_found:
            self._set_phase(Phase.FOLLOWING_PATH)
            yaw_rate = _clamp(self.yaw_gain * path.cx_norm, self.max_yaw_rate)
            return Command(
                vx=self.cruise_speed * self.path_speed_frac,
                yaw_rate=yaw_rate,
                phase=Phase.FOLLOWING_PATH,
                gates_passed=self.gates_passed,
            )

        # --- nothing in view (yet): scan ---
        self._set_phase(Phase.SEARCHING)
        return Command(
            vx=self.search_creep_speed,
            yaw_rate=self.search_yaw_rate,
            phase=Phase.SEARCHING,
            gates_passed=self.gates_passed,
        )
