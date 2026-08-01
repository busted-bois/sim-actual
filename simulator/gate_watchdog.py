"""Gate-stall watchdog for the single-shot control-flight path (make classical blue).

`make auto` gets retry-on-stall from simulator/auto_flight.py; auto_gp.py runs a bare
control loop with no supervision at all, so a drone that clips a post or loses the
corridor flies on until someone notices. Same policy here -- the race_monitor
predicates are reused verbatim -- but driven as a state machine instead of inline
sleeps: control-flight owns the live vision window and the mp4 recorder, and a
blocking reset would freeze the window and punch a hole in the run video.

The reset itself is a drone-pose teleport (MAVLink 31000), not a race restart. The sim
usually re-runs the countdown by itself in a TRAINING session; GPPilot sees the new
countdown and re-arms on its own.
"""

from __future__ import annotations

import time

from simulator import run_meta
from simulator.race_monitor import (
    GATE1_WATCH_INTERVAL_S,
    SIM_RESET_WAIT_S,
    course_complete,
    gate1_fail,
    gate1_watch_line,
    gate_progress_stall,
    gate_progress_watch_line,
    passed_first_gate,
)

# The sim drops one of two back-to-back resets, so 31000 goes out twice.
RESET_REPEAT_S = 0.5


class GateStallWatchdog:
    """Reset the sim when gate progress stops.

    Duck-typed pilot contract: `gates_passed`, `reset_for_attempt()`, and `flying`
    (optional -- absent means always watching).
    """

    def __init__(self, controller, pilot, data, enabled=True):
        self.controller = controller
        self.pilot = pilot
        self.data = data
        self.enabled = enabled
        # Clock only runs while the pilot is flying; holding zero thrust on the pad
        # before the countdown is not a stall.
        self._armed = False
        self._last_active = 0
        self._t_advance = 0.0
        self._last_watch_log = 0.0
        # Non-None while a reset is in flight (second send pending / settling).
        self._reset_at = None
        self._second_sent = False
        self.resets = 0

    def tick(self, now=None):
        if not self.enabled:
            return
        if now is None:
            now = time.monotonic()

        if self._reset_at is not None:
            self._tick_settle(now)
            return

        if not getattr(self.pilot, "flying", True):
            self._armed = False
            return

        if not self._armed:
            self._arm(now)
            return

        # A finished course is not a stall -- GPPilot aborts to WAIT on its own.
        if course_complete(self.data):
            self._armed = False
            return

        active = self._active()
        if active > self._last_active:
            self._last_active = active
            self._t_advance = now
            self._last_watch_log = now
            print(f"[RACE] GATE_ADVANCE active={active}", flush=True)
            return

        elapsed = now - self._t_advance
        if passed_first_gate(self.data):
            if gate_progress_stall(self.data, self._last_active, elapsed):
                self._trigger("gate_stall", now)
                return
            watch_line = gate_progress_watch_line(self.data, self._last_active, elapsed)
        else:
            pilot_passed = int(getattr(self.pilot, "gates_passed", 0) or 0)
            if gate1_fail(self.data, elapsed, pilot_passed):
                self._trigger("gate1_fail", now)
                return
            watch_line = gate1_watch_line(self.data, elapsed, pilot_passed)

        if now - self._last_watch_log >= GATE1_WATCH_INTERVAL_S:
            self._last_watch_log = now
            print(watch_line, flush=True)

    def _active(self):
        return int(self.data.get("active_gate_index", 0) or 0)

    def _arm(self, now):
        self._armed = True
        self._last_active = self._active()
        self._t_advance = now
        self._last_watch_log = now

    def _tick_settle(self, now):
        since = now - self._reset_at
        if not self._second_sent and since >= RESET_REPEAT_S:
            self._second_sent = True
            self.controller.send_sim_reset_command()
        if since >= RESET_REPEAT_S + SIM_RESET_WAIT_S:
            self._reset_at = None
            race = self.data.get("race_status") or {}
            print(
                f"[RACE] post_reset active={self._active()} "
                f"sim_boot={race.get('sim_boot_time_ms', '?')}ms",
                flush=True,
            )

    def _trigger(self, outcome, now):
        active = self._active()
        self.resets += 1
        # attempt=None attaches the verdict to the pilot's live attempt record --
        # reset_for_attempt() closes that CSV a line below, so note it first.
        run_meta.note_outcome(outcome, active=active)
        print(
            f"[RACE] OUTCOME={outcome} active={active} "
            f"reset={self.resets} — resetting sim",
            flush=True,
        )
        # Drops the pilot to WAIT_FOR_DATA holding neutral, so it is not banking
        # into the teleport; also zeroes n_passed and closes the attempt log.
        self.pilot.reset_for_attempt()
        # The shared StateEstimator is deliberately NOT reset here.
        #
        # Re-running its init would hold `ready` False for its whole ground
        # buffer (~1 s at the IMU rate), and MAVLinkRX._publish_estimated_state
        # early-returns while not ready -- so vel_ned and attitude would go
        # missing for the first second after the teleport. Under the VQ2 block
        # those are the ONLY source for the speed PD and the error-space
        # attitude wire, so the pilot would re-launch blind to its own motion,
        # which is worse than the drift a reset would clear.
        #
        # What the reset would fix is position and gyro-integrated attitude.
        # Neither reaches this pilot: GPPilot flies GPEstimation's own AHRS
        # (re-seeded in its _reset_state), and no consumer on the auto_gp path
        # reads the ESKF's position. Baro would have re-anchored z on its own
        # anyway if it existed -- it does not; `make probe` on 2026-08-01 still
        # reports pressure_alt=nan, as state_estimator recorded on 2026-07-01.
        self.controller.send_sim_reset_command()
        self._reset_at = now
        self._second_sent = False
        self._armed = False
