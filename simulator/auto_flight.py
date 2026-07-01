"""Continuous in-session retry loop. Enabled via AUTO_FLIGHT=1 (make auto)."""

from __future__ import annotations

import os
import signal
import time

from simulator.pilot import HOVER_THRUST
from simulator.preflight import (
    is_restart_arm_context,
    wait_for_fresh_race_start_vq2,
    wait_for_race_go,
    wait_for_race_start,
    wait_for_race_status,
    wait_for_session_ready,
)
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

_AUTO_FLIGHT_VALUES = frozenset({"1", "true", "yes"})


def auto_flight_enabled() -> bool:
    return os.environ.get("AUTO_FLIGHT", "").strip().lower() in _AUTO_FLIGHT_VALUES


def _pilot_hover_thrust(pilot) -> float:
    return HOVER_THRUST


def _sleep_s(cancel: CancelListener, seconds: float) -> bool:
    """Sleep up to seconds. Returns True if cancelled."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancel.cancelled():
            return True
        time.sleep(0.05)
    return False


def _run_attempt_preflight(controller, shared_data, pilot, attempt: int, cancel) -> bool:
    if attempt == 1:
        print(f"[RACE] attempt={attempt} waiting for VQ2 session...", flush=True)
        if not wait_for_session_ready(shared_data, cancel=cancel):
            return False

        print(
            "Click Race in FlightSim (SUBMISSION or TRAINING — same map)...",
            flush=True,
        )
        if not wait_for_race_status(shared_data, timeout_s=60.0, cancel=cancel):
            if cancel.cancelled():
                return False
            print("WARNING: race_status not in telemetry yet", flush=True)

        if not wait_for_race_start(shared_data, timeout_s=60.0, cancel=cancel):
            if cancel.cancelled():
                return False
            print(
                "ERROR: Race not started — click Race in FlightSim",
                flush=True,
            )
            return False
    else:
        print(f"[RACE] attempt={attempt} waiting for restart countdown...", flush=True)
        race = shared_data.get("race_status") or {}
        is_restart = is_restart_arm_context(race.get("sim_boot_time_ms", 0))
        if not wait_for_fresh_race_start_vq2(
            shared_data, timeout_s=30.0, cancel=cancel, is_restart=is_restart
        ):
            if cancel.cancelled():
                return False
            print(
                "ERROR: no fresh race_start — click Restart Race in FlightSim",
                flush=True,
            )
            return False

    print("Arming drone...", flush=True)
    controller.arm()
    if _sleep_s(cancel, 0.15):
        return False

    race = shared_data.get("race_status") or {}
    armed_sim_boot_ms = race.get("sim_boot_time_ms", 0)
    is_restart = attempt > 1 or is_restart_arm_context(armed_sim_boot_ms)

    if not wait_for_race_go(
        shared_data,
        armed_sim_boot_ms=armed_sim_boot_ms,
        is_restart=is_restart,
        cancel=cancel,
    ):
        return False

    if hasattr(pilot, "on_attempt_start"):
        pilot.on_attempt_start()

    return True


def _clear_attempt_state(shared_data) -> None:
    race = shared_data.get("race_status") or {}
    shared_data["_preflight_race_start_baseline"] = race.get(
        "race_start_boot_time_ms", -1
    )
    shared_data.pop("_latched_go_boot_ms", None)


def _sleep_reset_wait(cancel: CancelListener) -> bool:
    """Sleep SIM_RESET_WAIT_S. Returns True if cancelled."""
    deadline = time.monotonic() + SIM_RESET_WAIT_S
    while time.monotonic() < deadline:
        if cancel.cancelled():
            return True
        time.sleep(0.05)
    return False


def _retry_after_outcome(
    outcome: str,
    controller,
    pilot,
    shared_data,
    attempt: int,
    cancel: CancelListener,
    lap_elapsed_s: float | None = None,
    best_lap_s: float | None = None,
) -> bool:
    """Reset sim after fail/success. Returns True if user cancelled."""
    active = int(shared_data.get("active_gate_index", 0) or 0)
    if outcome == "success":
        best_str = (
            f" best={best_lap_s:.1f}s" if best_lap_s is not None else ""
        )
        lap_str = f" lap={lap_elapsed_s:.1f}s" if lap_elapsed_s is not None else ""
        print(
            f"[RACE] OUTCOME=success attempt={attempt}{lap_str}{best_str} — restarting",
            flush=True,
        )
    elif outcome == "gate_stall":
        print(
            f"[RACE] OUTCOME=gate_stall attempt={attempt} active={active} retrying",
            flush=True,
        )
    else:
        print(f"[RACE] OUTCOME=gate1_fail attempt={attempt} retrying", flush=True)

    hover = _pilot_hover_thrust(pilot)
    controller.set_attitude_rates(0, 0, 0, hover)
    time.sleep(0.2)
    controller.send_sim_reset_command()
    time.sleep(0.5)
    controller.send_sim_reset_command()
    if _sleep_reset_wait(cancel):
        return True

    race = shared_data.get("race_status") or {}
    active = shared_data.get("active_gate_index", 0)
    sim_boot = race.get("sim_boot_time_ms", "?")
    print(f"[RACE] post_reset active={active} sim_boot={sim_boot}ms", flush=True)
    pilot.reset_for_attempt()
    _clear_attempt_state(shared_data)
    print(
        "[RACE] reset sent — click Restart Race in FlightSim "
        "if countdown doesn't start within 30s",
        flush=True,
    )
    return False


class CancelListener:
    """Ctrl+C cancels automation."""

    def __init__(self) -> None:
        self._cancelled = False

    def start(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, signum, frame) -> None:
        if not self._cancelled:
            print("[AUTO] cancel requested (Ctrl+C)", flush=True)
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled


def run_auto_flight_loop(controller, pilot, shared_data) -> tuple[str, bool]:
    """Run continuous retry loop until Ctrl+C.

    Returns (outcome, was_flying).
    outcome: cancelled | preflight_fail
    """
    cancel = CancelListener()
    cancel.start()
    print(
        "[AUTO] overnight automation on — same pilot as make sim; "
        "Ctrl+C stops (preflight, flight, or retry)",
        flush=True,
    )

    attempt = 0
    was_flying = False
    best_lap_s: float | None = None

    while not cancel.cancelled():
        attempt += 1
        if not _run_attempt_preflight(controller, shared_data, pilot, attempt, cancel):
            if cancel.cancelled():
                print("[AUTO] cancelled — resuming normal sim mode", flush=True)
                return "cancelled", was_flying
            print(
                "ERROR: enter AI-GP VIRTUAL QUALIFIER R2 — SUBMISSION or TRAINING",
                flush=True,
            )
            return "preflight_fail", was_flying
        if cancel.cancelled():
            print("[AUTO] cancelled — resuming normal sim mode", flush=True)
            return "cancelled", was_flying

        print(f"[RACE] attempt={attempt} control loop started", flush=True)
        t_go = time.monotonic()
        gate1_latched = False
        retry_attempt = False
        last_watch_log = t_go
        last_active = 0
        t_last_advance = t_go

        while not cancel.cancelled():
            controller.update()
            was_flying = True

            if course_complete(shared_data):
                lap_elapsed_s = time.monotonic() - t_go
                if best_lap_s is None or lap_elapsed_s < best_lap_s:
                    best_lap_s = lap_elapsed_s
                if _retry_after_outcome(
                    "success",
                    controller,
                    pilot,
                    shared_data,
                    attempt,
                    cancel,
                    lap_elapsed_s=lap_elapsed_s,
                    best_lap_s=best_lap_s,
                ):
                    print("[AUTO] cancelled — resuming normal sim mode", flush=True)
                    return "cancelled", was_flying
                retry_attempt = True
                break

            active = int(shared_data.get("active_gate_index", 0) or 0)

            if gate1_latched:
                if active > last_active:
                    last_active = active
                    t_last_advance = time.monotonic()
                    print(f"[RACE] GATE_ADVANCE active={active}", flush=True)
                else:
                    elapsed_advance = time.monotonic() - t_last_advance
                    if gate_progress_stall(
                        shared_data, last_active, elapsed_advance
                    ):
                        if _retry_after_outcome(
                            "gate_stall",
                            controller,
                            pilot,
                            shared_data,
                            attempt,
                            cancel,
                        ):
                            print(
                                "[AUTO] cancelled — resuming normal sim mode",
                                flush=True,
                            )
                            return "cancelled", was_flying
                        retry_attempt = True
                        break
                    if time.monotonic() - last_watch_log >= GATE1_WATCH_INTERVAL_S:
                        print(
                            gate_progress_watch_line(
                                shared_data, last_active, elapsed_advance
                            ),
                            flush=True,
                        )
                        last_watch_log = time.monotonic()
                continue

            elapsed = time.monotonic() - t_go
            pilot_passed = pilot.gates_passed

            if passed_first_gate(shared_data):
                gate1_latched = True
                last_active = active
                t_last_advance = time.monotonic()
                last_watch_log = t_last_advance
                print("[RACE] GATE1=pass active=1", flush=True)
            elif gate1_fail(shared_data, elapsed, pilot_passed):
                if _retry_after_outcome(
                    "gate1_fail",
                    controller,
                    pilot,
                    shared_data,
                    attempt,
                    cancel,
                ):
                    print("[AUTO] cancelled — resuming normal sim mode", flush=True)
                    return "cancelled", was_flying
                retry_attempt = True
                break
            elif time.monotonic() - last_watch_log >= GATE1_WATCH_INTERVAL_S:
                print(
                    gate1_watch_line(shared_data, elapsed, pilot_passed),
                    flush=True,
                )
                last_watch_log = time.monotonic()

        if cancel.cancelled():
            print("[AUTO] cancelled — resuming normal sim mode", flush=True)
            return "cancelled", was_flying
        if retry_attempt:
            continue

    print("[AUTO] cancelled — resuming normal sim mode", flush=True)
    return "cancelled", was_flying
