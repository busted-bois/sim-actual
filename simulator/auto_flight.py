"""Gate-1 retry loop until course complete. Enabled via AUTO_FLIGHT=1 (make auto)."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from simulator.pilot import HOVER_THRUST
from simulator.preflight import (
    is_restart_arm_context,
    wait_for_fresh_track,
    wait_for_race_go,
    wait_for_race_start,
)
from simulator.race_monitor import (
    GATE1_WATCH_INTERVAL_S,
    SIM_RESET_WAIT_S,
    course_complete,
    gate1_fail,
    gate1_watch_line,
    passed_first_gate,
)

_AUTO_FLIGHT_VALUES = frozenset({"1", "true", "yes"})


def auto_flight_enabled() -> bool:
    return os.environ.get("AUTO_FLIGHT", "").strip().lower() in _AUTO_FLIGHT_VALUES


def is_ctrl_cancel(ch: bytes) -> bool:
    if not ch:
        return False
    code = ch[0]
    return code == 3 or 1 <= code <= 26


def _pilot_hover_thrust(pilot) -> float:
    from rl.fly2_course import Fly2CoursePilot, HOVER_T

    if isinstance(pilot, Fly2CoursePilot):
        return HOVER_T
    return HOVER_THRUST


def _run_attempt_preflight(controller, shared_data, pilot, attempt: int) -> bool:
    print(f"[RACE] attempt={attempt} waiting for track_gates...", flush=True)
    if not wait_for_fresh_track(shared_data):
        print(
            "ERROR: no track burst — click Restart Race in FlightSim, then retry",
            flush=True,
        )
        return False

    if not wait_for_race_start(shared_data, timeout_s=15.0):
        print(
            "WARNING: race_start not in telemetry yet — arm anyway and keep waiting for GO",
            flush=True,
        )

    print("Arming drone...", flush=True)
    controller.arm()
    time.sleep(0.15)

    race = shared_data.get("race_status") or {}
    armed_sim_boot_ms = race.get("sim_boot_time_ms", 0)
    is_restart = attempt > 1 and is_restart_arm_context(armed_sim_boot_ms)

    if not wait_for_race_go(
        shared_data,
        armed_sim_boot_ms=armed_sim_boot_ms,
        is_restart=is_restart,
    ):
        print("ERROR: countdown never reached GO", flush=True)
        return False

    if hasattr(pilot, "on_attempt_start"):
        pilot.on_attempt_start()

    return True


def _clear_attempt_state(shared_data) -> None:
    shared_data.pop("track_gates", None)
    shared_data.pop("gates", None)
    shared_data.pop("_latched_go_boot_ms", None)


def _sleep_reset_wait(cancel: CancelListener) -> bool:
    """Sleep SIM_RESET_WAIT_S. Returns True if cancelled."""
    deadline = time.monotonic() + SIM_RESET_WAIT_S
    while time.monotonic() < deadline:
        if cancel.cancelled():
            return True
        time.sleep(0.05)
    return False


def _retry_after_gate1_fail(
    controller, pilot, shared_data, attempt: int, cancel: CancelListener
) -> bool:
    """Reset sim after gate-1 fail. Returns True if user cancelled."""
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
    """Ctrl+C or Ctrl+letter cancels automation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)
        self._thread = threading.Thread(target=self._listen_keys, daemon=True)
        self._thread.start()

    def _on_sigint(self, signum, frame) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def _listen_keys(self) -> None:
        if sys.platform == "win32":
            import msvcrt

            while not self._event.is_set():
                if msvcrt.kbhit():
                    if is_ctrl_cancel(msvcrt.getch()):
                        self._event.set()
                        return
                time.sleep(0.05)
            return

        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            return

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1).encode("latin-1", errors="ignore")
                if is_ctrl_cancel(ch):
                    self._event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_auto_flight_loop(controller, pilot, shared_data) -> tuple[str, bool]:
    """Run retry loop. Returns (outcome, was_flying).

    outcome: success | cancelled | preflight_fail
    was_flying: True if the 250 Hz loop had started at least once
    """
    cancel = CancelListener()
    cancel.start()
    print(
        "[AUTO] automation on (fly2 speed pilot) — "
        "Ctrl+C or Ctrl+letter cancels to normal sim",
        flush=True,
    )

    attempt = 0
    was_flying = False

    while not cancel.cancelled():
        attempt += 1
        if not _run_attempt_preflight(controller, shared_data, pilot, attempt):
            return "preflight_fail", was_flying
        if cancel.cancelled():
            return "cancelled", was_flying

        print(f"[RACE] attempt={attempt} control loop started", flush=True)
        t_go = time.monotonic()
        gate1_latched = False
        retry_attempt = False
        last_watch_log = t_go

        while not cancel.cancelled():
            controller.update()
            was_flying = True

            if course_complete(shared_data):
                race = shared_data.get("race_status") or {}
                finish_ns = race.get("race_finish_time_ns", -1)
                active = shared_data.get("active_gate_index", 0)
                print(
                    f"[RACE] OUTCOME=success finish_ns={finish_ns} active={active}",
                    flush=True,
                )
                return "success", was_flying

            if gate1_latched:
                continue

            elapsed = time.monotonic() - t_go
            pilot_passed = pilot.gates_passed

            if passed_first_gate(shared_data):
                gate1_latched = True
                print("[RACE] GATE1=pass active=1", flush=True)
            elif gate1_fail(shared_data, elapsed, pilot_passed):
                if _retry_after_gate1_fail(
                    controller, pilot, shared_data, attempt, cancel
                ):
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
            return "cancelled", was_flying
        if retry_attempt:
            continue

    return "cancelled", was_flying
