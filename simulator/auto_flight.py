"""Gate-1 retry loop until course complete. Enabled via AUTO_FLIGHT=1 (make auto)."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from simulator.flight_debug import dbg_now, motion_snapshot
from simulator.preflight import (
    preflight_keep_safe,
    wait_for_fresh_race_start,
    wait_for_fresh_track,
    wait_for_race_go,
    wait_for_visual_go,
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
CANCEL_CONFIRM_WINDOW_S = float(os.environ.get("CANCEL_CONFIRM_WINDOW_S", "3"))


def auto_flight_enabled() -> bool:
    return os.environ.get("AUTO_FLIGHT", "").strip().lower() in _AUTO_FLIGHT_VALUES


def is_ctrl_cancel(ch: bytes) -> bool:
    if not ch:
        return False
    code = ch[0]
    return code == 3 or 1 <= code <= 26


def _run_attempt_preflight(controller, shared_data, pilot, attempt: int) -> bool:
    from simulator.countdown_detector import reset_countdown_gate

    print(f"[RACE] attempt={attempt} waiting for track_gates...", flush=True)
    controller.set_controls_enabled(False)
    controller.disarm()
    controller._safe_hold_logged = False
    reset_countdown_gate(shared_data)
    dbg_now("milestone", f"preflight_start attempt={attempt} {motion_snapshot(shared_data)}")

    race = shared_data.get("race_status") or {}
    shared_data["_preflight_race_start_baseline"] = race.get(
        "race_start_boot_time_ms", -1
    )

    if not wait_for_fresh_track(shared_data, controller=controller):
        print(
            "ERROR: no track burst — click Restart Race in FlightSim, then retry",
            flush=True,
        )
        return False

    dbg_now("milestone", f"track_ok {motion_snapshot(shared_data)}")

    if not wait_for_fresh_race_start(shared_data, timeout_s=15.0, controller=controller):
        print(
            "WARNING: race_start stale or missing — "
            "click Restart Race in FlightSim, then retry",
            flush=True,
        )
        return False

    dbg_now("milestone", f"fresh_race_start_ok {motion_snapshot(shared_data)}")

    if not wait_for_race_go(shared_data, controller=controller):
        print("ERROR: countdown never reached GO", flush=True)
        return False

    dbg_now("milestone", f"race_go_ok {motion_snapshot(shared_data)}")

    if not wait_for_visual_go(shared_data, controller=controller):
        print("ERROR: visual countdown never cleared", flush=True)
        return False

    dbg_now("milestone", f"visual_go_ok {motion_snapshot(shared_data)}")

    race = shared_data.get("race_status") or {}
    go_boot = shared_data.get("_latched_go_boot_ms", "?")
    print("Arming drone...", flush=True)
    controller.arm()
    time.sleep(0.15)
    controller.set_controls_enabled(True)
    print(
        "[RACE] controls enabled "
        f"sim_boot={race.get('sim_boot_time_ms')}ms "
        f"race_start={race.get('race_start_boot_time_ms')}ms "
        f"go_boot={go_boot}ms",
        flush=True,
    )
    dbg_now("milestone", f"controls_enabled {motion_snapshot(shared_data)}")

    if hasattr(pilot, "on_attempt_start"):
        pilot.on_attempt_start()

    return True


def _clear_attempt_state(shared_data) -> None:
    from simulator.countdown_detector import reset_countdown_gate

    shared_data.pop("track_gates", None)
    shared_data.pop("gates", None)
    shared_data.pop("_latched_go_boot_ms", None)
    shared_data.pop("_track_race_start_ms", None)
    shared_data.pop("_track_sim_boot_ms", None)
    shared_data.pop("_preflight_race_start_baseline", None)
    reset_countdown_gate(shared_data)


def _sleep_reset_wait(cancel: CancelListener) -> bool:
    """Sleep SIM_RESET_WAIT_S. Returns True if cancelled."""
    deadline = time.monotonic() + SIM_RESET_WAIT_S
    while time.monotonic() < deadline:
        if cancel.cancelled():
            return True
        time.sleep(0.05)
    return False


def _retry_after_fail(
    outcome: str,
    controller,
    pilot,
    shared_data,
    attempt: int,
    cancel: CancelListener,
) -> bool:
    """Reset sim after gate fail. Returns True if user cancelled."""
    active = int(shared_data.get("active_gate_index", 0) or 0)
    if outcome == "gate_stall":
        print(
            f"[RACE] OUTCOME=gate_stall attempt={attempt} active={active} retrying",
            flush=True,
        )
    else:
        print(f"[RACE] OUTCOME=gate1_fail attempt={attempt} retrying", flush=True)

    controller.set_controls_enabled(False)
    controller.set_attitude_rates(0, 0, 0, 0)
    controller.disarm()
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
    """Ctrl+C or Ctrl+letter twice within window cancels automation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._armed = False
        self._armed_at = 0.0

    def start(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)
        self._thread = threading.Thread(target=self._listen_keys, daemon=True)
        self._thread.start()

    def _on_sigint(self, signum, frame) -> None:
        self._handle_cancel_press()

    def cancelled(self) -> bool:
        return self._event.is_set()

    def _handle_cancel_press(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._event.is_set():
                return
            if (
                self._armed
                and now - self._armed_at <= CANCEL_CONFIRM_WINDOW_S
            ):
                self._event.set()
                return
            self._armed = True
            self._armed_at = now
        print(
            "[AUTO] press Ctrl+C or Ctrl+letter again to exit automation",
            flush=True,
        )

    def _listen_keys(self) -> None:
        if sys.platform == "win32":
            import msvcrt

            while not self._event.is_set():
                if msvcrt.kbhit():
                    if is_ctrl_cancel(msvcrt.getch()):
                        self._handle_cancel_press()
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
                    self._handle_cancel_press()
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
        "[AUTO] automation on (vision pilot) — "
        "Ctrl+C or Ctrl+letter twice cancels to normal sim",
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
        dbg_now("milestone", f"control_loop_start {motion_snapshot(shared_data)}")
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
                race = shared_data.get("race_status") or {}
                finish_ns = race.get("race_finish_time_ns", -1)
                active = shared_data.get("active_gate_index", 0)
                print(
                    f"[RACE] OUTCOME=success finish_ns={finish_ns} active={active}",
                    flush=True,
                )
                return "success", was_flying

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
                        if _retry_after_fail(
                            "gate_stall",
                            controller,
                            pilot,
                            shared_data,
                            attempt,
                            cancel,
                        ):
                            return "cancelled", was_flying
                        retry_attempt = True
                        break
                    elif (
                        time.monotonic() - last_watch_log
                        >= GATE1_WATCH_INTERVAL_S
                    ):
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
                if _retry_after_fail(
                    "gate1_fail",
                    controller,
                    pilot,
                    shared_data,
                    attempt,
                    cancel,
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
