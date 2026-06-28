import os
import socket
import time

from simulator.flight_debug import dbg, dbg_now, motion_snapshot

PREFLIGHT_TIMEOUT_S = 30.0
AUTO_TRACK_TIMEOUT_S = 60.0
PREFLIGHT_POLL_S = 0.1
RACE_GO_TIMEOUT_S = 45.0
# Match control loop rate so we react within one tick of GO.
RACE_GO_POLL_S = 0.004

# Sim 3-2-1 countdown length.
RACE_COUNTDOWN_MS = int(os.environ.get("RACE_COUNTDOWN_MS", "3000"))

# If race_start is this far ahead of sim_boot at latch, treat it as scheduled GO time.
COUNTDOWN_SCHEDULED_THRESHOLD_MS = 1500

# race_start must be within this window of track burst sim_boot to count as fresh.
TRACK_RACE_START_TOLERANCE_MS = 500

GO_POST_MARGIN_MS = int(os.environ.get("GO_POST_MARGIN_MS", "400"))
VISUAL_GO_SEE_TIMEOUT_S = float(os.environ.get("VISUAL_GO_SEE_TIMEOUT_S", "8"))
PREFLIGHT_SAFE_INTERVAL_S = 0.1

_AUTO_TRUE = frozenset({"1", "true", "yes"})

# After an in-sim restart sim_boot_time_ms resets near zero while the first run stays high.
RESTART_ARM_BOOT_THRESHOLD_MS = int(
    os.environ.get("RESTART_ARM_BOOT_THRESHOLD_MS", "10000")
)


def is_restart_arm_context(armed_sim_boot_ms):
    if armed_sim_boot_ms is None or armed_sim_boot_ms <= 0:
        return False
    return armed_sim_boot_ms < RESTART_ARM_BOOT_THRESHOLD_MS


def restart_go_boot_ms(race_start_boot_ms):
    return race_start_boot_ms + RACE_COUNTDOWN_MS


def race_go_boot_ms(sim_boot_ms, race_start_boot_ms, is_restart=False):
    """Sim boot ms when on-screen timer hits 0."""
    del is_restart
    if race_start_boot_ms < 0:
        return None
    delta = race_start_boot_ms - sim_boot_ms
    if delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        return race_start_boot_ms
    return restart_go_boot_ms(race_start_boot_ms)


def race_go_already_passed(data, is_restart=False):
    del is_restart
    race = data.get("race_status") or {}
    race_start = race.get("race_start_boot_time_ms", -1)
    if race_start < 0:
        return False
    go_boot = race_go_boot_ms(race.get("sim_boot_time_ms", 0), race_start)
    if go_boot is None:
        return False
    return race.get("sim_boot_time_ms", 0) >= go_boot


def latch_race_go_boot_ms(sim_boot_ms, race_start_boot_ms, is_restart=False):
    """Latch GO instant: scheduled race_start or race_start + countdown."""
    del is_restart
    if race_start_boot_ms < 0:
        return None, None

    go_boot_ms = race_go_boot_ms(sim_boot_ms, race_start_boot_ms)
    delta = race_start_boot_ms - sim_boot_ms
    if sim_boot_ms >= go_boot_ms:
        branch = "at_go"
    elif delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        branch = "scheduled"
    else:
        branch = "countdown"

    return go_boot_ms, branch


class RaceGoLatch:
    """Shared latch state for wait_for_race_go and pilot restart countdown."""

    def __init__(self):
        self.go_boot_ms = None
        self.branch = None
        self.armed_sim_boot_ms = None
        self.is_restart = False

    def reset_for_arm(self, armed_sim_boot_ms=None, is_restart=False):
        self.go_boot_ms = None
        self.branch = None
        self.armed_sim_boot_ms = armed_sim_boot_ms
        self.is_restart = is_restart

    def try_latch(self, sim_boot_ms, race_start_boot_ms, allow_at_go=True):
        if self.go_boot_ms is not None:
            return self.go_boot_ms, self.branch

        if race_start_boot_ms < 0:
            return None, None

        if self.armed_sim_boot_ms is not None and sim_boot_ms <= self.armed_sim_boot_ms:
            return None, None

        go_boot_ms, branch = latch_race_go_boot_ms(
            sim_boot_ms,
            race_start_boot_ms,
            is_restart=self.is_restart,
        )
        if branch == "at_go" and not allow_at_go:
            return None, None

        self.go_boot_ms = go_boot_ms
        self.branch = branch
        return self.go_boot_ms, self.branch


def poll_race_go(data, latch, allow_at_go=True, margin_ms=0):
    """One wait_for_race_go iteration."""
    race = data.get("race_status") or {}
    race_start = race.get("race_start_boot_time_ms", -1)
    sim_boot = race.get("sim_boot_time_ms", 0)

    latch.try_latch(sim_boot, race_start, allow_at_go=allow_at_go)

    if latch.go_boot_ms is None:
        return False, None

    return (
        race_go_allowed(data, go_boot_ms=latch.go_boot_ms, margin_ms=margin_ms),
        latch.go_boot_ms,
    )


def auto_go_vision_enabled() -> bool:
    return os.environ.get("AUTO_GO_VISION", "1").strip().lower() in _AUTO_TRUE


_last_preflight_safe = 0.0
_last_preflight_safe_log = 0.0


def preflight_keep_safe(controller) -> None:
    """Disarm + zero thrust during preflight waits."""
    global _last_preflight_safe, _last_preflight_safe_log
    if controller is None:
        return
    now = time.monotonic()
    if now - _last_preflight_safe < PREFLIGHT_SAFE_INTERVAL_S:
        return
    _last_preflight_safe = now
    controller.set_controls_enabled(False)
    controller.send_safe_hold()
    if now - _last_preflight_safe_log >= 0.5:
        _last_preflight_safe_log = now
        data = getattr(controller, "data", {}) or {}
        dbg(
            "safe_tick",
            f"armed={data.get('armed', False)} "
            f"controls={controller._controls_enabled} "
            f"thrust_cmd={controller._thrust:.3f} "
            f"vel_ned={data.get('vel_ned', '?')}",
            throttle_s=0.0,
        )


def race_finished(data):
    """True when sim reports race_finish_time_ns >= 0 (lap complete)."""
    race = data.get("race_status")
    if race is None:
        return False
    finish_ns = race.get("race_finish_time_ns", -1)
    return finish_ns is not None and finish_ns >= 0


def race_go_allowed(data, go_boot_ms=None, margin_ms=0, is_restart=False):
    """True when sim_boot >= latched GO (+ optional margin)."""
    del is_restart
    race = data.get("race_status")
    if race is None or go_boot_ms is None:
        return False
    return race.get("sim_boot_time_ms", 0) >= go_boot_ms + margin_ms


def wait_for_track(data, timeout_s=PREFLIGHT_TIMEOUT_S, controller=None):
    print("Preflight: waiting for track_gates (click Race in FlightSim)...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        preflight_keep_safe(controller)
        if data.get("track_gates"):
            print("Preflight OK: track_gates loaded", flush=True)
            return True
        time.sleep(PREFLIGHT_POLL_S)
    print("Preflight timeout: track_gates not ready", flush=True)
    return False


def wait_for_fresh_track(data, timeout_s=AUTO_TRACK_TIMEOUT_S, controller=None):
    """Wait for a new track burst; clears stale gate data first."""
    data.pop("track_gates", None)
    data.pop("gates", None)
    print(
        "Preflight: waiting for fresh track_gates "
        "(click Race or Restart Race in FlightSim)...",
        flush=True,
    )
    print(
        "  Run make auto first, then click Race "
        "(or Restart Race if you already raced)",
        flush=True,
    )
    deadline = time.time() + timeout_s
    last_status = 0.0
    while time.time() < deadline:
        preflight_keep_safe(controller)
        if data.get("track_gates"):
            race = data.get("race_status") or {}
            data["_track_race_start_ms"] = race.get("race_start_boot_time_ms", -1)
            data["_track_sim_boot_ms"] = race.get("sim_boot_time_ms", -1)
            sim_boot = race.get("sim_boot_time_ms", 0)
            race_start = race.get("race_start_boot_time_ms", -1)
            go_boot, branch = latch_race_go_boot_ms(sim_boot, race_start)
            passed = race_go_already_passed({"race_status": race})
            dbg_now(
                "track_burst",
                f"track_sim_boot={sim_boot} track_race_start={race_start} "
                f"go_boot={go_boot} branch={branch} go_passed={passed} "
                f"{motion_snapshot(data)}",
            )
            print("Preflight OK: track_gates loaded", flush=True)
            return True
        now = time.time()
        if now - last_status >= 5.0:
            race = data.get("race_status") or {}
            print(
                "[RACE] waiting... "
                f"odometry={bool(data.get('odometry'))} "
                f"race_start={race.get('race_start_boot_time_ms', -1)} "
                f"track_gates={bool(data.get('track_gates'))}",
                flush=True,
            )
            last_status = now
        time.sleep(PREFLIGHT_POLL_S)
    print(
        "Preflight timeout: track_gates not received — "
        "click Restart Race in FlightSim, then retry make auto",
        flush=True,
    )
    return False


def wait_for_race_start(data, timeout_s=10.0):
    """Wait until sim reports race_start_boot_time_ms (race countdown scheduled)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        race = data.get("race_status") or {}
        if race.get("race_start_boot_time_ms", -1) >= 0:
            return True
        time.sleep(PREFLIGHT_POLL_S)
    return False


def _race_start_valid_after_track(data, race, race_start):
    """race_start must change from track burst or be scheduled future GO."""
    sim_boot = race.get("sim_boot_time_ms", 0)
    track_race_start = data.get("_track_race_start_ms")
    if track_race_start is not None and race_start != track_race_start:
        return True
    if race_start - sim_boot > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        return True
    return False


def wait_for_fresh_race_start(data, timeout_s=15.0, controller=None):
    """Wait for race_start tied to track burst and GO not yet passed."""
    deadline = time.time() + timeout_s
    last_reject_log = 0.0
    while time.time() < deadline:
        preflight_keep_safe(controller)
        race = data.get("race_status") or {}
        race_start = race.get("race_start_boot_time_ms", -1)
        if race_start < 0:
            now = time.time()
            if now - last_reject_log >= 0.5:
                last_reject_log = now
                dbg("fresh_race_start", "reject=no_race_start", throttle_s=0.5)
            time.sleep(PREFLIGHT_POLL_S)
            continue
        if race_go_already_passed({"race_status": race}):
            now = time.time()
            if now - last_reject_log >= 0.5:
                last_reject_log = now
                dbg(
                    "fresh_race_start",
                    f"reject=go_already_passed race_start={race_start} "
                    f"sim_boot={race.get('sim_boot_time_ms', 0)}",
                    throttle_s=0.5,
                )
            time.sleep(PREFLIGHT_POLL_S)
            continue
        if not _race_start_valid_after_track(data, race, race_start):
            now = time.time()
            if now - last_reject_log >= 0.5:
                last_reject_log = now
                sim_boot = race.get("sim_boot_time_ms", 0)
                delta = race_start - sim_boot
                dbg(
                    "fresh_race_start",
                    f"reject=stale_same_as_track race_start={race_start} "
                    f"track_race_start={data.get('_track_race_start_ms')} delta={delta}",
                    throttle_s=0.5,
                )
            time.sleep(PREFLIGHT_POLL_S)
            continue
        print(
            "[RACE] fresh race_start="
            f"{race_start} track_sim_boot={data.get('_track_sim_boot_ms')} "
            f"track_race_start={data.get('_track_race_start_ms')} "
            f"baseline={data.get('_preflight_race_start_baseline')}",
            flush=True,
        )
        dbg_now("fresh_race_start", f"accept {motion_snapshot(data)}")
        return True
    return False


def wait_for_race_go(
    data,
    timeout_s=RACE_GO_TIMEOUT_S,
    armed_sim_boot_ms=None,
    is_restart=None,
    controller=None,
):
    print("Waiting for race go (countdown -> 0)...", flush=True)
    deadline = time.time() + timeout_s
    del is_restart
    latch = RaceGoLatch()
    latch.reset_for_arm(armed_sim_boot_ms, is_restart=False)
    countdown_witnessed = False
    last_phase_log = 0.0
    last_progress_log = 0.0

    while time.time() < deadline:
        preflight_keep_safe(controller)
        race = data.get("race_status") or {}
        sim_boot = race.get("sim_boot_time_ms", 0)
        race_start = race.get("race_start_boot_time_ms", -1)

        if not countdown_witnessed:
            if latch.go_boot_ms is None and race_start >= 0:
                go_boot_try, branch_try = latch_race_go_boot_ms(sim_boot, race_start)
                if branch_try == "at_go":
                    dbg(
                        "race_go_p1",
                        f"reject_at_go sim_boot={sim_boot} race_start={race_start} "
                        f"go_boot={go_boot_try}",
                        throttle_s=0.5,
                    )
            poll_race_go(data, latch, allow_at_go=False)
            if latch.go_boot_ms is not None and latch.branch in (
                "scheduled",
                "countdown",
            ):
                if not race_go_allowed(data, go_boot_ms=latch.go_boot_ms):
                    countdown_witnessed = True
                    print(
                        "[RACE] countdown latched "
                        f"branch={latch.branch} go_boot={latch.go_boot_ms}ms",
                        flush=True,
                    )
                    dbg_now("race_go_p1", f"witnessed branch={latch.branch} go_boot={latch.go_boot_ms}")
            now = time.monotonic()
            if now - last_phase_log >= 0.5:
                last_phase_log = now
                dbg(
                    "race_go_p1",
                    f"sim_boot={sim_boot} go_boot={latch.go_boot_ms} "
                    f"branch={latch.branch} witnessed={countdown_witnessed}",
                    throttle_s=0.0,
                )
            time.sleep(RACE_GO_POLL_S)
            continue

        allowed, go_boot_ms = poll_race_go(
            data, latch, allow_at_go=True, margin_ms=GO_POST_MARGIN_MS
        )
        now = time.monotonic()
        if now - last_progress_log >= 0.5:
            last_progress_log = now
            remaining = (go_boot_ms or 0) + GO_POST_MARGIN_MS - sim_boot
            dbg(
                "race_go_p2",
                f"sim_boot={sim_boot} go_boot={go_boot_ms} "
                f"remaining_ms={remaining} allowed={allowed}",
                throttle_s=0.0,
            )
        if allowed:
            data["_latched_go_boot_ms"] = go_boot_ms
            race = data.get("race_status") or {}
            sim_boot = race.get("sim_boot_time_ms", 0)
            race_start = race.get("race_start_boot_time_ms", 0)
            delta = race_start - sim_boot
            print(
                "Race go! "
                f"sim_boot={sim_boot}ms "
                f"race_start={race_start}ms "
                f"go_boot={go_boot_ms}ms "
                f"margin={GO_POST_MARGIN_MS}ms "
                f"delta={delta}ms "
                f"track_sim_boot={data.get('_track_sim_boot_ms', '?')} "
                f"branch={latch.branch}",
                flush=True,
            )
            return True
        time.sleep(RACE_GO_POLL_S)

    print("Race go timeout: race never started", flush=True)
    return False


def wait_for_visual_go(data, timeout_s=15.0, controller=None):
    """Wait for on-screen countdown seen then cleared (vision)."""
    if not auto_go_vision_enabled():
        return True

    from simulator.countdown_detector import (
        countdown_gate_cleared,
        countdown_gate_saw,
        countdown_gate_state,
        countdown_roi_metrics,
    )

    print("Waiting for visual countdown clear...", flush=True)
    deadline = time.time() + timeout_s
    see_deadline = time.time() + VISUAL_GO_SEE_TIMEOUT_S
    last_gate_state = "idle"
    last_roi_log = 0.0

    while time.time() < deadline:
        preflight_keep_safe(controller)
        state = countdown_gate_state(data)
        if state != last_gate_state:
            dbg_now("visual_go", f"state {last_gate_state}->{state}")
            last_gate_state = state

        frame = data.get("frame")
        img = frame.get("img") if frame else None
        if img is not None:
            now = time.monotonic()
            if now - last_roi_log >= 0.5:
                last_roi_log = now
                visible, bright_frac, edge_density = countdown_roi_metrics(img)
                dbg(
                    "visual_roi",
                    f"visible={visible} bright={bright_frac:.3f} "
                    f"edge={edge_density:.3f} state={state}",
                    throttle_s=0.0,
                )

        if countdown_gate_cleared(data):
            print("[RACE] visual GO (countdown cleared)", flush=True)
            dbg_now("visual_go", f"cleared {motion_snapshot(data)}")
            return True
        if time.time() > see_deadline and not countdown_gate_saw(data):
            print(
                "[RACE] visual GO fallback (no countdown seen on camera)",
                flush=True,
            )
            dbg_now("visual_go", f"fallback no_countdown_seen {motion_snapshot(data)}")
            return True
        time.sleep(PREFLIGHT_POLL_S)

    print("Visual GO timeout", flush=True)
    dbg_now("visual_go", "timeout")
    return False


def wait_for_ready(data, timeout_s=PREFLIGHT_TIMEOUT_S):
    return wait_for_track(data, timeout_s=timeout_s)


def probe_udp_port(host="0.0.0.0", port=5600):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
        return True, None
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()


def run_preflight_checks(vision_port=5600):
    ok, err = probe_udp_port(port=vision_port)
    if ok:
        print(f"Preflight OK: UDP port {vision_port} available", flush=True)
        return True
    print(f"Preflight FAIL: UDP port {vision_port} unavailable ({err})", flush=True)
    return False
