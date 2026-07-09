import os
import socket
import time

PREFLIGHT_TIMEOUT_S = 30.0
AUTO_TRACK_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 5.0
PREFLIGHT_POLL_S = 0.1
RACE_GO_TIMEOUT_S = 45.0
# Match control loop rate so we react within one tick of GO.
RACE_GO_POLL_S = 0.004

# Sim 3-2-1 countdown length.
RACE_COUNTDOWN_MS = int(os.environ.get("RACE_COUNTDOWN_MS", "3000"))

# If race_start is this far ahead of sim_boot at latch, treat it as scheduled GO time.
COUNTDOWN_SCHEDULED_THRESHOLD_MS = 1500

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
    if race_start_boot_ms < 0:
        return None
    if is_restart:
        return restart_go_boot_ms(race_start_boot_ms)
    delta = race_start_boot_ms - sim_boot_ms
    if delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        return race_start_boot_ms
    return restart_go_boot_ms(race_start_boot_ms)


def race_go_already_passed(data, is_restart=False):
    race = data.get("race_status") or {}
    race_start = race.get("race_start_boot_time_ms", -1)
    if race_start < 0:
        return False
    go_boot = race_go_boot_ms(race.get("sim_boot_time_ms", 0), race_start, is_restart=is_restart)
    if go_boot is None:
        return False
    return race.get("sim_boot_time_ms", 0) >= go_boot


def latch_race_go_boot_ms(sim_boot_ms, race_start_boot_ms, is_restart=False):
    """Latch the sim_boot_time_ms when the on-screen timer hits 0.

    First run: race_start_boot_time_ms is the scheduled GO instant.
    Restart (sim_boot reset): race_start marks countdown start; GO is +countdown.
    """
    if race_start_boot_ms < 0:
        return None, None

    if is_restart:
        go_boot_ms = restart_go_boot_ms(race_start_boot_ms)
        delta = race_start_boot_ms - sim_boot_ms
        if sim_boot_ms >= go_boot_ms:
            branch = "restart_at_go"
        elif delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
            branch = "restart_scheduled"
        else:
            branch = "restart_countdown"
        return go_boot_ms, branch

    delta = race_start_boot_ms - sim_boot_ms
    if sim_boot_ms >= race_start_boot_ms:
        branch = "at_go"
    elif delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        branch = "scheduled"
    else:
        branch = "countdown"

    return race_start_boot_ms, branch


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

    def try_latch(self, sim_boot_ms, race_start_boot_ms):
        if self.go_boot_ms is not None:
            return self.go_boot_ms, self.branch

        if race_start_boot_ms < 0:
            return None, None

        if self.armed_sim_boot_ms is not None and sim_boot_ms <= self.armed_sim_boot_ms:
            return None, None

        self.go_boot_ms, self.branch = latch_race_go_boot_ms(
            sim_boot_ms,
            race_start_boot_ms,
            is_restart=self.is_restart,
        )
        return self.go_boot_ms, self.branch


def poll_race_go(data, latch):
    """One wait_for_race_go iteration. True when sim_boot >= latched GO (timer at 0)."""
    race = data.get("race_status") or {}
    race_start = race.get("race_start_boot_time_ms", -1)
    sim_boot = race.get("sim_boot_time_ms", 0)

    latch.try_latch(sim_boot, race_start)

    if latch.go_boot_ms is None:
        return False, None

    return race_go_allowed(
        data,
        go_boot_ms=latch.go_boot_ms,
        is_restart=latch.is_restart,
    ), latch.go_boot_ms


def race_finished(data):
    """True when sim reports race_finish_time_ns >= 0 (lap complete)."""
    race = data.get("race_status")
    if race is None:
        return False
    finish_ns = race.get("race_finish_time_ns", -1)
    return finish_ns is not None and finish_ns >= 0


def race_go_allowed(data, go_boot_ms=None, is_restart=False):
    """True when sim reports the race has started (on-screen GO! / timer at 0)."""
    race = data.get("race_status")
    if race is None:
        return False

    race_start = race.get("race_start_boot_time_ms", -1)
    if race_start < 0:
        return False

    sim_boot = race.get("sim_boot_time_ms", 0)

    if is_restart:
        return sim_boot >= restart_go_boot_ms(race_start)

    if go_boot_ms is None:
        return False

    return sim_boot >= go_boot_ms


def wait_for_track(data, timeout_s=PREFLIGHT_TIMEOUT_S):
    print("Preflight: waiting for track_gates (click Race in FlightSim)...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if data.get("track_gates"):
            print("Preflight OK: track_gates loaded", flush=True)
            return True
        time.sleep(PREFLIGHT_POLL_S)
    print("Preflight timeout: track_gates not ready", flush=True)
    return False


def vision_ready(data) -> bool:
    cam = data.get("camera")
    return cam is not None and cam.get("received_at") is not None


def imu_ready(data) -> bool:
    return data.get("imu") is not None


def connect_ready(data) -> bool:
    return imu_ready(data) or vision_ready(data)


def _cancel_requested(cancel) -> bool:
    return cancel is not None and cancel.cancelled()


def wait_for_connect(data, timeout_s=CONNECT_TIMEOUT_S):
    """VQ2 startup: IMU or vision frame (odometry not required)."""
    print("Waiting for IMU or vision...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if connect_ready(data):
            print(
                f"Connect OK: imu={imu_ready(data)} vision={vision_ready(data)}",
                flush=True,
            )
            return True
        time.sleep(PREFLIGHT_POLL_S)
    print("Connect timeout: no IMU or vision yet", flush=True)
    return False


def wait_for_session_ready(data, timeout_s=AUTO_TRACK_TIMEOUT_S, cancel=None):
    """User entered VQ2 R2 SUBMISSION or TRAINING flight session (vision streaming)."""
    print(
        "Preflight: waiting for vision (enter AI-GP VIRTUAL QUALIFIER R2 — "
        "SUBMISSION or TRAINING)...",
        flush=True,
    )
    print("  Ctrl+C cancels automation", flush=True)
    deadline = time.time() + timeout_s
    last_status = 0.0
    while time.time() < deadline:
        if _cancel_requested(cancel):
            return False
        if vision_ready(data):
            print("Preflight OK: vision streaming", flush=True)
            return True
        now = time.time()
        if now - last_status >= 5.0:
            print(
                f"[RACE] waiting... vision={vision_ready(data)} "
                "(load SUBMISSION or TRAINING flight session)",
                flush=True,
            )
            last_status = now
        time.sleep(PREFLIGHT_POLL_S)
    print("Preflight timeout: no vision — enter a VQ2 R2 flight session", flush=True)
    return False


def wait_for_race_status(data, timeout_s=AUTO_TRACK_TIMEOUT_S, cancel=None):
    """Wait for race_status telemetry (may arrive before Race is clicked)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _cancel_requested(cancel):
            return False
        if data.get("race_status") is not None:
            return True
        time.sleep(PREFLIGHT_POLL_S)
    return False


def wait_for_vq2_ready(data, timeout_s=AUTO_TRACK_TIMEOUT_S):
    """Legacy combined wait — prefer wait_for_session_ready + wait_for_race_status."""
    if not wait_for_session_ready(data, timeout_s=timeout_s):
        return False
    return wait_for_race_status(data, timeout_s=10.0)


def wait_for_fresh_track(data, timeout_s=AUTO_TRACK_TIMEOUT_S):
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
        if data.get("track_gates"):
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


def wait_for_race_start(data, timeout_s=10.0, cancel=None):
    """Wait until sim reports race_start_boot_time_ms (race countdown scheduled)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _cancel_requested(cancel):
            return False
        race = data.get("race_status") or {}
        if race.get("race_start_boot_time_ms", -1) >= 0:
            return True
        time.sleep(PREFLIGHT_POLL_S)
    return False


def _race_start_valid_after_baseline(data, race, race_start):
    """race_start is fresh if the sim rebooted (sim_boot reset small), it's a
    scheduled future GO, or it differs from the pre-reset baseline."""
    sim_boot = race.get("sim_boot_time_ms", 0)
    if is_restart_arm_context(sim_boot):
        return True
    if race_start - sim_boot > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        return True
    baseline = data.get("_preflight_race_start_baseline")
    if baseline is not None and race_start != baseline:
        return True
    return False


def wait_for_fresh_race_start_vq2(data, timeout_s=30.0, cancel=None, is_restart=False):
    """Wait for new race_start after in-session reset (VQ2, no track_gates)."""
    print("[RACE] waiting for fresh race_start after reset...", flush=True)
    deadline = time.time() + timeout_s
    last_status = 0.0
    while time.time() < deadline:
        if _cancel_requested(cancel):
            return False
        race = data.get("race_status") or {}
        race_start = race.get("race_start_boot_time_ms", -1)
        if race_start < 0:
            time.sleep(PREFLIGHT_POLL_S)
            continue
        if race_go_already_passed({"race_status": race}, is_restart=is_restart):
            time.sleep(PREFLIGHT_POLL_S)
            continue
        if not _race_start_valid_after_baseline(data, race, race_start):
            now = time.time()
            if now - last_status >= 5.0:
                print(
                    f"[RACE] waiting fresh race_start... "
                    f"race_start={race_start} sim_boot={race.get('sim_boot_time_ms', 0)}",
                    flush=True,
                )
                last_status = now
            time.sleep(PREFLIGHT_POLL_S)
            continue
        sim_boot = race.get("sim_boot_time_ms", 0)
        print(
            f"[RACE] fresh race_start={race_start} sim_boot={sim_boot}",
            flush=True,
        )
        return True
    print(
        "WARNING: fresh race_start timeout — "
        "click Restart Race in FlightSim if countdown doesn't start within 30s",
        flush=True,
    )
    return False


def wait_for_race_go(
    data,
    timeout_s=RACE_GO_TIMEOUT_S,
    armed_sim_boot_ms=None,
    is_restart=None,
    cancel=None,
):
    print("Waiting for race go (countdown -> 0)...", flush=True)
    deadline = time.time() + timeout_s
    latch = RaceGoLatch()
    if is_restart is None:
        is_restart = is_restart_arm_context(armed_sim_boot_ms)
    latch.reset_for_arm(armed_sim_boot_ms, is_restart=is_restart)

    while time.time() < deadline:
        if _cancel_requested(cancel):
            return False
        allowed, go_boot_ms = poll_race_go(data, latch)
        if allowed:
            data["_latched_go_boot_ms"] = go_boot_ms
            race = data.get("race_status") or {}
            print(
                "Race go! "
                f"sim_boot={race.get('sim_boot_time_ms')}ms "
                f"race_start={race.get('race_start_boot_time_ms')}ms "
                f"go_boot={go_boot_ms}ms "
                f"branch={latch.branch}",
                flush=True,
            )
            return True
        time.sleep(RACE_GO_POLL_S)

    print("Race go timeout: race never started", flush=True)
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


def udp_port_in_use(host="127.0.0.1", port=14550):
    """True if another process already holds this UDP port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def run_preflight_checks(vision_port=5600):
    ok, err = probe_udp_port(port=vision_port)
    if ok:
        print(f"Preflight OK: UDP port {vision_port} available", flush=True)
        return True
    print(f"Preflight FAIL: UDP port {vision_port} unavailable ({err})", flush=True)
    return False
