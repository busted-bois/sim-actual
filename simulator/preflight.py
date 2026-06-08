import os
import socket
import time

PREFLIGHT_TIMEOUT_S = 30.0
PREFLIGHT_POLL_S = 0.1
RACE_GO_TIMEOUT_S = 15.0
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
    return (
        armed_sim_boot_ms is not None
        and armed_sim_boot_ms < RESTART_ARM_BOOT_THRESHOLD_MS
    )


def restart_go_boot_ms(race_start_boot_ms):
    return race_start_boot_ms + RACE_COUNTDOWN_MS


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
        self.last_race_start = -1
        self.armed_sim_boot_ms = None
        self.is_restart = False

    def reset_for_arm(self, armed_sim_boot_ms=None, is_restart=False):
        self.go_boot_ms = None
        self.branch = None
        self.last_race_start = -1
        self.armed_sim_boot_ms = armed_sim_boot_ms
        self.is_restart = is_restart

    def try_latch(self, sim_boot_ms, race_start_boot_ms):
        if self.go_boot_ms is not None:
            return self.go_boot_ms, self.branch

        if race_start_boot_ms < 0:
            return None, None

        if self.last_race_start >= 0:
            return None, None

        if self.armed_sim_boot_ms is not None and sim_boot_ms <= self.armed_sim_boot_ms:
            return None, None

        self.go_boot_ms, self.branch = latch_race_go_boot_ms(
            sim_boot_ms,
            race_start_boot_ms,
            is_restart=self.is_restart,
        )
        return self.go_boot_ms, self.branch

    def observe_race_start(self, race_start_boot_ms):
        if race_start_boot_ms >= 0:
            self.last_race_start = race_start_boot_ms


def poll_race_go(data, latch):
    """One wait_for_race_go iteration. True when sim_boot >= latched GO (timer at 0)."""
    race = data.get("race_status") or {}
    race_start = race.get("race_start_boot_time_ms", -1)
    sim_boot = race.get("sim_boot_time_ms", 0)

    latch.try_latch(sim_boot, race_start)
    latch.observe_race_start(race_start)

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
    print("Preflight: waiting for track_gates...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if data.get("track_gates"):
            print("Preflight OK: track_gates loaded", flush=True)
            return True
        time.sleep(PREFLIGHT_POLL_S)
    print("Preflight timeout: track_gates not ready", flush=True)
    return False


def wait_for_race_go(data, timeout_s=RACE_GO_TIMEOUT_S, armed_sim_boot_ms=None):
    print("Waiting for race go...", flush=True)
    deadline = time.time() + timeout_s
    latch = RaceGoLatch()
    latch.reset_for_arm(armed_sim_boot_ms)

    while time.time() < deadline:
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


def run_preflight_checks(vision_port=5600):
    ok, err = probe_udp_port(port=vision_port)
    if ok:
        print(f"Preflight OK: UDP port {vision_port} available", flush=True)
        return True
    print(f"Preflight FAIL: UDP port {vision_port} unavailable ({err})", flush=True)
    return False
