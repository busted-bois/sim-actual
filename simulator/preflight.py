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


def latch_race_go_boot_ms(sim_boot_ms, race_start_boot_ms, is_restart=False):
    """Latch GO time once when race_start becomes valid after arm.

    First run: a far-future race_start is the scheduled GO instant.
    Restart: a far-future race_start marks countdown start (GO = race_start + countdown).
    """
    if race_start_boot_ms < 0:
        return None, None

    delta = race_start_boot_ms - sim_boot_ms
    if delta > COUNTDOWN_SCHEDULED_THRESHOLD_MS:
        if is_restart:
            return race_start_boot_ms + RACE_COUNTDOWN_MS, "countdown_future"
        return race_start_boot_ms, "scheduled"

    return race_start_boot_ms + RACE_COUNTDOWN_MS, "countdown"


def arm_relative_go_boot_ms(armed_sim_boot_ms):
    if armed_sim_boot_ms is None:
        return None
    return armed_sim_boot_ms + RACE_COUNTDOWN_MS


class RaceGoLatch:
    """Shared latch state for wait_for_race_go and pilot restart countdown."""

    def __init__(self):
        self.go_boot_ms = None
        self.branch = None
        self.last_race_start = -1
        self.is_restart = False

    def reset_for_arm(self, is_restart=False):
        self.go_boot_ms = None
        self.branch = None
        self.last_race_start = -1
        self.is_restart = is_restart

    def try_latch(self, sim_boot_ms, race_start_boot_ms):
        if self.go_boot_ms is not None:
            return self.go_boot_ms, self.branch

        if race_start_boot_ms < 0:
            return None, None

        if self.last_race_start >= 0:
            return None, None

        self.go_boot_ms, self.branch = latch_race_go_boot_ms(
            sim_boot_ms, race_start_boot_ms, is_restart=self.is_restart
        )
        return self.go_boot_ms, self.branch

    def observe_race_start(self, race_start_boot_ms):
        if race_start_boot_ms >= 0:
            self.last_race_start = race_start_boot_ms


def race_go_allowed(data, go_boot_ms=None):
    """True when sim reports the race has started (countdown finished)."""
    if go_boot_ms is None:
        return False

    race = data.get("race_status")
    if race is None:
        return False

    race_start = race.get("race_start_boot_time_ms", -1)
    if race_start < 0:
        return False

    sim_boot = race.get("sim_boot_time_ms", 0)
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
    latch.reset_for_arm(is_restart=False)

    while time.time() < deadline:
        race = data.get("race_status") or {}
        race_start = race.get("race_start_boot_time_ms", -1)
        sim_boot = race.get("sim_boot_time_ms", 0)

        latch.try_latch(sim_boot, race_start)
        latch.observe_race_start(race_start)

        if race_go_allowed(data, go_boot_ms=latch.go_boot_ms):
            data["_latched_go_boot_ms"] = latch.go_boot_ms
            print(
                "Race go! "
                f"sim_boot={sim_boot}ms "
                f"race_start={race_start}ms "
                f"go_boot={latch.go_boot_ms}ms "
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
