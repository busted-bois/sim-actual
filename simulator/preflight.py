import time



PREFLIGHT_TIMEOUT_S = 30.0

PREFLIGHT_POLL_S = 0.1

RACE_COUNTDOWN_S = 3.0





def ensure_countdown_fallback(data):

    if "_fallback_race_go_at" in data:

        return

    data["_fallback_race_go_at"] = time.monotonic() + RACE_COUNTDOWN_S





def race_go_allowed(data):

    race = data.get("race_status")

    if race is not None:

        race_start = race.get("race_start_boot_time_ms", -1)

        if race_start >= 0:

            sim_boot = race.get("sim_boot_time_ms", 0)

            return sim_boot >= race_start

        return False



    go_at = data.get("_fallback_race_go_at")

    if go_at is None:

        return False

    return time.monotonic() >= go_at





def wait_for_ready(data, timeout_s=PREFLIGHT_TIMEOUT_S):

    print("Preflight: waiting for armed + track_gates...", flush=True)

    deadline = time.time() + timeout_s

    while time.time() < deadline:

        if data.get("armed") and data.get("track_gates"):

            print("Preflight OK: armed and track_gates loaded", flush=True)

            return True

        time.sleep(PREFLIGHT_POLL_S)

    print("Preflight timeout: armed or track_gates not ready", flush=True)

    return False

