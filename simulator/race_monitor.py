import os

from simulator.preflight import race_finished

GATE1_TIMEOUT_S = float(os.environ.get("GATE1_TIMEOUT_S", "15"))
GATE1_MIN_ELAPSED_S = float(os.environ.get("GATE1_MIN_ELAPSED_S", "8"))
GATE_PROGRESS_TIMEOUT_S = float(os.environ.get("GATE_PROGRESS_TIMEOUT_S", "15"))
SIM_RESET_WAIT_S = float(os.environ.get("SIM_RESET_WAIT_S", "5"))
GATE1_WATCH_INTERVAL_S = float(os.environ.get("GATE1_WATCH_INTERVAL_S", "5"))
DEFAULT_GATE_COUNT = 6


def gate_count(data):
    if data.get("gate_count"):
        return int(data["gate_count"])
    tg = data.get("track_gates")
    if tg:
        return len(tg)
    race = data.get("race_status")
    if race and race.get("race_start_boot_time_ms", -1) >= 0:
        return DEFAULT_GATE_COUNT
    return 0


def passed_first_gate(data):
    return int(data.get("active_gate_index", 0) or 0) >= 1


def course_complete(data):
    if race_finished(data):
        return True
    n = gate_count(data)
    if n <= 0:
        return False
    return int(data.get("active_gate_index", 0) or 0) >= n


def _quat_to_R(w, x, y, z):
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def signed_dist_gate0(data):
    """Signed distance to gate-0 plane (+ = past gate along through-axis)."""
    if data.get("track_positions_valid") is False:
        return None
    odometry = data.get("odometry")
    gates = data.get("track_gates") or []
    if not odometry or not gates:
        return None

    gate = gates[0]
    pos = gate.get("position_ned")
    orient = gate.get("orientation_ned")
    if not pos or not orient or len(pos) < 3 or len(orient) < 4:
        return None

    w, x, y, z = orient[0], orient[1], orient[2], orient[3]
    R = _quat_to_R(w, x, y, z)
    n = (R[0][0], R[1][0], R[2][0])

    px = odometry.get("x", 0.0)
    py = odometry.get("y", 0.0)
    pz = odometry.get("z", 0.0)
    gx, gy, gz = pos[0], pos[1], pos[2]
    dx, dy, dz = px - gx, py - gy, pz - gz
    return n[0] * dx + n[1] * dy + n[2] * dz


def passed_gate0_plane(data, min_signed_m=0.5):
    signed = signed_dist_gate0(data)
    return signed is not None and signed > min_signed_m


def gate1_fail(data, elapsed_s, pilot_gates_passed):
    if passed_first_gate(data):
        return False
    if elapsed_s <= 0:
        return False
    if pilot_gates_passed > 0:
        return False
    if elapsed_s > GATE1_TIMEOUT_S:
        return True
    return False


def gate1_watch_line(data, elapsed_s, pilot_gates_passed):
    active = int(data.get("active_gate_index", 0) or 0)
    return (
        f"[RACE] gate1_watch active={active} "
        f"elapsed={elapsed_s:.0f}s pilot_passed={pilot_gates_passed}"
    )


def gate_progress_stall(data, last_active, elapsed_since_advance_s):
    if course_complete(data):
        return False
    if elapsed_since_advance_s < GATE_PROGRESS_TIMEOUT_S:
        return False
    active = int(data.get("active_gate_index", 0) or 0)
    return active <= last_active


def gate_progress_watch_line(data, last_active, elapsed_s):
    active = int(data.get("active_gate_index", 0) or 0)
    return (
        f"[RACE] gate_progress_watch active={active} "
        f"last_active={last_active} elapsed={elapsed_s:.0f}s"
    )
