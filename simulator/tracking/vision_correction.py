import math

from simulator.tracking.camera import bearing_local_yaw_rad

VISION_YAW_BLEND = 0.15


def apply_gate_yaw_correction(state, gate_target, pitch_up_degrees=20.0):
    if not gate_target.get("detected"):
        return state

    nx = float(gate_target.get("nx", 0.0))
    ny = float(gate_target.get("ny", 0.0))
    bearing = bearing_local_yaw_rad(nx, ny, pitch_up_degrees)
    corrected_yaw = state["yaw"] + VISION_YAW_BLEND * bearing
    while corrected_yaw > math.pi:
        corrected_yaw -= 2.0 * math.pi
    while corrected_yaw < -math.pi:
        corrected_yaw += 2.0 * math.pi

    updated = dict(state)
    updated["yaw"] = corrected_yaw
    return updated
