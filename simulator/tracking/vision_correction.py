import math

from simulator.flight_config import CAM_TILT_UP_DEG, VISION_YAW_BLEND
from simulator.tracking.camera import bearing_local_yaw_rad


def apply_gate_yaw_correction(state, gate_target, pitch_up_degrees=CAM_TILT_UP_DEG):
    if not gate_target.get("detected"):
        return state

    nx = float(gate_target.get("nx", 0.0))
    ny = float(gate_target.get("ny", 0.0))
    r_frac = float(gate_target.get("r_frac", 0.0))
    bearing = bearing_local_yaw_rad(nx, ny, pitch_up_degrees)
    proximity = min(1.0, max(0.0, r_frac / 0.12))
    blend = VISION_YAW_BLEND * (0.35 + 0.65 * proximity)
    corrected_yaw = state["yaw"] + blend * bearing
    while corrected_yaw > math.pi:
        corrected_yaw -= 2.0 * math.pi
    while corrected_yaw < -math.pi:
        corrected_yaw += 2.0 * math.pi

    updated = dict(state)
    updated["yaw"] = corrected_yaw
    return updated
