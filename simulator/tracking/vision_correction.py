from simulator.flight_config import (
    CAM_TILT_UP_DEG,
    VISION_CORRECTION_R_FRAC,
    VISION_YAW_BLEND,
)
from simulator.math_util import normalize_angle
from simulator.tracking.camera import bearing_local_yaw_rad


def apply_gate_yaw_correction(state, gate_target, pitch_up_degrees=CAM_TILT_UP_DEG):
    if not gate_target.get("detected"):
        return state

    nx = float(gate_target.get("nx", 0.0))
    ny = float(gate_target.get("ny", 0.0))
    r_frac = float(gate_target.get("r_frac", 0.0))
    bearing = bearing_local_yaw_rad(nx, ny, pitch_up_degrees)
    proximity = min(1.0, max(0.0, r_frac / VISION_CORRECTION_R_FRAC))
    blend = VISION_YAW_BLEND * (0.35 + 0.65 * proximity)
    corrected_yaw = normalize_angle(state["yaw"] + blend * bearing)

    updated = dict(state)
    updated["yaw"] = corrected_yaw
    return updated
