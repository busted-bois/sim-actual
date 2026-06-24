import colorsys
import numpy as np
from dataclasses import dataclass

GATE_HEX_COLOR = "#F3390F"

HSV_TOLERANCE = 40  # wider to handle shadow/shading on gate

# Compute HSV bounds from hex color at import time
_hex_rgb = tuple(
    int(GATE_HEX_COLOR.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)
)
_hsv_norm = colorsys.rgb_to_hsv(*_hex_rgb)
_h = int(_hsv_norm[0] * 179)  # Scale to OpenCV range [0, 179]
_s = int(_hsv_norm[1] * 255)
_v = int(_hsv_norm[2] * 255)

HSV_LOWER = np.array(
    [
        [
            max(0, _h - HSV_TOLERANCE),
            max(0, _s - HSV_TOLERANCE),
            max(0, _v - HSV_TOLERANCE),
        ]
    ]
)
HSV_UPPER = np.array(
    [
        [
            min(179, _h + HSV_TOLERANCE),
            min(255, _s + HSV_TOLERANCE),
            min(255, _v + HSV_TOLERANCE),
        ]
    ]
)

MORPH_KERNEL_SIZE = 5
MORPH_ITERS = 2
MIN_CONTOUR_AREA_PX = 500
MAX_ASPECT_RATIO = 5.0
MIN_ASPECT_RATIO = 0.2
REFERENCE_GATE_WIDTH_M = 1.5
FOCAL_LENGTH_PX_INIT = None
SELF_CAL_SAMPLES = 10
YAW_KP = 0.02
LATERAL_KP = 0.5
FORWARD_BASE_SPEED_MPS = 2.0
FORWARD_GAIN_PER_AREA = 0.5
ALTITUDE_TARGET_M = 3.0
TAKEOFF_THRUST = 0.65
MAX_YAW_RATE = 1.5
DEADBAND_PX = 10
DETECTION_AGE_OUT_MS = 150
SEARCH_SWEEP_YAW_RATE = 1.0
SEARCH_SWEEP_PERIOD_S = 2.0
SEARCH_EXPAND_STEP_M = 2.0
SEARCH_MAX_EXPAND_M = 10.0
SEARCH_FORWARD_MPS = 1.0
SEARCH_SWEEPS_BEFORE_EXPAND = 3
PASS_RANGE_M = 2.0
PASS_AREA_FRAC = 0.15
LOST_FRAMES_THRESHOLD = 30
TAKEOFF_TIMEOUT_S = 10.0
DEBUG = False

# Gate vs obstacle disambiguation
GATE_CONFIDENCE_MIN_NAV = 0.35
GATE_CONFIDENCE_AMBIGUOUS = 0.55
GATE_TEMPORAL_MIN_STREAK = 2
OBSTACLE_CONFIDENCE_AVOID = 0.40
OBSTACLE_MIN_AREA_PX = 200
OBSTACLE_GRAY_THRESH = 15
OBSTACLE_BRIGHT_EXCLUDE = 80


@dataclass
class GateDetection:
    frame_id: int
    sim_time_ns: int
    centroid_x_px: float
    centroid_y_px: float
    area_px: float
    width_px: float
    height_px: float
    contour_valid: bool
    corners_px: tuple[tuple[float, float], ...] | None = None
    reproj_err_px: float | None = None
    quality: float = 0.0


@dataclass
class ObstacleDetection:
    frame_id: int
    sim_time_ns: int
    centroid_x_px: float
    centroid_y_px: float
    area_px: float
    width_px: float
    height_px: float
    confidence: float


@dataclass
class DroneState:
    pos_ned: tuple[float, float, float]
    vel_ned: tuple[float, float, float]
    yaw_rad: float
    yaw_rate: float
    time_boot_ms: int
    has_position: bool


@dataclass
class TrackGate:
    gate_id: int
    pos_ned: tuple[float, float, float]
    orient_quat: tuple[float, float, float, float]
    width_m: float
    height_m: float
