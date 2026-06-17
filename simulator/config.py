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

# Color/vision detector (VisionRX). The Round-2 approach navigates from telemetry
# (track_gates + odometry), NOT color — and the per-frame OpenCV work competes with the
# 250 Hz control loop for the GIL. Keep OFF. Set True only to revisit the archived
# color-detection baseline.
ENABLE_VISION = False


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


# Known Round-1 qualifier track (NED, relative to the start pad). The sim broadcasts the
# live gate table only ONCE at race start, so a pilot that connects mid-race never receives
# it. The course is fixed and these positions are identical across every clean run, so we
# fall back to them when live `track_gates` is absent (see pilot.py). Same dict shape that
# mavlink_rx builds for `track_gates`.
_R1_ORIENT = (0.7071067094802856, 0.0, -8.657315930804543e-08, 0.7071067690849304)
_R1_W = 2.7200000286102295
ROUND1_GATES = [
    {
        "position_ned": (-23.2979679107666, -0.39990234375, -0.03195800632238388),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
    {
        "position_ned": (-46.89374923706055, -2.499990224838257, 5.068041801452637),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
    {
        "position_ned": (-74.59375, 1.2000097036361694, 13.668041229248047),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
    {
        "position_ned": (-111.49374389648438, -5.099989891052246, 24.56804084777832),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
    {
        "position_ned": (-135.49374389648438, -0.7999902367591858, 25.355653762817383),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
    {
        "position_ned": (-159.19374084472656, -4.399990081787109, 25.968040466308594),
        "orientation_ned": _R1_ORIENT,
        "width": _R1_W,
        "height": _R1_W,
    },
]
