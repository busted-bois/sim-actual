import math
import os

# AGP camera intrinsics (VADR-TS-002 §3.8; see elodin-sys/ai-grand-prix context/agp-spec-reference.md)
CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_FX = 320.0
CAM_FY = 320.0
CAM_CX = 320.0
CAM_CY = 180.0
CAM_TILT_UP_DEG = 20.0
CAM_FOV_VERT_DEG = 2.0 * math.degrees(math.atan(CAM_CY / CAM_FY))

# MAVLink command rate (spec §4.4: client commands must be < 100 Hz)
CONTROL_HZ = 50

# Tracking (LOCAL_NED_BLEND raised after VQ1 log analysis — IMU z drifts without NED)
LOCAL_NED_BLEND = 0.38
ATTITUDE_BLEND = 0.2
MIN_IMU_SAMPLES_FOR_HEALTHY = 2
TRACKING_LOG_HZ = 10
TRACKING_MAX_IMU_GAP_S = 0.1

# Gate geometry (AGP spec §3.7)
GATE_OUTER_MM = 2700.0
GATE_INNER_MM = 1500.0
GATE_DEPTH_MM = 260.0
GATE_WIDTH_MM = GATE_OUTER_MM

# Pilot flight
HOVER_THRUST = 0.5
CRUISE_THRUST = 0.55
CRUISE_PITCH_RATE = -0.2
V_MAX_BODY = 9.0
V_MIN_BODY = 1.5
V_TURN_SLOWDOWN = 0.62
COLLISION_THRUST = 0.4
COLLISION_HOLD_S = 2.0

# Racing planner (Tier 2 TOGT-lite)
GATE_INNER_M = GATE_INNER_MM / 1000.0
GATE_CORNER_CUT = 0.55
PURE_PURSUIT_LOOKAHEAD_M = 4.0

# IBVS / PN (Tier 1)
IBVS_YAW_GAIN = math.radians(45.0)
IBVS_PITCH_GAIN = math.radians(18.0)
IBVS_FORWARD_GAIN = 8.0
PN_GAIN = 3.5
PN_BLEND = 0.45

# Velocity MPC (Tier 2)
MPC_HORIZON_S = 0.12
MPC_KP_V = 1.4
MPC_KD_V = 0.35

# Gate PnP fusion (Tier 2)
PNP_BLEND = 0.25
PNP_MIN_CORNERS = 4


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_auto_reset_on_collision(override=None):
    """True when CLI passes --collision-reset, else AUTO_RESET_ON_COLLISION env, else off."""
    if override is not None:
        return override
    return _env_bool("AUTO_RESET_ON_COLLISION", False)


VISION_MAX_AGE_S = 0.5
VISION_PROXIMITY_R_FRAC = 0.08
VISION_CORRECTION_R_FRAC = 0.12
PNP_R_FRAC_CAP = 0.35
BEARING_VELOCITY_FALLBACK_DEG = 55.0
ALTITUDE_TRIM = 0.55
KP_Z = 0.28
KI_Z = 0.025
KD_Z = 0.22
ALTITUDE_VZ_CLAMP = 8.0
Z_TARGET_NED = -5.0

# Vision temporal filter + desaturated VQ1 scenes
VISION_FILTER_ALPHA = 0.35
VISION_FILTER_MAX_JUMP = 0.45
VISION_DESAT_V_THRESHOLD = 100

# Tracker-only bearing nudge (tests / optional; pilot uses IBVS in flight_control)
VISION_YAW_BLEND = 0.15
