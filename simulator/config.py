"""
Loads autonomy/vision/navigation/safety settings from ``settings.json`` at the
repo root, deep-merged over the defaults below.

Keeping a single source of defaults here means a partial (or missing)
``settings.json`` still produces a fully-populated config, so the rest of the
code never has to guard against missing keys.
"""

import json
import os

# --------------------------------------------------------------------------------------
# Defaults. Anything in settings.json overrides the matching key here.
# --------------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Master toggle for the autonomous pilot.
    #   enabled   - False  -> fall back to the template's plain motor control
    #   algorithm - which autonomy strategy the controller runs. "orange_gate"
    #               is the gate-following pilot; "off" behaves like enabled=False.
    "autonomy": {
        "enabled": True,
        "algorithm": "orange_gate",
    },
    # Color detection. HSV ranges are OpenCV convention: H 0-179, S/V 0-255.
    # Each entry in a *_hsv_ranges list is [[h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi]];
    # masks from every range in the list are OR-ed together (so you can span the
    # red/orange hue wrap with two bands).
    "vision": {
        "gate_hsv_ranges": [
            [[0, 90, 70], [18, 255, 255]],
            [[170, 90, 70], [180, 255, 255]],
        ],
        "path_hsv_ranges": [
            [[100, 110, 50], [130, 255, 255]],
        ],
        "min_blob_area_frac": 0.0008,  # ignore blobs smaller than this fraction of the frame
        "blur_ksize": 5,  # gaussian blur kernel before thresholding (odd, 0/1 = off)
        "morph_ksize": 5,  # open/close kernel to clean the mask (odd, 0/1 = off)
        "debug_save_every": 0,  # if >0, save an annotated frame every N frames
        "debug_dir": "debug_frames",
    },
    # Velocity/yaw command shaping for the navigator. Speeds are m/s, rates rad/s.
    "navigation": {
        "cruise_speed": 2.0,
        "approach_speed": 1.2,
        "pass_through_speed": 2.5,
        "pass_through_seconds": 0.8,
        "path_speed_frac": 0.7,
        "yaw_gain": 1.4,
        "max_yaw_rate": 1.2,
        "vertical_gain": 1.0,
        "max_vertical_speed": 1.0,
        "gate_pass_area_frac": 0.10,  # gate this big in frame -> commit to fly through
        "gate_pass_center_tol": 0.25,  # ...only if horizontally centered within this
        "search_yaw_rate": 0.5,  # spin rate while looking for the next gate
        "search_creep_speed": 0.0,  # optional slow forward creep while searching
    },
    # End-of-course / fail-safe behavior.
    "safety": {
        "require_detection_before_end": True,  # don't "finish" before we ever see anything
        "end_of_course_seconds": 3.0,  # no gate AND no path this long -> course complete
        "max_run_seconds": 0,  # hard time cap (0 = disabled)
        "end_action": "hover",  # hover | land | disarm once the course is complete
    },
}

SETTINGS_FILENAME = "settings.json"


def repo_root():
    """Directory that contains the ``simulator`` package (i.e. the repo root)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base`` (dicts only)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path=None):
    """
    Return the full config dict (defaults deep-merged with settings.json).

    A missing file is fine -> defaults are used. A malformed file is reported but
    still falls back to defaults so the pilot can always start.
    """
    if path is None:
        path = os.path.join(repo_root(), SETTINGS_FILENAME)

    if not os.path.exists(path):
        print(f"[config] {path} not found, using built-in defaults.", flush=True)
        return _deep_merge(DEFAULT_CONFIG, {})

    try:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] failed to read {path} ({exc}); using defaults.", flush=True)
        return _deep_merge(DEFAULT_CONFIG, {})

    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    print(
        f"[config] loaded {path}: autonomy.enabled={cfg['autonomy']['enabled']} "
        f"algorithm={cfg['autonomy']['algorithm']}",
        flush=True,
    )
    return cfg
