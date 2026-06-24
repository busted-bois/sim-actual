"""A/B pilot profiles — switch dynamics and Jacobian vision via env vars.



Presets (PILOT_AB):

  main            — main-branch thrust/PID, pitch-rate, obstacles (no Jacobian)

  branch          — main flight + active_gate telemetry (no Jacobian)

  jacobian        — main flight + yaw-only Jacobian (explicit opt-in via preset)

  main-jacobian   — alias for jacobian

  branch-legacy   — old low-thrust angle-P branch (A/B only; misses gate 1)



Or set independently:

  PILOT_DYNAMICS=main|branch|branch-legacy

  PILOT_JACOBIAN=0|1

"""

from __future__ import annotations


import math

import os

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PilotProfile:
    label: str

    hover_thrust: float

    cruise_thrust: float

    collision_thrust: float

    altitude_trim: float

    kp_z: float

    ki_z: float

    kd_z: float

    cruise_pitch: float

    pitch_is_rate: bool

    use_angle_p: bool

    k_att: float

    sign_roll: float

    sign_pitch: float

    sign_yaw: float

    rate_clip: float

    yaw_clip: float

    vision_center_deadband: float

    vision_proximity_r_frac: float

    stabilize_hold_s: float

    telemetry_yaw_gain: float

    telemetry_proximity_m: float

    telemetry_align_rad: float | None

    post_gate_hover_s: float

    search_forward_pitch: float

    obstacle_avoidance: bool

    obstacle_clear_zone: float

    telemetry_use_active_gate: bool

    thrust_min: float

    thrust_max: float

    jacobian_blend: float

    advance_near_hover: bool


_MAIN = PilotProfile(
    label="main",
    hover_thrust=0.5,
    cruise_thrust=0.55,
    collision_thrust=0.4,
    altitude_trim=0.55,
    kp_z=0.15,
    ki_z=0.01,
    kd_z=0.20,
    cruise_pitch=-0.2,
    pitch_is_rate=True,
    use_angle_p=False,
    k_att=0.6,
    sign_roll=-1.0,
    sign_pitch=1.0,
    sign_yaw=-1.0,
    rate_clip=0.30,
    yaw_clip=0.5,
    vision_center_deadband=0.30,
    vision_proximity_r_frac=0.10,
    stabilize_hold_s=0.1,
    telemetry_yaw_gain=1.0,
    telemetry_proximity_m=3.0,
    telemetry_align_rad=None,
    post_gate_hover_s=2.5,
    search_forward_pitch=-0.04,
    obstacle_avoidance=True,
    obstacle_clear_zone=0.25,
    telemetry_use_active_gate=False,
    thrust_min=0.0,
    thrust_max=1.0,
    jacobian_blend=0.0,
    advance_near_hover=True,
)


# Main flight tuning + sim active_gate telemetry (Jacobian-Vision branch extras).

_BRANCH = replace(
    _MAIN,
    label="branch",
    telemetry_yaw_gain=1.2,
    telemetry_proximity_m=2.0,
    telemetry_align_rad=0.35,
    telemetry_use_active_gate=True,
)


# Pre-A/B low-thrust angle-P pilot — kept for regression comparison only.

_BRANCH_LEGACY = PilotProfile(
    label="branch-legacy",
    hover_thrust=0.27,
    cruise_thrust=0.36,
    collision_thrust=0.22,
    altitude_trim=0.27,
    kp_z=0.025,
    ki_z=0.0,
    kd_z=0.030,
    cruise_pitch=-0.18,
    pitch_is_rate=False,
    use_angle_p=True,
    k_att=0.6,
    sign_roll=-1.0,
    sign_pitch=1.0,
    sign_yaw=-1.0,
    rate_clip=0.30,
    yaw_clip=0.5,
    vision_center_deadband=0.20,
    vision_proximity_r_frac=0.08,
    stabilize_hold_s=0.05,
    telemetry_yaw_gain=1.2,
    telemetry_proximity_m=2.0,
    telemetry_align_rad=0.35,
    post_gate_hover_s=0.8,
    search_forward_pitch=-0.04,
    obstacle_avoidance=False,
    obstacle_clear_zone=0.25,
    telemetry_use_active_gate=True,
    thrust_min=0.18,
    thrust_max=0.5,
    jacobian_blend=0.0,
    advance_near_hover=False,
)


_PROFILES: dict[str, PilotProfile] = {
    "main": _MAIN,
    "a": _MAIN,
    "branch": _BRANCH,
    "b": _BRANCH,
    "branch-legacy": _BRANCH_LEGACY,
}


_PRESETS: dict[str, tuple[str, bool]] = {
    "main": ("main", False),
    "branch": ("branch", False),
    "jacobian": ("main", True),
    "main-jacobian": ("main", True),
    "branch-legacy": ("branch-legacy", False),
}


def load_profile() -> PilotProfile:

    ab = os.environ.get("PILOT_AB", "").strip().lower()

    if ab:
        if ab not in _PRESETS:
            known = ", ".join(sorted(_PRESETS))

            raise ValueError(f"Unknown PILOT_AB={ab!r}; use one of: {known}")

        dyn_key, use_jacobian = _PRESETS[ab]

    else:
        dyn_key = os.environ.get("PILOT_DYNAMICS", "main").strip().lower()

        jac_raw = os.environ.get("PILOT_JACOBIAN", "0").strip().lower()

        use_jacobian = jac_raw not in ("0", "false", "off", "no")

    if dyn_key not in _PROFILES:
        raise ValueError(
            f"Unknown PILOT_DYNAMICS={dyn_key!r}; use main, branch, or branch-legacy"
        )

    base = _PROFILES[dyn_key]

    blend = 0.4 if use_jacobian else 0.0

    tag = base.label + ("+jacobian" if use_jacobian else "")

    fields = {
        f.name: getattr(base, f.name)
        for f in base.__dataclass_fields__.values()
        if f.name not in ("label", "jacobian_blend")
    }

    return PilotProfile(**fields, label=tag, jacobian_blend=blend)


VISION_YAW_GAIN = math.radians(40)

VISION_MAX_AGE_S = 0.5

VISION_VY_GAIN = 6.0

VISION_MAX_ALT_ADJUST = 2.0

FLYTHROUGH_MIN_PEAK_R_FRAC = 0.18

FLYTHROUGH_DROP_RATIO = 0.55

Z_TARGET_NED = -5.0

SEARCH_SWEEP_YAW_RATE = 0.8

SEARCH_SWEEP_PERIOD_S = 2.0

SEARCH_WARMUP_S = 1.5

PASSED_GATE_NEAR_M = 3.0

PASSED_GATE_ANGLE_RAD = math.radians(45)

CONTROL_DT_S = 1 / 250
