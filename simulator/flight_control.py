"""Tier 1 IBVS/PN and Tier 2 velocity profiling + simplified MPC."""

from __future__ import annotations

import math

from simulator.flight_config import (
    IBVS_FORWARD_GAIN,
    IBVS_PITCH_GAIN,
    IBVS_YAW_GAIN,
    MPC_HORIZON_S,
    MPC_KP_V,
    MPC_KD_V,
    PN_BLEND,
    PN_GAIN,
    V_MAX_BODY,
    V_MIN_BODY,
    VISION_PROXIMITY_R_FRAC,
    V_TURN_SLOWDOWN,
)
from simulator.math_util import clamp
from simulator.navigation import bearing_error_from_pose


def perception_aware_speed(bearing_err, gate_visible, r_frac=None):
    """Tier 1: slow for turns / lost vision, fast on straights."""
    alignment = max(0.0, math.cos(bearing_err))
    turn_scale = V_TURN_SLOWDOWN + (1.0 - V_TURN_SLOWDOWN) * alignment
    vision_scale = 1.0 if gate_visible else 0.65
    proximity_scale = 1.0
    if r_frac is not None and r_frac > VISION_PROXIMITY_R_FRAC:
        proximity_scale = clamp(r_frac / VISION_PROXIMITY_R_FRAC, 0.45, 1.0)
    return clamp(
        V_MAX_BODY * turn_scale * vision_scale * proximity_scale, V_MIN_BODY, V_MAX_BODY
    )


def ibvs_body_rates(gate_target, speed_body):
    """Tier 1 IBVS: map image offsets to pitch/yaw rates and forward speed."""
    nx = float(gate_target.get("nx", 0.0))
    ny = float(gate_target.get("ny", 0.0))
    r_frac = float(gate_target.get("r_frac", 0.0))
    yaw_rate = clamp(IBVS_YAW_GAIN * nx, -2.5, 2.5)
    pitch_rate = clamp(-IBVS_PITCH_GAIN * ny, -1.2, 1.2)
    forward = clamp(
        speed_body * (0.55 + 0.45 * max(0.0, 1.0 - abs(nx))), V_MIN_BODY, speed_body
    )
    if r_frac > 0.02:
        forward = clamp(IBVS_FORWARD_GAIN * r_frac, forward, speed_body)
    return {
        "yaw_rate": yaw_rate,
        "pitch_rate": pitch_rate,
        "forward_speed": forward,
    }


def proportional_navigation_yaw(nx, nx_prev, dt_s, forward_speed):
    """Tier 1 PN on horizontal line-of-sight rate."""
    if nx_prev is None or dt_s <= 0.0:
        return IBVS_YAW_GAIN * nx
    los_rate = (nx - nx_prev) / dt_s
    return clamp(PN_GAIN * forward_speed * los_rate, -2.5, 2.5)


def blended_lateral_command(gate_target, nx_prev, dt_s, bearing_err, forward_speed):
    """Blend IBVS, PN, and map bearing into body yaw rate."""
    ibvs = ibvs_body_rates(gate_target, forward_speed)
    pn_yaw = proportional_navigation_yaw(
        float(gate_target.get("nx", 0.0)), nx_prev, dt_s, forward_speed
    )
    track_yaw = clamp(1.2 * bearing_err, -2.5, 2.5)
    yaw_rate = (
        (1.0 - PN_BLEND) * ibvs["yaw_rate"] + PN_BLEND * pn_yaw + 0.35 * track_yaw
    )
    return clamp(yaw_rate, -2.5, 2.5), ibvs["pitch_rate"], ibvs["forward_speed"]


def world_error_to_body(err_x, err_y, yaw):
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    body_x = cy * err_x + sy * err_y
    body_y = -sy * err_x + cy * err_y
    return body_x, body_y


def velocity_mpc_body(pose, target_ned, target_speed):
    """Tier 2: short-horizon velocity command toward pursuit target."""
    err_x = target_ned[0] - pose["x"]
    err_y = target_ned[1] - pose["y"]
    body_x, body_y = world_error_to_body(err_x, err_y, pose["yaw"])

    horizon = max(MPC_HORIZON_S, 1e-3)
    desired_vx = body_x / horizon
    desired_vy = body_y / horizon
    desired_speed = math.hypot(desired_vx, desired_vy)
    if desired_speed > 1e-6:
        scale = min(1.0, target_speed / desired_speed)
        desired_vx *= scale
        desired_vy *= scale
    else:
        desired_vx = target_speed

    vx = pose.get("vx", 0.0)
    vy = pose.get("vy", 0.0)
    cvx, cvy = world_error_to_body(vx, vy, pose["yaw"])
    cmd_vx = MPC_KP_V * (desired_vx - cvx) + cvx + MPC_KD_V * (desired_vx - cvx)
    cmd_vy = MPC_KP_V * (desired_vy - cvy) + cvy + MPC_KD_V * (desired_vy - cvy)
    speed = math.hypot(cmd_vx, cmd_vy)
    if speed > target_speed:
        scale = target_speed / speed
        cmd_vx *= scale
        cmd_vy *= scale
    return cmd_vx, cmd_vy


def racing_command(
    pose,
    target_ned,
    gate,
    gate_target,
    nx_prev,
    dt_s,
):
    """Combined Tier 1 + Tier 2 body-frame velocity command."""
    bearing_err = bearing_error_from_pose(pose["x"], pose["y"], pose["yaw"], gate)
    gate_visible = bool(gate_target and gate_target.get("detected"))
    r_frac = gate_target.get("r_frac") if gate_target else None
    target_speed = perception_aware_speed(bearing_err, gate_visible, r_frac)

    if gate_visible:
        _, _, forward_hint = blended_lateral_command(
            gate_target, nx_prev, dt_s, bearing_err, target_speed
        )
        target_speed = max(target_speed, forward_hint)

    cmd_vx, cmd_vy = velocity_mpc_body(pose, target_ned, target_speed)
    return {
        "vx": cmd_vx,
        "vy": cmd_vy,
        "target_speed": target_speed,
        "bearing_err": bearing_err,
    }


def attitude_fallback_command(bearing_err, gate_target, nx_prev, dt_s, thrust):
    """Tier 1 attitude fallback when velocity mode is unsafe."""
    if gate_target and gate_target.get("detected"):
        yaw_rate, pitch_rate, _ = blended_lateral_command(
            gate_target, nx_prev, dt_s, bearing_err, V_MIN_BODY
        )
    else:
        yaw_rate = clamp(1.2 * bearing_err, -2.5, 2.5)
        pitch_rate = -0.15 * max(0.0, 1.0 - abs(bearing_err) / math.pi)
    alignment = max(0.0, 1.0 - abs(bearing_err) / math.pi)
    pitch_rate = min(pitch_rate, -0.08 - 0.22 * alignment)
    return {
        "yaw_rate": yaw_rate,
        "pitch_rate": pitch_rate,
        "thrust": thrust,
    }
