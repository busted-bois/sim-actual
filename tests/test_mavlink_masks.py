from pymavlink import mavutil

from simulator.mavlink_masks import (
    build_attitude_only_type_mask,
    build_body_rate_type_mask,
    build_position_type_mask,
    build_velocity_type_mask,
)


def test_body_rate_mask_ignores_attitude_only():
    mask = build_body_rate_type_mask()
    assert mask & mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    assert not (mask & mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE)


def test_velocity_mask_uses_force_set():
    mask = build_velocity_type_mask()
    assert mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_FORCE_SET
    assert not (mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE)


def test_position_mask_uses_xyz():
    mask = build_position_type_mask()
    assert not (mask & mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE)


def test_attitude_only_mask_keeps_quaternion():
    mask = build_attitude_only_type_mask()
    assert not (mask & mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE)
