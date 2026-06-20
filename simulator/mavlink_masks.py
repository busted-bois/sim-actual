"""MAVLink SET_ATTITUDE_TARGET / SET_POSITION_TARGET type-mask builders."""

from pymavlink import mavutil


def build_attitude_target_type_mask(
    *,
    ignore_attitude: bool = False,
    ignore_roll_rate: bool = True,
    ignore_pitch_rate: bool = True,
    ignore_yaw_rate: bool = True,
    ignore_thrust: bool = False,
) -> int:
    mask = 0
    if ignore_attitude:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    if ignore_roll_rate:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
    if ignore_pitch_rate:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
    if ignore_yaw_rate:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    if ignore_thrust:
        mask |= mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE
    return mask


def build_body_rate_type_mask() -> int:
    return build_attitude_target_type_mask(
        ignore_attitude=True,
        ignore_roll_rate=False,
        ignore_pitch_rate=False,
        ignore_yaw_rate=False,
    )


def build_attitude_only_type_mask() -> int:
    return build_attitude_target_type_mask()


def build_position_target_type_mask(
    *,
    use_position: bool = False,
    use_velocity: bool = False,
    use_yaw: bool = False,
    use_yaw_rate: bool = False,
    force_set: bool = False,
) -> int:
    mask = 0
    if not use_position:
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    if not use_velocity:
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    if not use_yaw:
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    if not use_yaw_rate:
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
    if force_set:
        mask |= mavutil.mavlink.POSITION_TARGET_TYPEMASK_FORCE_SET
    return mask


def build_velocity_type_mask(*, force_set: bool = True) -> int:
    return build_position_target_type_mask(use_velocity=True, force_set=force_set)


def build_position_type_mask() -> int:
    return build_position_target_type_mask(use_position=True)
