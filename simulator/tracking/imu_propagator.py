import math

from simulator.flight_config import ATTITUDE_BLEND


def rotate_body_to_ned(vec_body, roll, pitch, yaw):
    bx, by, bz = vec_body
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    x = cy * cp * bx + (cy * sp * sr - sy * cr) * by + (cy * sp * cr + sy * sr) * bz
    y = sy * cp * bx + (sy * sp * sr + cy * cr) * by + (sy * sp * cr - cy * sr) * bz
    z = -sp * bx + cp * sr * by + cp * cr * bz
    return x, y, z


def propagate_position_velocity(state, accel_body, gyro, dt_s):
    roll = state["roll"] + float(gyro[0]) * dt_s
    pitch = state["pitch"] + float(gyro[1]) * dt_s
    yaw = state["yaw"] + float(gyro[2]) * dt_s

    accel_ned = rotate_body_to_ned(accel_body, roll, pitch, yaw)
    gravity_ned = (0.0, 0.0, 9.81)
    ax = accel_ned[0]
    ay = accel_ned[1]
    az = accel_ned[2] + gravity_ned[2]

    vx = state["vx"] + ax * dt_s
    vy = state["vy"] + ay * dt_s
    vz = state["vz"] + az * dt_s
    x = state["x"] + vx * dt_s
    y = state["y"] + vy * dt_s
    z = state["z"] + vz * dt_s
    return {
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def blend_attitude(state, measured_roll, measured_pitch, measured_yaw):
    alpha = ATTITUDE_BLEND
    return {
        "roll": (1.0 - alpha) * state["roll"] + alpha * measured_roll,
        "pitch": (1.0 - alpha) * state["pitch"] + alpha * measured_pitch,
        "yaw": (1.0 - alpha) * state["yaw"] + alpha * measured_yaw,
    }
