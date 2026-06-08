import math

from simulator.math_util import normalize_angle


def active_gate(data):
    gates = data.get("track_gates")
    if not gates:
        return None
    race = data.get("race_status") or {}
    index = race.get("active_gate_index", 0)
    if index < 0 or index >= len(gates):
        return None
    return gates[index]


def yaw_from_state(odometry, attitude=None):
    if attitude is not None:
        return float(attitude["yaw"])
    qx = float(odometry["qx"])
    qy = float(odometry["qy"])
    qz = float(odometry["qz"])
    qw = float(odometry["qw"])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def bearing_error_from_pose(x, y, yaw, gate):
    gx, gy, _gz = gate["position_ned"]
    target_bearing = math.atan2(gy - float(y), gx - float(x))
    return normalize_angle(target_bearing - float(yaw))


def bearing_error_ned(odometry, gate, attitude=None):
    current_yaw = yaw_from_state(odometry, attitude)
    return bearing_error_from_pose(odometry["x"], odometry["y"], current_yaw, gate)


def distance_to_gate(odometry, gate):
    gx, gy, gz = gate["position_ned"]
    dx = gx - float(odometry["x"])
    dy = gy - float(odometry["y"])
    dz = gz - float(odometry["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)
