"""Telemetry drain → freshest State each tick.

Prefers ODOMETRY/ATTITUDE when present. Under the VQ2 block (3385 TRAINING
included) those are absent — fall back to StateEstimator on HIGHRES_IMU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from simulator.state_estimator import StateEstimator


@dataclass
class State:
    t: float
    pos: tuple[float, float, float]  # NED m
    vel: tuple[float, float, float]  # NED m/s
    roll: float
    pitch: float
    yaw: float
    gyro: tuple[float, float, float]  # rad/s
    armed: bool
    pose_age: float  # s since last pose update
    has_pose: bool
    pose_source: str = "none"  # odometry | estimator | none


def quat_to_rpy(
    qw: float, qx: float, qy: float, qz: float
) -> tuple[float, float, float]:
    """Quaternion (w,x,y,z) → roll, pitch, yaw (rad). Matches fly2_course.rpy."""
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


class StateTracker:
    """Parse drained MAVLink messages into a single freshest State."""

    def __init__(self) -> None:
        self.armed = False
        self.pos: tuple[float, float, float] | None = None
        self.vel: tuple[float, float, float] | None = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.gyro = (0.0, 0.0, 0.0)
        self._last_pose_t = 0.0
        self._seen_odometry = False
        self._seen_attitude = False
        self._seen_imu = False
        self.estimator = StateEstimator()
        self.pose_source = "none"

    @property
    def has_pose_telemetry(self) -> bool:
        # Odometry/attitude OR estimator finished ground init on IMU.
        return self._seen_odometry or self._seen_attitude or self.estimator.ready

    def reset_estimator(self) -> None:
        self.estimator.reset()
        if not self._seen_odometry:
            self.pos = None
            self.vel = None
            self._last_pose_t = 0.0
            self.pose_source = "none"

    def ingest(self, msg) -> None:
        msg_type = msg.get_type()
        now = time.monotonic()

        if msg_type == "HEARTBEAT":
            self.armed = bool(msg.base_mode & 128)

        elif msg_type == "ODOMETRY":
            self._seen_odometry = True
            self.pose_source = "odometry"
            self.pos = (float(msg.x), float(msg.y), float(msg.z))
            self.vel = (float(msg.vx), float(msg.vy), float(msg.vz))
            self.roll, self.pitch, self.yaw = quat_to_rpy(
                float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])
            )
            self.gyro = (
                float(msg.rollspeed),
                float(msg.pitchspeed),
                float(msg.yawspeed),
            )
            self._last_pose_t = now

        elif msg_type == "ATTITUDE":
            self._seen_attitude = True
            self.roll = float(msg.roll)
            self.pitch = float(msg.pitch)
            self.yaw = float(msg.yaw)
            self.gyro = (
                float(msg.rollspeed),
                float(msg.pitchspeed),
                float(msg.yawspeed),
            )
            if self.pos is not None:
                self._last_pose_t = now

        elif msg_type == "LOCAL_POSITION_NED":
            self.pos = (float(msg.x), float(msg.y), float(msg.z))
            self.vel = (float(msg.vx), float(msg.vy), float(msg.vz))
            self._last_pose_t = now
            if not self._seen_odometry:
                self.pose_source = "local_ned"

        elif msg_type == "HIGHRES_IMU":
            self._seen_imu = True
            self.gyro = (float(msg.xgyro), float(msg.ygyro), float(msg.zgyro))
            imu = {
                "ax": float(msg.xacc),
                "ay": float(msg.yacc),
                "az": float(msg.zacc),
                "gx": float(msg.xgyro),
                "gy": float(msg.ygyro),
                "gz": float(msg.zgyro),
                "mx": float(msg.xmag),
                "my": float(msg.ymag),
                "mz": float(msg.zmag),
                "abs_pressure": float(msg.abs_pressure),
                "pressure_alt": float(msg.pressure_alt),
                "temperature": float(msg.temperature),
                "time_us": int(msg.time_usec),
            }
            self.estimator.on_imu(imu)
            # Prefer odometry when present; else pull ESKF pose.
            if not self._seen_odometry:
                est = self.estimator.pose()
                if est is not None:
                    p, v, q = est
                    self.pos = (float(p[0]), float(p[1]), float(p[2]))
                    self.vel = (float(v[0]), float(v[1]), float(v[2]))
                    self.roll, self.pitch, self.yaw = quat_to_rpy(
                        float(q[0]), float(q[1]), float(q[2]), float(q[3])
                    )
                    self._last_pose_t = now
                    self.pose_source = "estimator"

    def snapshot(self) -> State:
        now = time.monotonic()
        pose_age = (now - self._last_pose_t) if self._last_pose_t > 0 else 1e9
        has_pose = self.pos is not None and self.vel is not None and pose_age < 1.0
        pos = self.pos if self.pos is not None else (0.0, 0.0, 0.0)
        vel = self.vel if self.vel is not None else (0.0, 0.0, 0.0)
        return State(
            t=now,
            pos=pos,
            vel=vel,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            gyro=self.gyro,
            armed=self.armed,
            pose_age=pose_age,
            has_pose=has_pose,
            pose_source=self.pose_source,
        )
