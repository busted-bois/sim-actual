"""Gyroscope attitude integrator with optional complementary accel fusion.

Gyro integration is accurate short-term but drifts; the accelerometer's
gravity direction is stable long-term but corrupted by thrust/manoeuvre
specific force. `update_with_accel` fuses them (complementary filter),
gating the accel correction so thrust does not pull the estimate off.
`update` keeps the original gyro-only path.
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
DEFAULT_ALPHA = 0.98  # weight on gyro; (1-alpha) on accel gravity tilt
ACC_GATE_TOL = 0.15  # correct only when ||accel| - g| <= tol * g
ALPHA_REF_DT = 0.01  # dt at which alpha is defined (100 Hz); scaled for other rates
RATE_GATE_RAD_S = 0.1  # correct only when all gyro rates below this (~5.7 deg/s)
MAX_CORR_RAD_S = math.radians(0.5)  # slew limit on accel correction (deg/s -> rad/s)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX Euler (rad, NED) -> quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class GyroAHRS:
    """Quaternion gyro integrator. NED ZYX Euler: +roll RWD, +pitch nose up."""

    def __init__(self, initial_pitch_deg: float = 0.0, initial_roll_deg: float = 0.0):
        self.q = np.array(
            euler_to_quat(
                math.radians(initial_roll_deg),
                math.radians(initial_pitch_deg),
                0.0,
            ),
            dtype=np.float64,
        )

    def _integrate_gyro(self, gx: float, gy: float, gz: float, dt: float) -> None:
        """Quaternion first-order gyro integration (rad/s, sign-corrected)."""
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        qw_n = qw + h * (-qx * gx - qy * gy - qz * gz)
        qx_n = qx + h * (qw * gx + qy * gz - qz * gy)
        qy_n = qy + h * (qw * gy - qx * gz + qz * gx)
        qz_n = qz + h * (qw * gz + qx * gy - qy * gx)
        n = math.sqrt(qw_n**2 + qx_n**2 + qy_n**2 + qz_n**2)
        self.q = np.array([qw_n, qx_n, qy_n, qz_n], dtype=np.float64) / n

    def update(
        self, gx: float, gy: float, gz: float, dt: float
    ) -> tuple[float, float, float]:
        """Integrate one gyro sample (rad/s, sign-corrected). Returns Euler deg."""
        self._integrate_gyro(gx, gy, gz, dt)
        return self.euler_deg()

    def update_with_accel(
        self,
        gx: float,
        gy: float,
        gz: float,
        ax: float,
        ay: float,
        az: float,
        dt: float,
        alpha: float = DEFAULT_ALPHA,
        gate_tol: float = ACC_GATE_TOL,
    ) -> tuple[float, float, float]:
        """Complementary filter: gyro-integrate, then leak roll/pitch toward
        the accelerometer gravity tilt when the drone is quiescent.

        Accel signs match this sim's specific-force convention (level body
        reads az ~= -g, see gp_estimation._integrate): the gravity-in-body
        vector is -a, giving roll = atan2(-ay, -az), pitch = atan2(ax,
        hypot(ay, az)) (both 0 when level). Yaw stays gyro-only (accel is
        blind to heading).

        A hovering/leaning quad's accelerometer measures the thrust vector
        (|a| ~= g along body -z), which reads as ~zero tilt even when the
        drone is genuinely banked - so |a| ~= g alone is NOT a safe gate in
        flight. Three guards keep the correction drift-only:
        - |a| gate: reject thrust spikes / free-fall (|a| far from g);
        - rate gate: correct only when gyro rates are near zero (not
          manoeuvring);
        - slew limit: correction capped at MAX_CORR_RAD_S, plenty to cancel
          gyro drift but too weak to corrupt a few-second gate approach.
        alpha is defined at 100 Hz (ALPHA_REF_DT) and scaled by dt so the
        400 Hz estimator loop doesn't multiply the gain.
        """
        self._integrate_gyro(gx, gy, gz, dt)
        if alpha >= 1.0:
            return self.euler_deg()

        a_mag = math.sqrt(ax * ax + ay * ay + az * az)
        if a_mag <= 1e-6 or abs(a_mag - G) > gate_tol * G:
            return self.euler_deg()  # thrust/manoeuvre: pure gyro this tick
        if max(abs(gx), abs(gy), abs(gz)) > RATE_GATE_RAD_S:
            return self.euler_deg()  # rotating: accel tilt untrustworthy

        roll_g, pitch_g, yaw_g = self._euler_rad()
        roll_a = math.atan2(-ay, -az)
        pitch_a = math.atan2(ax, math.hypot(ay, az))

        k = (1.0 - alpha) * (dt / ALPHA_REF_DT)
        max_step = MAX_CORR_RAD_S * dt

        def _step(err: float) -> float:
            err = math.atan2(math.sin(err), math.cos(err))  # wrap to [-pi, pi]
            return max(-max_step, min(max_step, k * err))

        roll = roll_g + _step(roll_a - roll_g)
        pitch = pitch_g + _step(pitch_a - pitch_g)
        self.q = np.array(euler_to_quat(roll, pitch, yaw_g), dtype=np.float64)
        return self.euler_deg()

    def _euler_rad(self) -> tuple[float, float, float]:
        qw, qx, qy, qz = self.q
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        pitch = math.asin(sp)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return roll, pitch, yaw

    def euler_deg(self) -> tuple[float, float, float]:
        roll, pitch, yaw = self._euler_rad()
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    @property
    def quaternion(self) -> np.ndarray:
        return self.q.copy()
