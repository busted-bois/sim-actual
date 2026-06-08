"""Pilot — attitude-mode gate racer with altitude PID.

Called at ~250 Hz by controller.update(). Uses ATTITUDE mode with pitch_rate
for forward motion and an altitude PID for thrust control. Reads shared_data
(written by mavlink_rx and vision_rx) and sets controller commands directly.
"""

from __future__ import annotations

import math
import time as _time

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
HOVER_THRUST = 0.5
CRUISE_THRUST = 0.55
CRUISE_PITCH_RATE = -0.2
COLLISION_THRUST = 0.4
COLLISION_HOLD_S = 2.0

ALTITUDE_TRIM = 0.55
KP_Z = 0.25
KI_Z = 0.035
KD_Z = 0.12
Z_TARGET_NED = -5.0

VISION_YAW_GAIN = math.radians(40)
VISION_CENTER_DEADBAND = 0.14
VISION_PROXIMITY_R_FRAC = 0.10
VISION_MAX_AGE_S = 0.5

TELEMETRY_YAW_GAIN = 1.0

CONTROL_DT_S = 1 / 250


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class Pilot:
    """Gate-traversal pilot using ATTITUDE mode + altitude PID."""

    def __init__(self, controller, data):  # type: ignore[type-arg]
        self.controller = controller
        self.data = data
        self._z_integral = 0.0
        self._collision_time: float | None = None
        controller.set_control_mode("attitude")
        controller.set_attitude_rates(0, 0, 0, HOVER_THRUST)

    # ------------------------------------------------------------------
    # Main tick — called every cycle at 250 Hz
    # ------------------------------------------------------------------
    def tick(self) -> None:
        armed = self.data.get("armed", False)

        if not armed:
            self._hover()
            return

        # Collision hold
        if self._collision_time is not None:
            elapsed = _time.monotonic() - self._collision_time
            if elapsed < COLLISION_HOLD_S:
                self.controller.set_control_mode("attitude")
                self.controller.set_attitude_rates(0, 0, 0, COLLISION_THRUST)
                return
            self._collision_time = None

        # Check for new collision
        collision = self.data.get("collision")
        if collision is not None:
            self._collision_time = _time.monotonic()

        # No gate data → hover
        track_gates = self.data.get("track_gates")
        if not track_gates:
            self._hover()
            return

        # Gate from vision
        gate_target = self.data.get("gate_target")
        cam = self.data.get("camera")
        if gate_target and gate_target.get("detected"):
            if cam is not None:
                age = _time.monotonic() - cam.get("received_at", 0)
                if age < VISION_MAX_AGE_S:
                    self._fly_toward_gate_vision(gate_target)
                    return

        # Gate from telemetry
        race_status = self.data.get("race_status", {})
        active_idx = race_status.get("active_gate_index", 0)
        odometry = self.data.get("odometry")
        if odometry is not None and len(track_gates) > active_idx:
            self._fly_toward_gate_telemetry(track_gates[active_idx], odometry)
            return

        # Nothing → cruise forward
        self._cruise_forward()

    # ------------------------------------------------------------------
    # Flight primitives
    # ------------------------------------------------------------------
    def _hover(self, z_target: float | None = None) -> None:
        thrust = self._altitude_thrust(HOVER_THRUST, z_target)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, 0, 0, thrust)

    def _cruise_forward(self) -> None:
        thrust = self._altitude_thrust(CRUISE_THRUST)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, CRUISE_PITCH_RATE, 0, thrust)

    def _fly_toward_gate_vision(self, gate_target: dict) -> None:  # type: ignore[type-arg]
        nx = gate_target.get("nx", 0.0)
        ny = gate_target.get("ny", 0.0)
        r_frac = gate_target.get("r_frac", 0.0)

        yaw_rate = _clamp(VISION_YAW_GAIN * nx, -2.0, 2.0)

        near = r_frac >= VISION_PROXIMITY_R_FRAC
        centered = (
            near
            and abs(nx) < VISION_CENTER_DEADBAND
            and abs(ny) < VISION_CENTER_DEADBAND
        )

        if centered:
            pitch = 0.0
            thrust = self._altitude_thrust(HOVER_THRUST)
        else:
            alignment = max(0.0, 1.0 - abs(nx))
            pitch = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
            thrust = self._altitude_thrust(CRUISE_THRUST)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    def _fly_toward_gate_telemetry(self, gate: dict, odometry: dict) -> None:  # type: ignore[type-arg]
        gx, gy, gz = gate["position_ned"]
        ox, oy = odometry["x"], odometry["y"]

        dx = gx - ox
        dy = gy - oy

        # Yaw from quaternion in odometry
        qw = odometry.get("qw", 1.0)
        qx = odometry.get("qx", 0.0)
        qy = odometry.get("qy", 0.0)
        qz = odometry.get("qz", 0.0)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Bearing error: angle from drone heading to gate direction
        bearing_to_gate = math.atan2(dy, dx)
        bearing_error = bearing_to_gate - yaw
        # Normalize to [-pi, pi]
        bearing_error = (bearing_error + math.pi) % (2 * math.pi) - math.pi

        yaw_rate = _clamp(TELEMETRY_YAW_GAIN * bearing_error, -2.0, 2.0)
        alignment = max(0.0, 1.0 - abs(bearing_error) / math.pi)
        pitch = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
        thrust = self._altitude_thrust(CRUISE_THRUST)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(0, pitch, yaw_rate, thrust)

    # ------------------------------------------------------------------
    # Altitude PID
    # ------------------------------------------------------------------
    def _altitude_thrust(self, fallback: float, z_target: float | None = None) -> float:
        odometry = self.data.get("odometry")
        if odometry is None:
            return fallback

        z = odometry.get("z", 0.0)
        vz = odometry.get("vz", 0.0)
        target = z_target if z_target is not None else Z_TARGET_NED

        error = target - z
        self._z_integral += error * CONTROL_DT_S
        # Anti-windup
        self._z_integral = _clamp(self._z_integral, -2.0, 2.0)

        thrust = ALTITUDE_TRIM + KP_Z * error + KI_Z * self._z_integral + KD_Z * vz
        return _clamp(thrust, 0.0, 1.0)
