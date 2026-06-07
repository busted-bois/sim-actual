import math
import time

from simulator.navigation import active_gate, bearing_error_ned
from simulator.preflight import ensure_countdown_fallback, race_go_allowed

CONTROL_DT_S = 1.0 / 250.0

HOVER_THRUST = 0.5
CRUISE_THRUST = 0.55
CRUISE_PITCH_RATE = -0.2
COLLISION_THRUST = 0.4
COLLISION_HOLD_S = 2.0

VISION_YAW_GAIN_RAD_S = math.radians(40.0)
VISION_CENTER_DEADBAND = 0.14
VISION_PROXIMITY_R_FRAC = 0.10
VISION_MAX_AGE_S = 0.5

TELEMETRY_YAW_GAIN = 1.0
ALTITUDE_TRIM = 0.55
KP_Z = 0.25
KI_Z = 0.035
KD_Z = 0.12
Z_TARGET_NED = -5.0


def _clamp(value, low, high):
    return max(low, min(high, value))


class Pilot:
    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self._collision_hold_start = None
        self._z_integral = 0.0

    def tick(self):
        if not self.data.get("armed"):
            self._hover()
            return

        if self._in_collision_hold():
            return

        if not self.data.get("track_gates"):
            self._hover()
            return

        if not race_go_allowed(self.data):
            if self.data.get("armed") and self.data.get("track_gates"):
                ensure_countdown_fallback(self.data)
            self._hover()
            return

        gate_target = self._fresh_gate_target()
        if gate_target is not None:
            self._fly_toward_gate(gate_target)
            return

        gate = active_gate(self.data)
        odometry = self.data.get("odometry")
        if gate is not None and odometry is not None:
            self._fly_toward_gate_telemetry(gate, odometry)
            return

        self._cruise_forward()

    def _in_collision_hold(self):
        collision = self.data.get("collision")
        if collision is None:
            self._collision_hold_start = None
            return False

        now = time.monotonic()
        if self._collision_hold_start is None:
            self._collision_hold_start = now

        if now - self._collision_hold_start < COLLISION_HOLD_S:
            self.controller.set_attitude_rates(
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=0.0,
                thrust=COLLISION_THRUST,
            )
            return True

        del self.data["collision"]
        self._collision_hold_start = None
        self._hover()
        return True

    def _vision_fresh(self):
        camera = self.data.get("camera")
        if not camera:
            return False
        age = time.time() - float(camera.get("received_at", 0.0))
        return age <= VISION_MAX_AGE_S

    def _fresh_gate_target(self):
        if not self._vision_fresh():
            return None
        gate_target = self.data.get("gate_target") or {}
        if gate_target.get("detected"):
            return gate_target
        return None

    def _altitude_thrust(self, fallback, z_target=None):
        odometry = self.data.get("odometry")
        if odometry is None:
            return fallback
        z = float(odometry["z"])
        vz = float(odometry.get("vz", 0.0))
        target = Z_TARGET_NED if z_target is None else z_target
        ex_z = z - target
        self._z_integral = _clamp(self._z_integral + ex_z * CONTROL_DT_S, -6.0, 6.0)
        return _clamp(
            ALTITUDE_TRIM + KP_Z * ex_z + KI_Z * self._z_integral + KD_Z * vz,
            0.0,
            1.0,
        )

    def _hover(self):
        thrust = self._altitude_thrust(HOVER_THRUST)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=0.0, pitch_rate=0.0, yaw_rate=0.0, thrust=thrust
        )

    def _cruise_forward(self):
        thrust = self._altitude_thrust(CRUISE_THRUST)
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=0.0,
            pitch_rate=CRUISE_PITCH_RATE,
            yaw_rate=0.0,
            thrust=thrust,
        )

    def _fly_toward_gate(self, gate_target):
        nx = float(gate_target["nx"])
        ny = float(gate_target["ny"])
        r_frac = float(gate_target["r_frac"])

        yaw_rate = _clamp(VISION_YAW_GAIN_RAD_S * nx, -2.0, 2.0)
        near_target = r_frac >= VISION_PROXIMITY_R_FRAC
        centered = (
            near_target
            and abs(nx) < VISION_CENTER_DEADBAND
            and abs(ny) < VISION_CENTER_DEADBAND
        )

        if centered:
            pitch_rate = 0.0
            thrust = self._altitude_thrust(HOVER_THRUST)
        else:
            alignment = max(0.0, 1.0 - abs(nx))
            pitch_rate = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
            thrust = self._altitude_thrust(CRUISE_THRUST)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=0.0,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            thrust=thrust,
        )

    def _fly_toward_gate_telemetry(self, gate, odometry):
        attitude = self.data.get("attitude")
        bearing_err = bearing_error_ned(odometry, gate, attitude)
        yaw_rate = _clamp(TELEMETRY_YAW_GAIN * bearing_err, -2.0, 2.0)
        alignment = max(0.0, 1.0 - abs(bearing_err) / math.pi)
        pitch_rate = CRUISE_PITCH_RATE * (0.35 + 0.65 * alignment)
        thrust = self._altitude_thrust(CRUISE_THRUST)

        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=0.0,
            pitch_rate=pitch_rate,
            yaw_rate=yaw_rate,
            thrust=thrust,
        )
