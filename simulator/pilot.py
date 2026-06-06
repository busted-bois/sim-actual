import math
import time

from simulator.navigation import active_gate, bearing_error_ned
from simulator.preflight import RaceGoLatch, poll_race_go, race_go_allowed

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

# Sim reboot / session reset drops sim_boot_time_ms sharply.
SIM_BOOT_RESET_DROP_MS = 500


def _clamp(value, low, high):
    return max(low, min(high, value))


class Pilot:
    def __init__(self, controller, data):
        self.controller = controller
        self.data = data
        self._collision_hold_start = None
        self._z_integral = 0.0
        self._last_sim_boot_ms = None
        self._last_race_start_boot_ms = None
        self._session_armed = False
        self._go_boot_ms = None
        self._race_go_latch = RaceGoLatch()
        self._awaiting_race_go = False
        self._protect_initial_go = False
        self._passed_go = False

    def tick(self):
        self._consume_main_latch()
        self._update_race_session()

        if not self.data.get("armed"):
            self._hover()
            return

        if self._in_collision_hold():
            return

        if not self.data.get("track_gates"):
            self._hover()
            return

        if self._awaiting_race_go:
            allowed, go_boot_ms = poll_race_go(self.data, self._race_go_latch)
            if allowed and go_boot_ms is not None:
                self._go_boot_ms = go_boot_ms
                self._awaiting_race_go = False
                race = self.data.get("race_status") or {}
                print(
                    "Race go (restart)! "
                    f"sim_boot={race.get('sim_boot_time_ms')}ms "
                    f"race_start={race.get('race_start_boot_time_ms')}ms "
                    f"go_boot={go_boot_ms}ms "
                    f"branch={self._race_go_latch.branch}",
                    flush=True,
                )
            else:
                self._hover()
                return

        if not self._race_go_allowed():
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

    def _consume_main_latch(self):
        latched = self.data.pop("_latched_go_boot_ms", None)
        if latched is not None:
            self._go_boot_ms = latched
            self._session_armed = True
            self._awaiting_race_go = False
            self._protect_initial_go = True
            self._passed_go = False
            race = self.data.get("race_status") or {}
            self._last_sim_boot_ms = race.get("sim_boot_time_ms")
            self._last_race_start_boot_ms = race.get("race_start_boot_time_ms", -1)

    def _race_go_allowed(self):
        return race_go_allowed(self.data, go_boot_ms=self._go_boot_ms)

    def _update_race_session(self):
        race = self.data.get("race_status") or {}
        race_start = race.get("race_start_boot_time_ms", -1)
        sim_boot = race.get("sim_boot_time_ms", 0)

        if self._go_boot_ms is not None and sim_boot >= self._go_boot_ms:
            self._passed_go = True

        if race_start < 0:
            self._protect_initial_go = False

        if self._is_new_race_session(sim_boot, race_start):
            self._begin_new_race_session()

        if self.data.get("track_gates") and not self._session_armed and race_start < 0:
            self._arm_for_session(sim_boot)

        self._last_sim_boot_ms = sim_boot
        self._last_race_start_boot_ms = race_start

    def _arm_for_session(self, sim_boot):
        self.controller.arm()
        self._session_armed = True
        self._go_boot_ms = None
        self._awaiting_race_go = True
        self._race_go_latch.reset_for_arm(sim_boot)
        print(f"Armed for session at sim_boot={sim_boot}ms", flush=True)

    def _is_new_race_session(self, sim_boot, race_start):
        if self._last_sim_boot_ms is None:
            return False

        if self._protect_initial_go:
            return False

        if (
            self._passed_go
            and sim_boot < self._last_sim_boot_ms - SIM_BOOT_RESET_DROP_MS
        ):
            return True

        return (
            self._last_race_start_boot_ms is not None
            and self._last_race_start_boot_ms >= 0
            and race_start < 0
        )

    def _begin_new_race_session(self):
        self._session_armed = False
        self._go_boot_ms = None
        self._awaiting_race_go = False
        self._protect_initial_go = False
        self._passed_go = False
        self._last_race_start_boot_ms = -1
        self._race_go_latch.reset_for_arm()
        self._z_integral = 0.0
        self._collision_hold_start = None
        self.data.pop("collision", None)

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

    def _altitude_thrust(self, fallback):
        odometry = self.data.get("odometry")
        if odometry is None:
            return fallback
        z = float(odometry["z"])
        vz = float(odometry.get("vz", 0.0))
        ex_z = z - Z_TARGET_NED
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
