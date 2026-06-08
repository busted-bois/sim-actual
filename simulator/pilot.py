import math
import time

from simulator.flight_config import (
    ALTITUDE_TRIM,
    COLLISION_HOLD_S,
    COLLISION_THRUST,
    CONTROL_HZ,
    CRUISE_PITCH_RATE,
    CRUISE_THRUST,
    HOVER_THRUST,
    KD_Z,
    KI_Z,
    KP_Z,
    VISION_MAX_AGE_S,
    resolve_auto_reset_on_collision,
)
from simulator.flight_control import attitude_fallback_command, racing_command
from simulator.navigation import active_gate, yaw_from_state
from simulator.preflight import RaceGoLatch, poll_race_go, race_go_allowed
from simulator.racing_planner import precompute_racing_path, pursuit_target_from_data
from simulator.tracking.snapshot import TrackingSnapshot

CONTROL_DT_S = 1.0 / CONTROL_HZ
SIM_BOOT_RESET_DROP_MS = 500


def _clamp(value, low, high):
    return max(low, min(high, value))


class Pilot:
    def __init__(self, controller, data, auto_reset_on_collision=None):
        self.controller = controller
        self.data = data
        self.auto_reset_on_collision = resolve_auto_reset_on_collision(
            auto_reset_on_collision
        )
        self._collision_hold_start = None
        self._z_integral = 0.0
        self._last_sim_boot_ms = None
        self._last_race_start_boot_ms = None
        self._session_armed = False
        self._go_boot_ms = None
        self._race_go_latch = RaceGoLatch()
        self._awaiting_race_go = False
        self._protect_initial_go = False
        self._pending_restart_arm = False
        self._passed_go = False
        self._racing_path = None
        self._pn_prev_nx = None

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

        self._ensure_racing_path()
        gate = active_gate(self.data)
        if gate is None:
            self._cruise_forward()
            return

        self._fly_race(gate)

    def _ensure_racing_path(self):
        gates = self.data.get("track_gates")
        if not gates or self._racing_path is not None:
            return
        valid = [g for g in gates if g.get("position_ned")]
        if valid:
            self._racing_path = precompute_racing_path(valid)

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
        return race_go_allowed(
            self.data,
            go_boot_ms=self._go_boot_ms,
            is_restart=self._race_go_latch.is_restart,
        )

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
            self._arm_for_session(sim_boot, is_restart=self._pending_restart_arm)
            self._pending_restart_arm = False

        self._last_sim_boot_ms = sim_boot
        self._last_race_start_boot_ms = race_start

    def _arm_for_session(self, sim_boot, is_restart=False):
        if not self.data.get("armed"):
            self.controller.arm()
        self._session_armed = True
        self._go_boot_ms = None
        self._awaiting_race_go = True
        self._race_go_latch.reset_for_arm(sim_boot, is_restart=is_restart)
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
        self._pending_restart_arm = self._passed_go
        self._session_armed = False
        self._go_boot_ms = None
        self._awaiting_race_go = False
        self._protect_initial_go = False
        self._passed_go = False
        self._last_race_start_boot_ms = -1
        self._race_go_latch.reset_for_arm()
        self._z_integral = 0.0
        self._collision_hold_start = None
        self._racing_path = None
        self._pn_prev_nx = None
        self.data.pop("collision", None)
        self._reset_local_tracker()

    def _reset_local_tracker(self):
        tracker = self.data.get("_local_tracker")
        if tracker is not None:
            tracker.reset()
        self.data.pop("tracking_snapshot", None)
        self.data.pop("tracking_health", None)

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

        if self.auto_reset_on_collision and (
            collision.get("threat_level", 0) >= 2 or collision.get("id") == 1002
        ):
            self.controller.reset_sim()
            self._reset_local_tracker()

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

    def _healthy_tracking_snapshot(self):
        snapshot = self.data.get("tracking_snapshot")
        if not isinstance(snapshot, TrackingSnapshot):
            return None
        if snapshot.status != "tracking" or not snapshot.healthy:
            return None
        return snapshot

    def _flight_pose(self):
        snapshot = self._healthy_tracking_snapshot()
        if snapshot is not None:
            return {
                "x": snapshot.x,
                "y": snapshot.y,
                "z": snapshot.z,
                "vx": snapshot.vx,
                "vy": snapshot.vy,
                "vz": snapshot.vz,
                "yaw": snapshot.yaw,
                "roll": snapshot.roll,
                "pitch": snapshot.pitch,
            }

        odometry = self.data.get("odometry")
        if odometry is None:
            return None

        attitude = self.data.get("attitude") or {}
        return {
            "x": float(odometry["x"]),
            "y": float(odometry["y"]),
            "z": float(odometry["z"]),
            "vx": float(odometry.get("vx", 0.0)),
            "vy": float(odometry.get("vy", 0.0)),
            "vz": float(odometry.get("vz", 0.0)),
            "yaw": yaw_from_state(odometry, attitude or None),
            "roll": float(attitude.get("roll", 0.0)),
            "pitch": float(attitude.get("pitch", 0.0)),
        }

    def _altitude_source(self, target_z=None):
        snapshot = self._healthy_tracking_snapshot()
        if snapshot is not None:
            return snapshot.z, snapshot.vz
        odometry = self.data.get("odometry")
        if odometry is None:
            return None
        return float(odometry["z"]), float(odometry.get("vz", 0.0))

    def _altitude_thrust(self, fallback, target_z=None):
        source = self._altitude_source(target_z)
        if source is None:
            return fallback
        z, vz = source
        goal_z = target_z if target_z is not None else z
        ex_z = z - goal_z
        self._z_integral = _clamp(self._z_integral + ex_z * CONTROL_DT_S, -6.0, 6.0)
        return _clamp(
            ALTITUDE_TRIM + KP_Z * ex_z + KI_Z * self._z_integral + KD_Z * vz,
            0.0,
            1.0,
        )

    def _altitude_velocity_cmd(self, target_z):
        source = self._altitude_source(target_z)
        if source is None:
            return 0.0
        z, vz = source
        ex_z = target_z - z
        return _clamp(KP_Z * ex_z - KD_Z * vz, -1.5, 1.5)

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

    def _fly_race(self, gate):
        pose = self._flight_pose()
        gate_target = self._fresh_gate_target()

        if pose is None:
            if gate_target is not None:
                self._fly_vision_attitude(gate_target, gate)
            else:
                self._cruise_forward()
            return

        target = pursuit_target_from_data(pose, self.data, self._racing_path)
        cmd = racing_command(
            pose,
            target,
            gate,
            gate_target,
            self._pn_prev_nx,
            CONTROL_DT_S,
        )
        if gate_target is not None:
            self._pn_prev_nx = float(gate_target.get("nx", 0.0))

        if abs(cmd["bearing_err"]) > math.radians(55.0) and gate_target is not None:
            self._fly_vision_attitude(
                gate_target, gate, bearing_err=cmd["bearing_err"]
            )
            return

        vz = self._altitude_velocity_cmd(target[2])
        self.controller.set_control_mode("position")
        self.controller.set_velocity_body_ned(cmd["vx"], cmd["vy"], vz)

    def _fly_vision_attitude(self, gate_target, gate, bearing_err=0.0):
        target_z = gate.get("position_ned", (0.0, 0.0, -5.0))[2]
        thrust = self._altitude_thrust(CRUISE_THRUST, target_z=target_z)
        cmd = attitude_fallback_command(
            bearing_err,
            gate_target,
            self._pn_prev_nx,
            CONTROL_DT_S,
            thrust,
        )
        if gate_target is not None:
            self._pn_prev_nx = float(gate_target.get("nx", 0.0))
        self.controller.set_control_mode("attitude")
        self.controller.set_attitude_rates(
            roll_rate=0.0,
            pitch_rate=cmd["pitch_rate"],
            yaw_rate=cmd["yaw_rate"],
            thrust=cmd["thrust"],
        )
