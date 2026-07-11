import math
import os
import time

from pymavlink import mavutil

from simulator.gate_transition import GatePhase

# --------------------------------------------------------------------------------------
# RESET COMMAND
# --------------------------------------------------------------------------------------
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# MOTOR CONTROLS
# --------------------------------------------------------------------------------------
MOTOR_FRONT_LEFT = 0
MOTOR_FRONT_RIGHT = 1
MOTOR_BACK_LEFT = 0
MOTOR_BACK_RIGHT = 0


def update_motor_control(mavlink_conn, system_boot_ms):
    motor_rpms = [
        MOTOR_FRONT_LEFT,
        MOTOR_FRONT_RIGHT,
        MOTOR_BACK_LEFT,
        MOTOR_BACK_RIGHT,
        0,
        0,
        0,
        0,
    ]
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms,
    )


# --------------------------------------------------------------------------------------
# ATTITUDE CONTROLS
# --------------------------------------------------------------------------------------
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def _send_attitude_rates(
    mavlink_conn,
    system_boot_ms,
    roll_rate=0.0,
    pitch_rate=0.0,
    yaw_rate=0.0,
    thrust=0.6,
):
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],  # dummy quaternion (ignored)
        roll_rate,
        pitch_rate,
        yaw_rate,
        thrust,
    )


# --------------------------------------------------------------------------------------
# POSITION / VELOCITY CONTROLS
# --------------------------------------------------------------------------------------
VELOCITY_POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)


def _send_velocity_ned(
    mavlink_conn,
    system_boot_ms,
    vx=0.0,
    vy=0.0,
    vz=0.0,
    yaw_rate=0.0,
):
    now_ms = int(time.time() * 1000)
    mask = VELOCITY_POSITION_MASK
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE

    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        mask,
        0.0,
        0.0,
        0.0,  # ignored position NED
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,  # ignored acceleration
        0.0,  # ignored yaw
        yaw_rate,
    )


def send_body_velocity(mavlink_conn, system_boot_ms, vx, vy, vz):
    """Command a body-frame velocity setpoint (x forward, y right, z down)."""
    now_ms = int(time.time() * 1000)
    mask = VELOCITY_POSITION_MASK
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        mask,
        0.0,
        0.0,
        0.0,
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------
# Spec VADR-TS-003 4.4: command rate MUST stay below 100 Hz (was 250 --
# out-of-spec commands may be dropped or applied erratically by the sim).
CONTROL_HZ = 90

APPROACH_SPEED_MPS = 1.0  # forward speed while centering on the gate
LATERAL_GAIN = 1.0  # m/s of correction per metre of lateral offset
LATERAL_MAX_MPS = 1.0


class Controller:
    def __init__(
        self,
        sim_conn,
        data,
        system_boot_ms,
        velocity_estimator=None,
        tracker=None,
        dead_reckoner=None,
    ):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.velocity_estimator = velocity_estimator
        self.tracker = tracker
        self.dead_reckoner = dead_reckoner
        self._gate_seen_gap = False
        self._last_phase = None
        self.control_mode = "motor"
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._yaw_rate = 0.0
        self._thrust = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self.pilot = self._make_pilot()
        self._disarm_ticks = 0

    def _gate_transition_enabled(self) -> bool:
        """Use gate-transition brain when modules are wired and not in AUTO_FLIGHT."""
        if self.tracker is None or self.velocity_estimator is None:
            return False
        if self.dead_reckoner is None:
            return False
        from simulator.auto_flight import auto_flight_enabled

        if auto_flight_enabled():
            return False
        # GATE_TRANSITION=0 disables; default on when modules present
        return os.environ.get("GATE_TRANSITION", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _make_pilot(self):
        from simulator.auto_flight import auto_flight_enabled

        if auto_flight_enabled():
            # AUTO_PILOT selects the auto-flight brain. Default is the IBVS
            # pixel servo (needs no position estimate; smoothest live flight
            # so far). AUTO_PILOT=vnav selects the world-map vision navigator
            # (NaN-proofed EKF pose).
            if os.environ.get("AUTO_PILOT", "ibvs").strip().lower() == "vnav":
                from simulator.vision_nav_pilot import VisionNavPilot

                return VisionNavPilot(self, self.data)
            from simulator.ibvs_pilot import IBVSPilot

            return IBVSPilot(self, self.data)
        from simulator.pilot import Pilot

        return Pilot(self, self.data)

    def set_control_mode(self, mode):
        self.control_mode = mode

    def set_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self._roll_rate = roll_rate
        self._pitch_rate = pitch_rate
        self._yaw_rate = yaw_rate
        self._thrust = thrust

    def set_velocity_ned(self, vx, vy, vz, yaw_rate):
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self._yaw_rate = yaw_rate

    def disarm(self):
        pass

    def update(self):
        if self._gate_transition_enabled():
            self._gate_transition_update()
            if not self.data.get("armed", False):
                self._disarm_ticks += 1
                if self._disarm_ticks % 50 == 1:
                    self.arm()
            else:
                self._disarm_ticks = 0
            time.sleep(1.0 / CONTROL_HZ)
            return

        self.pilot.tick()

        if not self.data.get("armed", False):
            self._disarm_ticks += 1
            if self._disarm_ticks % 50 == 1:
                self.arm()
        else:
            self._disarm_ticks = 0

        if self.control_mode == "motor":
            update_motor_control(self.sim_conn, self.system_boot_ms)
        elif self.control_mode == "attitude":
            _send_attitude_rates(
                self.sim_conn,
                self.system_boot_ms,
                roll_rate=self._roll_rate,
                pitch_rate=self._pitch_rate,
                yaw_rate=self._yaw_rate,
                thrust=self._thrust,
            )
        elif self.control_mode == "position":
            _send_velocity_ned(
                self.sim_conn,
                self.system_boot_ms,
                vx=self._vx,
                vy=self._vy,
                vz=self._vz,
                yaw_rate=self._yaw_rate,
            )

        time.sleep(1.0 / CONTROL_HZ)

    def _gate_transition_update(self):
        now = time.monotonic()

        self.velocity_estimator.tick()
        state = self.data.get("state", {})
        speed = state.get("speed_mps", 0.0)

        perception = self.data.get("perception", {})
        d_signed = perception.get("gate_lateral_m", math.inf)
        gate_visible = perception.get("gate_visible", False)

        phase = self.tracker.update(abs(d_signed), speed, now)

        if self.data.get("race_gate_passed_event"):
            self.data["race_gate_passed_event"] = False
            self.tracker.on_gate_passed()
            self.velocity_estimator.reset()
            self.dead_reckoner.stop()
            self._gate_seen_gap = False
            phase = self.tracker.phase

        if phase == GatePhase.APPROACH:
            self.dead_reckoner.stop()
            if gate_visible:
                vy = max(
                    -LATERAL_MAX_MPS, min(LATERAL_MAX_MPS, LATERAL_GAIN * d_signed)
                )
                send_body_velocity(
                    self.sim_conn, self.system_boot_ms, APPROACH_SPEED_MPS, vy, 0.0
                )
            else:
                send_body_velocity(self.sim_conn, self.system_boot_ms, 0.0, 0.0, 0.0)

        elif phase == GatePhase.HOVERING:
            send_body_velocity(self.sim_conn, self.system_boot_ms, 0.0, 0.0, 0.0)

        elif phase == GatePhase.COMMITTED:
            self.dead_reckoner.start()
            self._gate_seen_gap = False
            send_body_velocity(
                self.sim_conn,
                self.system_boot_ms,
                self.dead_reckoner.tick(now),
                0.0,
                0.0,
            )

        elif phase == GatePhase.DEAD_RECKON:
            if not gate_visible:
                self._gate_seen_gap = True
            elif self._gate_seen_gap:
                self.tracker.on_next_gate_visible()
                self.dead_reckoner.stop()
                self._gate_seen_gap = False
            send_body_velocity(
                self.sim_conn,
                self.system_boot_ms,
                self.dead_reckoner.tick(now),
                0.0,
                0.0,
            )

        if phase != self._last_phase:
            print(
                f"[gate] phase={phase.value} thover={self.tracker.thover:.2f}s "
                f"d={self.tracker.d:.2f}m",
                flush=True,
            )
            self._last_phase = phase

        self.data["gate"] = {
            "phase": phase.value,
            "thover": self.tracker.thover,
            "d": self.tracker.d,
            "should_commit": self.tracker.should_commit(),
            "dead_reckon_distance_m": self.dead_reckoner.distance_m,
        }

    # -------------------------------
    # Arm the drone
    # -------------------------------
    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
