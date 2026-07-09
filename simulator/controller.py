import math
import time

from pymavlink import mavutil

from simulator.gate_transition import GatePhase

# --------------------------------------------------------------------------------------
# RESET COMMAND
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
PITCH_RATE = -0.3  # rad/s (negative = pitch forward)
ROLL_RATE = 0.0
YAW_RATE = 0.0
THRUST = 0.6  # 0.0 - 1.0

RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def update_attitude_flight_control(mavlink_conn, system_boot_ms):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle attitude. Used by an external controller to
    command the vehicle (manual controller or other system).
    
    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint8_t, values:ATTITUDE_TARGET_TYPEMASK)
    q                         : Attitude quaternion (w, x, y, z order, zero-rotation is 1, 0, 0, 0) (type:float)
    body_roll_rate            : Body roll rate [rad/s] (type:float)
    body_pitch_rate           : Body pitch rate [rad/s] (type:float)
    body_yaw_rate             : Body yaw rate [rad/s] (type:float)
    thrust                    : Collective thrust, normalized to 0 .. 1 (-1 .. 1 for vehicles capable of reverse trust) (type:float)
    """
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],  # dummy quaternion (ignored)
        ROLL_RATE,
        PITCH_RATE,
        YAW_RATE,
        THRUST,
    )


# --------------------------------------------------------------------------------------
# POSITION CONTROLS
# --------------------------------------------------------------------------------------
VELOCITY_POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


def update_position_flight_control(mavlink_conn, system_boot_ms):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle position in a local north-east-down coordinate
    frame. Used by an external controller to command the vehicle
    (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    coordinate_frame          : Valid options are: MAV_FRAME_LOCAL_NED = 1, MAV_FRAME_LOCAL_OFFSET_NED = 7, MAV_FRAME_BODY_NED = 8, MAV_FRAME_BODY_OFFSET_NED = 9 (type:uint8_t, values:MAV_FRAME)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint16_t, values:POSITION_TARGET_TYPEMASK)
    x                         : X Position in NED frame [m] (type:float)
    y                         : Y Position in NED frame [m] (type:float)
    z                         : Z Position in NED frame (note, altitude is negative in NED) [m] (type:float)
    vx                        : X velocity in NED frame [m/s] (type:float)
    vy                        : Y velocity in NED frame [m/s] (type:float)
    vz                        : Z velocity in NED frame [m/s] (type:float)
    afx                       : X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afy                       : Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afz                       : Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    yaw                       : yaw setpoint [rad] (type:float)
    yaw_rate                  : yaw rate setpoint [rad/s] (type:float)
    """
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0,
        0,
        0.0,  # ignored position NED
        2.0,
        0.0,
        0.0,  # Vel - 2 m/s forward
        0.0,
        0,
        0.0,  # ignored acceleration
        0,  # ignored yaw
        0.0,  # ignored yaw rate
    )


def send_body_velocity(mavlink_conn, system_boot_ms, vx, vy, vz):
    """Command a body-frame velocity setpoint (x forward, y right, z down)."""
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        VELOCITY_POSITION_MASK,
        0.0,
        0.0,
        0.0,  # ignored position
        vx,
        vy,
        vz,
        0.0,
        0.0,
        0.0,  # ignored acceleration
        0.0,  # ignored yaw
        0.0,  # ignored yaw rate
    )


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 250

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
        # while dead reckoning: current gate must first leave the frame before a
        # visible gate counts as the *next* gate
        self._gate_seen_gap = False
        self._last_phase = None

    def update(self):
        if self.tracker is not None:
            self._gate_transition_update()
        else:
            update_motor_control(self.sim_conn, self.system_boot_ms)

        time.sleep(1.0 / CONTROL_HZ)

    def _gate_transition_update(self):
        now = time.monotonic()

        # 1. velocity estimate from IMU
        self.velocity_estimator.tick()
        state = self.data.get("state", {})
        speed = state.get("speed_mps", 0.0)

        # 2. perception lateral offset (signed; inf when gate not visible)
        perception = self.data.get("perception", {})
        d_signed = perception.get("gate_lateral_m", math.inf)
        gate_visible = perception.get("gate_visible", False)

        # 3. advance state machine on |d|
        phase = self.tracker.update(abs(d_signed), speed, now)

        # 4. server-side gate pass - reset everything for the new gate
        if self.data.get("race_gate_passed_event"):
            self.data["race_gate_passed_event"] = False
            self.tracker.on_gate_passed()
            self.velocity_estimator.reset()
            self.dead_reckoner.stop()
            self._gate_seen_gap = False
            phase = self.tracker.phase

        # 5. act on phase
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
                # no gate in sight - hold position rather than fly blind
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
                # gate left the frame and a gate is visible again -> next gate
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
                f"[gate] phase={phase.value} thover={self.tracker.thover:.2f}s d={self.tracker.d:.2f}m",
                flush=True,
            )
            self._last_phase = phase

        # 6. publish for observability
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
