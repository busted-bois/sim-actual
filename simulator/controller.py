import time

from pymavlink import mavutil

from simulator.gate_navigation import GateNavigator

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
    motor_rpms = [MOTOR_FRONT_LEFT, MOTOR_FRONT_RIGHT, MOTOR_BACK_LEFT, MOTOR_BACK_RIGHT, 0, 0, 0, 0]
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6), mavlink_conn.target_system, mavlink_conn.target_component, 0, motor_rpms
    )


# --------------------------------------------------------------------------------------
# ATTITUDE CONTROLS
# --------------------------------------------------------------------------------------
PITCH_RATE = -0.3  # rad/s (negative = pitch forward)
ROLL_RATE = 0.0
YAW_RATE = 0.0
THRUST = 0.6  # 0.0 - 1.0
NAV_THRUST = 0.48
NAV_PITCH_RATE_PER_M_S = 0.03

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


def update_navigation_attitude_control(mavlink_conn, system_boot_ms, command, ramp=1.0):
    now_ms = int(time.time() * 1000)
    pitch_rate = -NAV_PITCH_RATE_PER_M_S * command.vx * ramp
    yaw_rate = command.yaw_rate * ramp
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],
        0.0,
        pitch_rate,
        yaw_rate,
        NAV_THRUST,
    )


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 250
STARTUP_HOLD_S = 1.0
COMMAND_RAMP_S = 3.0


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.navigator = GateNavigator()
        self.started_at_s = time.monotonic()

    def update(self):
        frame_id, detection, vision_time, _yaw_rad, yaw_ready = self._latest_detection()
        now_s = time.monotonic()
        if not yaw_ready or now_s - self.started_at_s < STARTUP_HOLD_S:
            update_navigation_attitude_control(
                self.sim_conn,
                self.system_boot_ms,
                self.navigator.last_command,
                ramp=0.0,
            )
            time.sleep(1.0 / CONTROL_HZ)
            return
        detection_age_s = now_s - vision_time if vision_time is not None else float("inf")
        command = self.navigator.update(frame_id, detection, detection_age_s, now_s)
        ramp = min((now_s - self.started_at_s - STARTUP_HOLD_S) / COMMAND_RAMP_S, 1.0)
        update_navigation_attitude_control(
            self.sim_conn,
            self.system_boot_ms,
            command,
            ramp=ramp,
        )

        time.sleep(1.0 / CONTROL_HZ)

    def _latest_detection(self):
        with self.data["lock"]:
            return (
                self.data.get("latest_frame_id"),
                self.data.get("latest_detection"),
                self.data.get("latest_vision_time"),
                self.data.get("yaw_rad", 0.0),
                self.data.get("yaw_ready", False),
            )

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
