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

RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
NAV_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_THROTTLE_IGNORE
)
NAV_PITCH_RATE_PER_M_S = 0.8


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


def update_navigation_rate_control(mavlink_conn, system_boot_ms, command, ramp):
    now_ms = int(time.time() * 1000)
    pitch_rate = -NAV_PITCH_RATE_PER_M_S * command.vx * ramp
    yaw_rate = command.yaw_rate * ramp
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        NAV_ATTITUDE_MASK,
        [1, 0, 0, 0],
        0.0,
        pitch_rate,
        yaw_rate,
        0.0,
    )


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 250
STARTUP_HOLD_S = 1.0
COMMAND_RAMP_S = 3.0
DEBUG_LOG_HZ = 5.0

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.navigator = GateNavigator()
        self.started_at_s = time.monotonic()
        self.last_debug_log_s = 0.0

    def update(self):
        frame_id, detection, vision_time, _yaw_rad, yaw_ready, pos_ned, vel_ned = self._latest_detection()
        now_s = time.monotonic()
        if not yaw_ready or pos_ned is None or now_s - self.started_at_s < STARTUP_HOLD_S:
            self._log_debug(now_s, "hold", None, self.navigator.last_command, 0.0, 0.0, pos_ned, vel_ned, yaw_ready)
            time.sleep(1.0 / CONTROL_HZ)
            return
        detection_age_s = now_s - vision_time if vision_time is not None else float("inf")
        if detection_age_s > 0.15:
            detection = None
        command = self.navigator.update(frame_id, detection, detection_age_s, now_s)
        ramp = min((now_s - self.started_at_s - STARTUP_HOLD_S) / COMMAND_RAMP_S, 1.0)
        sim_vz = 0.0
        mode = "nav" if detection is not None else "idle"
        self._log_debug(now_s, mode, detection, command, ramp, sim_vz, pos_ned, vel_ned, yaw_ready)
        update_navigation_rate_control(self.sim_conn, self.system_boot_ms, command, ramp)

        time.sleep(1.0 / CONTROL_HZ)

    def _log_debug(self, now_s, mode, detection, command, ramp, sim_vz, pos_ned, vel_ned, yaw_ready):
        if now_s - self.last_debug_log_s < 1.0 / DEBUG_LOG_HZ:
            return
        self.last_debug_log_s = now_s
        z = None if pos_ned is None else float(pos_ned[2])
        vz = None if vel_ned is None else float(vel_ned[2])
        if detection is None:
            det = "none"
        else:
            det = (
                "conf=%.2f range=%.2f ex=%.2f ey=%.2f bbox=%s"
                % (detection.confidence, detection.range_m, detection.ex, detection.ey, detection.bbox)
            )
        print(
            "ctrl mode=%s yaw_ready=%s z=%s vz=%s sim_vz=%.2f ramp=%.2f cmd=(vx=%.2f vy=%.2f vz=%.2f yaw=%.2f) det=%s"
            % (
                mode,
                yaw_ready,
                "none" if z is None else "%.2f" % z,
                "none" if vz is None else "%.2f" % vz,
                sim_vz,
                ramp,
                command.vx,
                command.vy,
                command.vz,
                command.yaw_rate,
                det,
            ),
            flush=True,
        )

    def _latest_detection(self):
        with self.data["lock"]:
            return (
                self.data.get("latest_frame_id"),
                self.data.get("latest_detection"),
                self.data.get("latest_vision_time"),
                self.data.get("yaw_rad", 0.0),
                self.data.get("yaw_ready", False),
                self.data.get("pos_ned"),
                self.data.get("vel_ned"),
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
