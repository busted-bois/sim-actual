import time

from pymavlink import mavutil

from simulator.pilot import Pilot

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


# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------
CONTROL_HZ = 250


class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.control_mode = "motor"
        self._roll_rate = 0.0
        self._pitch_rate = 0.0
        self._yaw_rate = 0.0
        self._thrust = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self.pilot = Pilot(self, data)

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
        self.pilot.tick()

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
