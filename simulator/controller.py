import time

from pymavlink import mavutil

from simulator.config import TAKEOFF_THRUST
from simulator.pilot import ControlSetpoint, Pilot

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


def update_attitude_flight_control(
    mavlink_conn,
    system_boot_ms,
    roll_rate=ROLL_RATE,
    pitch_rate=PITCH_RATE,
    yaw_rate=YAW_RATE,
    thrust=THRUST,
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


def update_position_flight_control(
    mavlink_conn,
    system_boot_ms,
    vx=0.0,
    vy=0.0,
    vz=0.0,
    yaw_rate=0.0,
):
    now_ms = int(time.time() * 1000)
    mask = VELOCITY_POSITION_MASK
    if vx != 0.0:
        mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE
    if vy != 0.0:
        mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE
    if vz != 0.0:
        mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE
    if yaw_rate != 0.0:
        mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

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
        self.pilot = Pilot(data)
        self.data["pilot"] = self.pilot

    def update(self):
        dt_s = 1.0 / CONTROL_HZ
        cs: ControlSetpoint = self.pilot.update(dt_s)

        if cs.mode == "attitude":
            update_attitude_flight_control(
                self.sim_conn,
                self.system_boot_ms,
                roll_rate=0.0,
                pitch_rate=0.0,
                yaw_rate=cs.yaw_rate or 0.0,
                thrust=cs.thrust or TAKEOFF_THRUST,
            )
        elif cs.mode == "velocity":
            vx, vy, vz = cs.vel_ned if cs.vel_ned else (0.0, 0.0, 0.0)
            update_position_flight_control(
                self.sim_conn,
                self.system_boot_ms,
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_rate=cs.yaw_rate or 0.0,
            )

        time.sleep(dt_s)

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
