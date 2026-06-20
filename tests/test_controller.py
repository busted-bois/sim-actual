from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pymavlink import mavutil

from simulator.controller import (
    MAVLINK_CMD_SIM_RESET,
    Controller,
    _send_motor_control,
    _send_position_target,
)
from simulator.mavlink_masks import build_body_rate_type_mask, build_velocity_type_mask


def _controller():
    mav = MagicMock()
    conn = SimpleNamespace(
        target_system=1,
        target_component=1,
        mav=mav,
    )
    ctrl = Controller(conn, {}, system_boot_ms=1000)
    ctrl.pilot = MagicMock()
    return ctrl, mav


def test_arm_sends_command():
    ctrl, mav = _controller()
    ctrl.arm()
    mav.command_long_send.assert_called_once_with(
        1,
        1,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_disarm_sends_command():
    ctrl, mav = _controller()
    ctrl.disarm()
    mav.command_long_send.assert_called_once_with(
        1,
        1,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_reset_sim_sends_command():
    ctrl, mav = _controller()
    ctrl.reset_sim()
    mav.command_long_send.assert_called_once_with(
        1,
        1,
        MAVLINK_CMD_SIM_RESET,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_request_highres_imu_sends_interval():
    ctrl, mav = _controller()
    ctrl.request_highres_imu(120)
    mav.command_long_send.assert_called_once()
    args = mav.command_long_send.call_args[0]
    assert args[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert args[4] == mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU


def test_set_motor_rpms_stores_values():
    ctrl, _ = _controller()
    ctrl.set_motor_rpms(10, 20, 30, 40)
    assert ctrl._motor_rpms == [10, 20, 30, 40, 0.0, 0.0, 0.0, 0.0]


def test_set_attitude_rates_stores_values():
    ctrl, _ = _controller()
    ctrl.set_attitude_rates(0.1, -0.2, 0.3, 0.7)
    assert ctrl._roll_rate == 0.1
    assert ctrl._pitch_rate == -0.2
    assert ctrl._yaw_rate == 0.3
    assert ctrl._thrust == 0.7
    assert ctrl.control_mode == "attitude"


def test_set_position_ned_sets_pose_mode():
    ctrl, _ = _controller()
    ctrl.set_position_ned(1.0, 2.0, -3.0, yaw=0.5)
    assert ctrl._px == 1.0
    assert ctrl._py == 2.0
    assert ctrl._pz == -3.0
    assert ctrl.control_mode == "position_pose"


def test_set_velocity_body_ned_uses_body_frame():
    ctrl, _ = _controller()
    ctrl.set_velocity_body_ned(1.0, 0.0, 0.0)
    assert ctrl._position_frame == "body_ned"


def test_set_control_mode_rejects_invalid():
    ctrl, _ = _controller()
    with pytest.raises(ValueError):
        ctrl.set_control_mode("invalid")


def test_update_motor_mode_sends_actuator_control():
    ctrl, mav = _controller()
    ctrl.set_control_mode("motor")
    ctrl.set_motor_rpms(1, 2, 3, 4)
    ctrl.update()
    ctrl.pilot.tick.assert_called_once()
    mav.set_actuator_control_target_send.assert_called_once()
    args = mav.set_actuator_control_target_send.call_args[0]
    assert args[1:4] == (1, 1, 0)
    assert args[4] == [1, 2, 3, 4, 0.0, 0.0, 0.0, 0.0]


def test_update_attitude_mode_sends_attitude_target():
    ctrl, mav = _controller()
    ctrl.set_attitude_rates(0.0, -0.5, 0.0, 0.8)
    ctrl.update()
    mav.set_attitude_target_send.assert_called_once()
    args = mav.set_attitude_target_send.call_args[0]
    assert args[3] == build_body_rate_type_mask()
    assert args[5:9] == (0.0, -0.5, 0.0, 0.8)


def test_update_position_mode_sends_position_target():
    ctrl, mav = _controller()
    ctrl.set_velocity_ned(3.0, 1.0, -0.5)
    ctrl.update()
    mav.set_position_target_local_ned_send.assert_called_once()
    args = mav.set_position_target_local_ned_send.call_args[0]
    assert args[3] == mavutil.mavlink.MAV_FRAME_LOCAL_NED
    assert args[4] == build_velocity_type_mask()
    assert args[8:11] == (3.0, 1.0, -0.5)


def test_get_tracking_snapshot_reads_shared_data():
    ctrl, _ = _controller()
    ctrl.data["tracking_snapshot"] = {"status": "tracking"}
    assert ctrl.get_tracking_snapshot()["status"] == "tracking"


def test_send_helpers_call_mavlink():
    mav = MagicMock()
    conn = SimpleNamespace(target_system=2, target_component=3, mav=mav)

    _send_motor_control(conn, [1, 2, 3, 4, 0, 0, 0, 0])
    mav.set_actuator_control_target_send.assert_called_once()

    _send_position_target(
        conn,
        500,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        build_velocity_type_mask(),
        0,
        0,
        0,
        2.0,
        0.0,
        -1.0,
        0.0,
        0.0,
    )
    mav.set_position_target_local_ned_send.assert_called_once()
