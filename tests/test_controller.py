from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pymavlink import mavutil

from simulator.controller import (
    MAVLINK_CMD_SIM_RESET,
    Controller,
    _send_attitude_rates,
    _send_motor_control,
    _send_velocity_ned,
)


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


def test_set_velocity_ned_stores_values():
    ctrl, _ = _controller()
    ctrl.set_velocity_ned(1.0, 2.0, 3.0)
    assert ctrl._vx == 1.0
    assert ctrl._vy == 2.0
    assert ctrl._vz == 3.0


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
    ctrl.set_control_mode("attitude")
    ctrl.set_attitude_rates(0.0, -0.5, 0.0, 0.8)
    ctrl.update()
    mav.set_attitude_target_send.assert_called_once()
    args = mav.set_attitude_target_send.call_args[0]
    assert args[3] == mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
    assert args[5:9] == (0.0, -0.5, 0.0, 0.8)


def test_update_position_mode_sends_position_target():
    ctrl, mav = _controller()
    ctrl.set_control_mode("position")
    ctrl.set_velocity_ned(3.0, 1.0, -0.5)
    ctrl.update()
    mav.set_position_target_local_ned_send.assert_called_once()
    args = mav.set_position_target_local_ned_send.call_args[0]
    assert args[3] == mavutil.mavlink.MAV_FRAME_LOCAL_NED
    assert args[8:11] == (3.0, 1.0, -0.5)


def test_send_helpers_call_mavlink():
    mav = MagicMock()
    conn = SimpleNamespace(target_system=2, target_component=3, mav=mav)

    _send_motor_control(conn, [1, 2, 3, 4, 0, 0, 0, 0])
    mav.set_actuator_control_target_send.assert_called_once()

    _send_attitude_rates(conn, 500, 0.1, -0.2, 0.3, 0.6)
    mav.set_attitude_target_send.assert_called_once()

    _send_velocity_ned(conn, 500, 2.0, 0.0, -1.0)
    mav.set_position_target_local_ned_send.assert_called_once()
