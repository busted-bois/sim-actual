import struct
from types import SimpleNamespace
from unittest.mock import MagicMock

from pymavlink import mavutil

from simulator.mavlink_rx import ENCAPSULATED_TRACK_INFO_MSG_ID, MAVLinkRX


def _rx():
    return MAVLinkRX(mavlink_connection=None, data={})


def test_on_heartbeat_armed():
    rx = _rx()
    armed_flag = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    rx.on_heartbeat(SimpleNamespace(base_mode=armed_flag))
    assert rx.data["armed"] is True

    rx.on_heartbeat(SimpleNamespace(base_mode=0))
    assert rx.data["armed"] is False


def test_on_attitude_stores_telemetry():
    rx = _rx()
    rx.on_attitude(
        SimpleNamespace(
            roll=0.1,
            pitch=-0.2,
            yaw=1.5,
            rollspeed=0.01,
            pitchspeed=0.02,
            yawspeed=0.03,
            time_boot_ms=9000,
        )
    )
    assert rx.data["attitude"] == {
        "roll": 0.1,
        "pitch": -0.2,
        "yaw": 1.5,
        "roll_speed": 0.01,
        "pitch_speed": 0.02,
        "yaw_speed": 0.03,
        "time_boot_ms": 9000,
    }


def test_on_highres_imu_stores_telemetry():
    rx = _rx()
    rx.on_highres_imu(
        SimpleNamespace(
            xacc=0.1,
            yacc=0.2,
            zacc=-9.8,
            xgyro=0.01,
            ygyro=0.02,
            zgyro=0.03,
            time_usec=50000,
        )
    )
    assert rx.data["highres_imu"]["zacc"] == -9.8
    assert rx.data["highres_imu"]["time_boot_us"] == 50000


def test_on_local_position_ned_stores_telemetry():
    rx = _rx()
    rx.on_local_position_ned(
        SimpleNamespace(x=1.0, y=2.0, z=-3.0, vx=0.1, vy=0.2, vz=-0.1, time_boot_ms=100)
    )
    pos = rx.data["local_position_ned"]
    assert pos["x"] == 1.0
    assert pos["vy"] == 0.2


def test_on_race_status_parses_encapsulated_payload():
    rx = _rx()
    payload = struct.pack("<BQqqIq", 1, 5000, 100, -1, 3, 42)
    rx.on_race_status(SimpleNamespace(data=bytearray(payload)))
    assert rx.data["race_status"] == {
        "sim_boot_time_ms": 5000,
        "race_start_boot_time_ms": 100,
        "race_finish_time_ns": -1,
        "active_gate_index": 3,
        "last_gate_race_time": 42,
    }


def test_on_track_data_stores_gates():
    rx = _rx()
    gate = struct.pack(
        "<Hfffffffff",
        0,
        1.0,
        2.0,
        -3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        4.0,
        2.5,
    )
    payload = struct.pack("<H", 1) + gate
    rx.on_track_data(payload)
    assert len(rx.data["track_gates"]) == 1
    gate_data = rx.data["track_gates"][0]
    assert gate_data["gate_id"] == 0
    assert gate_data["position_ned"] == (1.0, 2.0, -3.0)
    assert gate_data["width"] == 4.0
    assert gate_data["height"] == 2.5


def test_track_chunk_reassembly():
    rx = _rx()
    transfer_id = 7
    rx.expected_num_track_chunks[transfer_id] = 2
    rx.track_chunks[transfer_id] = {}

    gate = struct.pack(
        "<Hfffffffff",
        0,
        1.0,
        2.0,
        -3.0,
        1.0,
        0.0,
        0.0,
        0.0,
        4.0,
        2.5,
    )
    payload = struct.pack("<H", 1) + gate
    chunk_a = (
        struct.pack("<BH", ENCAPSULATED_TRACK_INFO_MSG_ID, transfer_id) + payload[:20]
    )
    chunk_b = (
        struct.pack("<BH", ENCAPSULATED_TRACK_INFO_MSG_ID, transfer_id) + payload[20:]
    )

    rx.on_track_data_packet(SimpleNamespace(data=bytearray(chunk_a), seqnr=0))
    assert "track_gates" not in rx.data
    rx.on_track_data_packet(SimpleNamespace(data=bytearray(chunk_b), seqnr=1))
    assert len(rx.data["track_gates"]) == 1


def test_on_collision_stores_event():
    rx = _rx()
    rx.on_collision(
        SimpleNamespace(id=1001, threat_level=2, horizontal_minimum_delta=1.25)
    )
    collision = rx.data["collision"]
    assert collision["id"] == 1001
    assert collision["threat_level"] == 2
    assert collision["impact"] == 1.25
    assert "received_at" in collision


def test_setup_requests_highres_imu():
    mav = MagicMock()
    conn = SimpleNamespace(target_system=1, target_component=1, mav=mav)
    from simulator.controller import Controller

    ctrl = Controller(conn, {}, system_boot_ms=1000)
    ctrl.pilot = MagicMock()
    ctrl.request_highres_imu(120)
    mav.command_long_send.assert_called_once()
    args = mav.command_long_send.call_args[0]
    assert args[2] == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert args[4] == mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU
