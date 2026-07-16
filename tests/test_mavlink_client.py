"""Tests for mavlink client and VQ2 telemetry profile."""

import time
import unittest
from unittest import mock

from flightlab.bus import Bus, VQ2_FALLBACK_S
from flightlab.controllers import _thrust
from flightlab.safety import HOVER_THRUST
from flightlab.state import State, Target
from simulator.mavlink_rx import MAVLinkRX


class EkfThrustTests(unittest.TestCase):
    def test_thrust_closes_loop_for_ekf_pose(self):
        s = State(
            t_mono=time.monotonic(),
            armed=True,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            roll_rate=0.0,
            pitch_rate=0.0,
            yaw_rate=0.0,
            pos_ned=(0.0, 0.0, 0.0),
            vel_ned=(0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            pose_age_s=0.0,
            alt_trusted=False,
            pose_source="ekf",
        )
        t = _thrust(s, Target(z=-3.0))
        self.assertGreater(t, HOVER_THRUST)


class GcsHeartbeatFilterTests(unittest.TestCase):
    def test_gcs_heartbeat_does_not_set_armed(self):
        rx = MAVLinkRX(None, {})
        msg = mock.MagicMock()
        msg.type = 6  # MAV_TYPE_GCS
        msg.base_mode = 0b10000000
        rx.on_heartbeat(msg)
        self.assertNotIn("armed", rx.data)


class RecvResilienceTests(unittest.TestCase):
    def test_receive_loop_survives_connection_reset(self):
        # Windows raises ConnectionResetError on UDP recv after an ICMP
        # port-unreachable (sim restart). The loop must keep listening and
        # handle later traffic, not die.
        conn = mock.MagicMock()
        rx = MAVLinkRX(conn, {})
        rx.is_running = True

        hb = mock.MagicMock()
        hb.get_type.return_value = "HEARTBEAT"
        hb.type = 2  # quadrotor, not GCS
        hb.base_mode = 0b10000000

        calls = {"n": 0}

        def recv(blocking=False):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionResetError("simulated ICMP port-unreachable")
            if calls["n"] == 2:
                return hb
            rx.is_running = False
            return None

        conn.recv_match.side_effect = recv
        rx.mavlink_receive_loop()  # must not raise or bail on call 1
        self.assertGreaterEqual(calls["n"], 3)
        self.assertTrue(rx.data.get("armed"))  # post-reset message handled

    def test_vehicle_heartbeat_sets_armed(self):
        rx = MAVLinkRX(None, {})
        msg = mock.MagicMock()
        msg.type = 2  # MAV_TYPE_QUADROTOR
        msg.base_mode = 0b10000000
        rx.on_heartbeat(msg)
        self.assertTrue(rx.data["armed"])


class Vq2FallbackTests(unittest.TestCase):
    def test_fallback_delay_is_short(self):
        self.assertLessEqual(VQ2_FALLBACK_S, 3.0)

    def test_try_vq2_fallback_when_ready(self):
        with mock.patch.object(Bus, "__init__", lambda self: None):
            bus = Bus()
        bus.data = {
            "race_status": {"race_start_boot_time_ms": 1000},
            "frame": {"img": None, "sim_time_ns": 1},
        }
        bus.estimator = mock.MagicMock()
        bus.estimator.ready = True
        bus.vq2_mode = False
        bus.pose_mode = ""
        bus._seen = {
            "imu": True,
            "attitude": False,
            "odometry": False,
            "local_position": False,
            "estimator": True,
        }
        with mock.patch("flightlab.bus.vision_ready", return_value=True):
            self.assertTrue(bus._try_vq2_fallback())
        self.assertTrue(bus.vq2_mode)
        self.assertEqual(bus.pose_mode, "vq2")

    def test_ekf_read_state_alt_trusted_in_vq2(self):
        with mock.patch.object(Bus, "__init__", lambda self: None):
            bus = Bus()
        bus.data = {
            "armed": True,
            "imu": {"ax": 0, "ay": 0, "az": -9.81, "gx": 0, "gy": 0, "gz": 0},
        }
        bus.vq2_mode = True
        bus._pose_mono = None
        bus._last_stamp = None
        bus.estimator = mock.MagicMock()
        bus.estimator.ready = True
        bus.estimator.pose.return_value = (
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )
        s = bus._read_state()
        assert s is not None
        self.assertEqual(s.pose_source, "ekf")
        self.assertTrue(s.alt_trusted)


if __name__ == "__main__":
    unittest.main()
