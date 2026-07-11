import time
import unittest
from unittest.mock import MagicMock, patch

from simulator.vq2_pilot import (
    CRUISE_PITCH_RATE,
    HOVER_T,
    SIGN_YAW,
    VISION_YAW_GAIN,
    VQ2VisionPilot,
    _AttitudeEstimator,
    _AltitudeEstimator,
)


class VQ2PilotTests(unittest.TestCase):
    def _make_pilot(self, data=None):
        controller = MagicMock()
        data = data or {"armed": True}
        pilot = VQ2VisionPilot(controller, data)
        return pilot, controller, data

    def _vision_data(self, nx=0.0, ny=0.0, r_frac=0.05):
        return {
            "armed": True,
            "imu": {
                "time_us": 1_000_000,
                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,
                "ax": 0.0,
                "ay": 0.0,
                "az": 9.81,
                "pressure_alt": 100.0,
            },
            "gate_target": {
                "detected": True,
                "nx": nx,
                "ny": ny,
                "r_frac": r_frac,
            },
            "camera": {"received_at": time.monotonic()},
        }

    def test_complementary_filter_levels_from_accel(self):
        est = _AttitudeEstimator()
        imu = {
            "time_us": 1_000_000,
            "gx": 0.0,
            "gy": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": 9.81,
        }
        roll, pitch = est.update(imu)
        self.assertAlmostEqual(roll, 0.0, places=2)
        self.assertAlmostEqual(pitch, 0.0, places=2)

        imu2 = {
            "time_us": 1_010_000,
            "gx": 0.0,
            "gy": 0.0,
            "ax": 0.0,
            "ay": 4.9,
            "az": 8.5,
        }
        roll2, _ = est.update(imu2)
        self.assertGreater(roll2, 0.1)

    def test_yaw_cmd_sign_gate_right_negative(self):
        pilot, controller, data = self._make_pilot(self._vision_data(nx=0.5))
        pilot.on_attempt_start()
        pilot.tick()
        _, _, yaw_cmd, _ = controller.set_attitude_rates.call_args[0]
        expected = SIGN_YAW * VISION_YAW_GAIN * 0.5
        self.assertLess(yaw_cmd, 0.0)
        self.assertAlmostEqual(yaw_cmd, expected, places=4)

    def test_thrust_rises_when_below_target_altitude(self):
        alt = _AltitudeEstimator()
        alt.reset(-5.0)
        alt._pressure_ref = 100.0
        alt.z = -3.0
        thrust_high = alt.thrust(-5.0)
        alt.z = -8.0
        thrust_low = alt.thrust(-5.0)
        self.assertGreater(thrust_high, HOVER_T)
        self.assertLess(thrust_low, thrust_high)

    def test_vision_z_target_anchored_to_hold_z(self):
        pilot, controller, data = self._make_pilot(self._vision_data(ny=0.0))
        pilot.on_attempt_start()
        with patch.object(pilot._alt, "thrust", wraps=pilot._alt.thrust) as mock_thrust:
            pilot.tick()
            mock_thrust.assert_called_with(-5.0, cruise=False)

    def test_advance_uses_cruise_pitch_rate(self):
        pilot, controller, data = self._make_pilot(
            self._vision_data(nx=0.0, ny=0.0, r_frac=0.08)
        )
        pilot.on_attempt_start()
        pilot._advancing = True
        pilot.tick()
        _, pitch_cmd, _, _ = controller.set_attitude_rates.call_args[0]
        self.assertLess(pitch_cmd, 0.0)
        self.assertAlmostEqual(pitch_cmd, CRUISE_PITCH_RATE, places=3)

    def test_gates_passed_follows_active_gate_index(self):
        pilot, _, data = self._make_pilot({"armed": True, "active_gate_index": 2})
        self.assertEqual(pilot.gates_passed, 2)

    def test_r_frac_drop_does_not_increment_gates_passed(self):
        pilot, _, data = self._make_pilot(
            self._vision_data(nx=0.0, ny=0.0, r_frac=0.02)
        )
        pilot.on_attempt_start()
        data["active_gate_index"] = 0
        pilot.tick()
        self.assertEqual(pilot.gates_passed, 0)

    def test_sim_gate_advance_triggers_post_gate(self):
        pilot, controller, data = self._make_pilot({"armed": True, "active_gate_index": 0})
        pilot.on_attempt_start()
        data["active_gate_index"] = 1
        pilot.tick()
        self.assertEqual(pilot._mode_str, "post_gate")
        self.assertIsNotNone(pilot._post_gate_time)

    def test_search_engages_without_detection(self):
        pilot, controller, data = self._make_pilot(
            {
                "armed": True,
                "imu": {
                    "time_us": 1_000_000,
                    "gx": 0.0,
                    "gy": 0.0,
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 9.81,
                    "pressure_alt": 50.0,
                },
            }
        )
        pilot.on_attempt_start()
        pilot._post_gate_time = time.monotonic() - 3.0
        pilot.tick()
        self.assertEqual(pilot._mode_str, "search")
        controller.set_attitude_rates.assert_called()

    def test_reset_clears_state(self):
        pilot, _, data = self._make_pilot(self._vision_data())
        pilot._searching = True
        pilot._advancing = True
        data["gate_target"] = {"detected": True}
        pilot.reset_for_attempt()
        self.assertFalse(pilot._searching)
        self.assertFalse(pilot._advancing)
        self.assertNotIn("gate_target", data)

    def test_hover_levels_attitude_not_zero_rates(self):
        pilot, controller, data = self._make_pilot(
            {
                "armed": True,
                "imu": {
                    "time_us": 1_000_000,
                    "gx": 0.0,
                    "gy": 0.0,
                    "ax": 0.0,
                    "ay": 4.9,
                    "az": 8.5,
                },
            }
        )
        pilot.on_attempt_start()
        pilot._att.update(data["imu"])
        pilot._hover_level()
        roll_cmd, pitch_cmd, yaw_cmd, _ = controller.set_attitude_rates.call_args[0]
        self.assertNotEqual(roll_cmd, 0.0)
        self.assertEqual(yaw_cmd, 0.0)


if __name__ == "__main__":
    unittest.main()
