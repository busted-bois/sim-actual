import math
import unittest
from unittest.mock import MagicMock

from rl.fly2_course import HOVER_T
from simulator.controller import Controller
from simulator.state_estimator import GRAVITY, StateEstimator
from simulator.vio_pilot import VIOPilot
from simulator.vision_nav import Cmd


def _make_ready_estimator() -> StateEstimator:
    """Feed enough clean, level, unpowered ground samples to boot the ESKF."""
    est = StateEstimator(init_samples=50)
    for k in range(50):
        est.on_imu(
            dict(
                ax=0.0,
                ay=0.0,
                az=-GRAVITY,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                mx=float("nan"),
                my=float("nan"),
                mz=float("nan"),
                abs_pressure=float("nan"),
                pressure_alt=float("nan"),
                temperature=float("nan"),
                time_us=k,
            )
        )
    assert est.ready
    return est


class VIOPilotTests(unittest.TestCase):
    def _make_pilot(self, data, estimator=None):
        controller = MagicMock()
        pilot = VIOPilot(controller, data, estimator)
        controller.reset_mock()
        return pilot, controller

    def test_gates_passed_tracks_guidance(self):
        pilot, _ = self._make_pilot({})
        self.assertEqual(pilot.gates_passed, 0)
        pilot.guide.passed.append(None)
        self.assertEqual(pilot.gates_passed, 1)

    def test_tick_hovers_and_publishes_not_ready_before_boot(self):
        pilot, controller = self._make_pilot({})
        pilot.tick()
        controller.set_attitude_rates.assert_called_once_with(0, 0, 0, HOVER_T)
        self.assertEqual(pilot.data["vio"], {"ready": False})

    def test_tick_publishes_pose_once_estimator_ready(self):
        data = {}
        pilot, _ = self._make_pilot(data, _make_ready_estimator())
        pilot.tick()
        self.assertTrue(data["vio"]["ready"])
        self.assertEqual(len(data["vio"]["pos_ned"]), 3)
        self.assertEqual(len(data["vio"]["quat"]), 4)

    def test_tick_converts_guidance_cmd_to_rates(self):
        pilot, controller = self._make_pilot({}, _make_ready_estimator())
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, -0.1, 0.0, 0.0, "GO")
        pilot.guide.n_passed = 0
        pilot.guide.last_matches = []
        pilot.tick()
        args = controller.set_attitude_rates.call_args[0]
        self.assertTrue(all(math.isfinite(a) for a in args))

    def test_tick_feeds_last_matches_into_landmark_update(self):
        import numpy as np

        pilot, _ = self._make_pilot({}, _make_ready_estimator())
        pilot.estimator.update_landmark = MagicMock(return_value=True)
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, 0.0, 0.0, 0.0, "GO")
        pilot.guide.n_passed = 0
        pilot.guide.last_matches = [(np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]))]
        pilot.tick()
        pilot.estimator.update_landmark.assert_called_once()

    def test_tick_passes_pose_gates_to_guidance_once_per_frame(self):
        gates = [{"conf": 0.9, "pose": {"gate_pos_body": [5.0, 0.0, 0.0]}}]
        data = {"pose": {"frame_id": 1, "gates": gates}}
        pilot, _ = self._make_pilot(data, _make_ready_estimator())
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, 0.0, 0.0, 0.0, "GO")
        pilot.guide.n_passed = 0
        pilot.guide.last_matches = []
        pilot.tick()
        self.assertIs(pilot.guide.update.call_args[0][0], gates)
        pilot.tick()
        self.assertEqual(pilot.guide.update.call_args[0][0], [])
        data["pose"] = {"frame_id": 2, "gates": gates}
        pilot.tick()
        self.assertIs(pilot.guide.update.call_args[0][0], gates)

    def test_collision_notifies_estimator_once_per_change(self):
        data = {"last_collision": 1}
        pilot, _ = self._make_pilot(data, _make_ready_estimator())
        pilot.estimator.notify_collision = MagicMock()
        pilot.guide = MagicMock()
        pilot.guide.update.return_value = Cmd(0.0, 0.0, 0.0, 0.0, "GO")
        pilot.guide.n_passed = 0
        pilot.guide.last_matches = []
        pilot.tick()
        pilot.estimator.notify_collision.assert_called_once()
        pilot.tick()
        pilot.estimator.notify_collision.assert_called_once()  # not re-fired
        data["last_collision"] = 2
        pilot.tick()
        self.assertEqual(pilot.estimator.notify_collision.call_count, 2)

    def test_reset_for_attempt_clears_stale_vision_state(self):
        data = {"gate_target": {"detected": True}, "pose": {"gates": []}, "vio": {}}
        pilot, controller = self._make_pilot(data, _make_ready_estimator())
        pilot.guide.passed.append(None)
        pilot.reset_for_attempt()
        self.assertNotIn("gate_target", data)
        self.assertNotIn("pose", data)
        self.assertNotIn("vio", data)
        self.assertEqual(pilot.gates_passed, 0)
        self.assertFalse(pilot.estimator.ready)
        controller.set_attitude_rates.assert_called_with(0, 0, 0, HOVER_T)


class ControllerThrustFeedTests(unittest.TestCase):
    def test_update_feeds_commanded_thrust_into_estimator(self):
        estimator = MagicMock()
        ctrl = Controller(MagicMock(), {}, 0, estimator=estimator)
        ctrl.pilot = MagicMock()
        ctrl.set_control_mode("attitude")
        ctrl.set_attitude_rates(0.0, 0.0, 0.0, 0.42)
        ctrl.update()
        self.assertAlmostEqual(estimator.thrust_cmd, 0.42, places=6)


if __name__ == "__main__":
    unittest.main()
