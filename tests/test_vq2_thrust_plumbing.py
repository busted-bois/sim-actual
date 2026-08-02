"""Previous applied thrust is forwarded into VQ2 controlled prediction."""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock

from rl.experts.fly2_course import Fly2CoursePilot
from simulator.controller import Controller
from simulator.vision_nav_pilot import VisionNavPilot


class ControllerThrustAccessorTests(unittest.TestCase):
    def test_reads_last_set_thrust(self):
        ctrl = types.SimpleNamespace(_thrust=0.35)
        prop = Controller.last_thrust.fget(ctrl)
        self.assertAlmostEqual(prop, 0.35)


class PilotThrustPlumbingTests(unittest.TestCase):
    def test_vision_nav_pilot_passes_thrust(self):
        ctrl = MagicMock()
        ctrl.last_thrust = 0.35
        pilot = VisionNavPilot(ctrl, {"odometry": None})
        pilot._pose = MagicMock()
        pilot.gate_map = [{"pos": [0, 0, -5]}]
        pilot._current_odometry()
        pilot._pose.tick.assert_called_once()
        _, kwargs = pilot._pose.tick.call_args
        self.assertAlmostEqual(kwargs["thrust_cmd"], 0.35)

    def test_fly2_pilot_passes_thrust(self):
        ctrl = MagicMock()
        ctrl.last_thrust = 0.28
        pilot = Fly2CoursePilot(ctrl, {})
        pilot._pose = MagicMock()
        pilot.gate_map = [{"pos": [0, 0, -5]}]
        pilot._current_odometry()
        pilot._pose.tick.assert_called_once()
        _, kwargs = pilot._pose.tick.call_args
        self.assertAlmostEqual(kwargs["thrust_cmd"], 0.28)


if __name__ == "__main__":
    unittest.main()
