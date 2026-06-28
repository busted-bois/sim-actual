import unittest
from unittest.mock import MagicMock

from simulator.pilot import Pilot


class PilotGatesPassedTests(unittest.TestCase):
    def test_gates_passed_property(self):
        controller = MagicMock()
        pilot = Pilot(controller, {})
        self.assertEqual(pilot.gates_passed, 0)
        pilot._gates_passed = 2
        self.assertEqual(pilot.gates_passed, 2)

    def test_may_fly_blocks_thrust_until_attempt_start(self):
        controller = MagicMock()
        pilot = Pilot(controller, {})
        pilot.tick()
        controller.set_attitude_rates.assert_called_with(0, 0, 0, 0)
        pilot.on_attempt_start()
        pilot.tick()
        self.assertGreater(controller.set_attitude_rates.call_count, 1)


if __name__ == "__main__":
    unittest.main()
