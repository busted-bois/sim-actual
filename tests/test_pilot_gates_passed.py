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


if __name__ == "__main__":
    unittest.main()
