"""Tests for deploy fallback active-gate selection."""

import unittest

from rl.deploy import _fallback_gate_index


class DeployGateIndexTests(unittest.TestCase):
    def test_telemetry_wins_over_internal_index(self):
        self.assertEqual(_fallback_gate_index({"active_gate_index": 2}, 1), 2)

    def test_telemetry_zero_is_not_replaced(self):
        self.assertEqual(_fallback_gate_index({"active_gate_index": 0}, 2), 0)

    def test_malformed_telemetry_uses_internal_index(self):
        self.assertEqual(_fallback_gate_index({"active_gate_index": "bad"}, 2), 2)


if __name__ == "__main__":
    unittest.main()
