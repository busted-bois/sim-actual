"""rl.calibration + consumer defaults (env/spec) — no sim required."""

import importlib
import json
import os
import tempfile
import unittest
from unittest import mock

import rl.calibration as calibration


class LoadCalibrationTests(unittest.TestCase):
    def test_absent_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "calibration.json")
            with mock.patch.object(calibration, "CAL_PATH", missing):
                self.assertEqual(calibration.load_calibration(), {})

    def test_garbage_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json")
            with mock.patch.object(calibration, "CAL_PATH", path):
                self.assertEqual(calibration.load_calibration(), {})

    def test_roundtrip_write_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calibration.json")
            data = {"hover_thrust": 0.27, "rate_tau_s": 0.06}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            with mock.patch.object(calibration, "CAL_PATH", path):
                cal = calibration.load_calibration()
        self.assertEqual(cal, {"hover_thrust": 0.27, "rate_tau_s": 0.06})


class ConsumerDefaultTests(unittest.TestCase):
    """rl.env / rl.spec constants: measured when the file exists, exact
    internal-model guesses when it does not."""

    def _reload_env_spec(self, cal: dict | None):
        import rl.env as env
        import rl.spec as spec

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "calibration.json")
        if cal is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cal, f)
        patcher = mock.patch.object(calibration, "CAL_PATH", path)
        patcher.start()

        def _restore():
            patcher.stop()
            importlib.reload(spec)
            importlib.reload(env)

        self.addCleanup(_restore)
        importlib.reload(spec)
        importlib.reload(env)
        return spec, env

    def test_defaults_without_calibration(self):
        spec, env = self._reload_env_spec(None)
        self.assertEqual(spec.HOVER_THRUST, 0.27)
        self.assertAlmostEqual(env.THRUST_ACCEL, spec.GRAVITY / 0.27, places=9)
        self.assertEqual(env.RATE_TAU, 0.05)

    def test_measured_values_picked_up(self):
        spec, env = self._reload_env_spec(
            {"hover_thrust": 0.27, "thrust_accel": 36.32, "rate_tau_s": 0.06}
        )
        self.assertEqual(spec.HOVER_THRUST, 0.27)
        self.assertEqual(env.THRUST_ACCEL, 36.32)
        self.assertEqual(env.RATE_TAU, 0.06)

    def test_partial_calibration_mixes_measured_and_default(self):
        spec, env = self._reload_env_spec({"hover_thrust": 0.27})
        self.assertEqual(spec.HOVER_THRUST, 0.27)
        # thrust_accel not measured -> derived from the measured hover point.
        self.assertAlmostEqual(env.THRUST_ACCEL, spec.GRAVITY / 0.27, places=9)
        self.assertEqual(env.RATE_TAU, 0.05)


if __name__ == "__main__":
    unittest.main()
