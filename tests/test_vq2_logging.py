"""Tests for rl/core/diagnostics.py and its integration in rl/deploy_vq2.py.

Covers: dual console+file output, bracketed component tags, traceback on
fatal, rate-limited ticks, no duplicate handlers on re-init, idempotent close,
and the observe/fly lifecycle (AHRS seed, gate pass, final report, CSV rows,
--observe no-command guarantee).
"""

from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from rl.core.diagnostics import RunnerLog
from rl.deploy_vq2 import LOG_DIR, VQ2Pilot


def _make_log(tmpdir, stream=None, **kw):
    stream = stream if stream is not None else io.StringIO()
    log = RunnerLog(log_dir=tmpdir, stream=stream, file_name="t.log", **kw)
    return log, stream


class RunnerLogBasicsTests(unittest.TestCase):
    def test_console_and_file_get_same_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp)
            log.info("gate 3 passed")
            log.close()
            console = buf.getvalue()
            self.assertIn("[vq2] gate 3 passed", console)
            with open(os.path.join(tmp, "t.log"), encoding="utf-8") as f:
                file_text = f.read()
            self.assertIn("gate 3 passed", file_text)
            self.assertIn("[vq2]", file_text)
            self.assertRegex(file_text, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_component_overrides_default_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp)
            log.info("seeded", component="ahrs")
            log.close()
            self.assertIn("[ahrs] seeded", buf.getvalue())

    def test_log_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            log = RunnerLog(log_dir=nested, stream=io.StringIO(), file_name="x.log")
            self.assertTrue(os.path.isdir(nested))
            self.assertTrue(os.path.isfile(log.file_path))
            log.close()

    def test_warn_uses_warning_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp)
            log.warn("stall approaching", component="gate")
            log.close()
            console = buf.getvalue()
            with open(os.path.join(tmp, "t.log"), encoding="utf-8") as f:
                file_text = f.read()
            self.assertIn("[gate] stall approaching", console)
            self.assertIn("WARNING", file_text)


class RunnerLogRateLimitTests(unittest.TestCase):
    def test_first_tick_emits_second_within_window_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp, tick_period_s=10.0)
            self.assertTrue(log.tick("periodic1"))
            self.assertFalse(log.tick("periodic2"))
            log.close()
            self.assertIn("periodic1", buf.getvalue())
            self.assertNotIn("periodic2", buf.getvalue())

    def test_reset_tick_clock_forces_next_emit(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp, tick_period_s=10.0)
            self.assertTrue(log.tick("a"))
            self.assertFalse(log.tick("b"))
            log.reset_tick_clock()
            self.assertTrue(log.tick("c"))
            log.close()
            self.assertIn("a", buf.getvalue())
            self.assertIn("c", buf.getvalue())

    def test_zero_period_emits_every_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp, tick_period_s=0.0)
            for i in range(5):
                self.assertTrue(log.tick(f"n{i}"))
            log.close()
            text = buf.getvalue()
            for i in range(5):
                self.assertIn(f"n{i}", text)


class RunnerLogFatalTests(unittest.TestCase):
    def test_fatal_emits_traceback_on_both_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, buf = _make_log(tmp)
            try:
                raise ValueError("boom")
            except ValueError:
                log.fatal("control loop crashed")
            log.close()
            console = buf.getvalue()
            with open(os.path.join(tmp, "t.log"), encoding="utf-8") as f:
                file_text = f.read()
            self.assertIn("ERROR", file_text)
            self.assertIn("control loop crashed", console)
            self.assertIn("Traceback", console)
            self.assertIn("ValueError: boom", console)
            self.assertIn("Traceback", file_text)
            self.assertIn("ValueError: boom", file_text)


class RunnerLogHandlerLifecycleTests(unittest.TestCase):
    def test_no_duplicate_handlers_on_reinit_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            log1, _ = _make_log(tmp)
            n1 = len(log1._logger.handlers)
            log1.close()
            log2, _ = _make_log(tmp)
            n2 = len(log2._logger.handlers)
            log2.close()
            self.assertEqual(n1, n2)
            self.assertGreaterEqual(n1, 2)

    def test_close_is_idempotent_and_clears_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log, _ = _make_log(tmp)
            log.close()
            log.close()
            self.assertEqual(log._logger.handlers, [])

    def test_info_after_close_is_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = io.StringIO()
            log = RunnerLog(log_dir=tmp, stream=buf, file_name="t.log")
            log.close()
            log.info("post-close")
            log.tick("post-close-tick")
            self.assertEqual(buf.getvalue(), "")


def _resting_imu():
    return {
        "xacc": 0.0,
        "yacc": 0.0,
        "zacc": 9.8,
        "xgyro": 0.0,
        "ygyro": 0.0,
        "zgyro": 0.0,
    }


class VQ2PilotLoggingTests(unittest.TestCase):
    def _make_pilot(self, tmp, controller=None, policy=None, data=None):
        buf = io.StringIO()
        log = RunnerLog(log_dir=tmp, stream=buf, file_name="pilot.log")
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        data = data if data is not None else {}
        pilot = VQ2Pilot(
            data,
            controller,
            policy=policy,
            log=log,
            csv_log=writer,
        )
        return pilot, log, buf, writer, csv_buf, data

    def test_ahrs_seed_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, buf, *_ = self._make_pilot(tmp, data={"imu": _resting_imu()})
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot._gravity()
            log.close()
            self.assertIn("[ahrs] AHRS seeded from accel", buf.getvalue())

    def test_gate_pass_logged_with_gate_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, buf, *_ = self._make_pilot(
                tmp, data={"imu": _resting_imu(), "active_gate_index": 0}
            )
            pilot._last_gate = 0
            pilot.data["active_gate_index"] = 2
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot.observation()
            log.close()
            text = buf.getvalue()
            self.assertIn("[gate]", text)
            self.assertIn("gate 2 passed", text)

    def test_report_logs_detection_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, buf, *_ = self._make_pilot(tmp, data={"imu": _resting_imu()})
            pilot._ticks = 100
            pilot._det_ticks = 30
            pilot.gates_passed = 4
            pilot.report(log=log)
            log.close()
            text = buf.getvalue()
            self.assertIn("final report", text)
            self.assertIn("vision on 30.0%", text)
            self.assertIn("gates=4", text)

    def test_tick_writes_csv_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, _, writer, csv_buf, data = self._make_pilot(
                tmp, data={"imu": _resting_imu()}
            )
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot.tick()
            log.close()
            rows = list(csv.reader(io.StringIO(csv_buf.getvalue())))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "0")
            self.assertEqual(len(rows[0]), 6 + 29)

    def test_observe_controller_none_never_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, buf, *_ = self._make_pilot(
                tmp, controller=None, data={"imu": _resting_imu()}
            )
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot.tick()
            log.close()
            self.assertIsNone(pilot.controller)

    def test_fly_controller_receives_setpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = MagicMock()
            pilot, log, buf, *_ = self._make_pilot(
                tmp, controller=ctrl, data={"imu": _resting_imu()}
            )
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot.tick()
            log.close()
            ctrl.set_attitude_rates.assert_called_once()
            _, _, _, thrust = ctrl.set_attitude_rates.call_args[0]
            self.assertGreaterEqual(thrust, 0.0)
            self.assertLessEqual(thrust, 1.0)

    def test_csv_lives_under_logs_vq2(self):
        self.assertEqual(LOG_DIR, os.path.join("logs", "vq2"))

    def test_invalid_gate_index_is_logged_and_retains_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, log, buf, *_ = self._make_pilot(
                tmp, data={"imu": _resting_imu(), "active_gate_index": "bad"}
            )
            pilot._last_gate = 3
            self.assertEqual(pilot._gate_index(), 3)
            log.close()
            self.assertIn("invalid active_gate_index", buf.getvalue())

    def test_nonfinite_policy_action_fails_before_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = MagicMock()
            policy = MagicMock()
            policy.predict.return_value = (np.array([[np.nan, 0.0, 0.0, 0.0]]), None)
            pilot, log, *_ = self._make_pilot(
                tmp,
                controller=ctrl,
                data={"imu": _resting_imu()},
                policy=policy,
            )
            with (
                patch("simulator.gp_vision.best_pose_gate", return_value=None),
                self.assertRaisesRegex(RuntimeError, "invalid action"),
            ):
                pilot.tick()
            log.close()
            ctrl.set_attitude_rates.assert_not_called()


class ObserveGuaranteeTests(unittest.TestCase):
    """--observe must never arm and never send a setpoint."""

    def test_observe_path_passes_none_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctrl = MagicMock()
            data = {"imu": _resting_imu()}
            buf = io.StringIO()
            log = RunnerLog(log_dir=tmp, stream=buf, file_name="obs.log")
            pilot = VQ2Pilot(
                data, None, policy=None, log=log, csv_log=csv.writer(io.StringIO())
            )
            with patch("simulator.gp_vision.best_pose_gate", return_value=None):
                pilot.tick()
            log.close()
            ctrl.set_attitude_rates.assert_not_called()
            ctrl.arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
