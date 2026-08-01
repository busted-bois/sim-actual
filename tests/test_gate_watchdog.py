import unittest
from unittest.mock import MagicMock, patch

from simulator.gate_watchdog import RESET_REPEAT_S, GateStallWatchdog

# race_monitor captures these from env at import, so the tests pin them instead
# of relying on whatever the ambient environment happens to say.
STALL_S = 10.0
GATE1_S = 20.0
SETTLE_S = 5.0


class FakePilot:
    """The duck-typed contract the watchdog needs from GPPilot."""

    def __init__(self, flying=True, gates_passed=0):
        self.flying = flying
        self.gates_passed = gates_passed
        self.resets = 0

    def reset_for_attempt(self):
        self.resets += 1


def _data(active=0, finish_ns=-1, gate_count=6):
    return {
        "active_gate_index": active,
        "gate_count": gate_count,
        "race_status": {
            "race_finish_time_ns": finish_ns,
            "sim_boot_time_ms": 1000,
        },
    }


@patch("simulator.gate_watchdog.run_meta", MagicMock())
@patch("simulator.gate_watchdog.SIM_RESET_WAIT_S", SETTLE_S)
@patch("simulator.race_monitor.GATE_PROGRESS_TIMEOUT_S", STALL_S)
@patch("simulator.race_monitor.GATE1_TIMEOUT_S", GATE1_S)
class GateStallWatchdogTests(unittest.TestCase):
    def _make(self, data, pilot=None, enabled=True):
        controller = MagicMock()
        pilot = pilot or FakePilot()
        wd = GateStallWatchdog(controller, pilot, data, enabled=enabled)
        return wd, controller, pilot

    def test_never_fires_while_not_flying(self):
        # Holding zero thrust on the pad before the countdown is not a stall.
        data = _data(active=0)
        wd, controller, pilot = self._make(data, FakePilot(flying=False))
        wd.tick(now=0.0)
        wd.tick(now=1000.0)
        controller.send_sim_reset_command.assert_not_called()
        self.assertEqual(pilot.resets, 0)

    def test_disabled_never_fires(self):
        data = _data(active=1)
        wd, controller, _ = self._make(data, enabled=False)
        wd.tick(now=0.0)
        wd.tick(now=1000.0)
        controller.send_sim_reset_command.assert_not_called()

    def test_no_reset_before_timeout(self):
        data = _data(active=1)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)  # arms the clock
        wd.tick(now=STALL_S - 1.0)
        controller.send_sim_reset_command.assert_not_called()

    def test_gate_stall_fires_after_timeout(self):
        data = _data(active=1)
        wd, controller, pilot = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=STALL_S + 0.5)
        controller.send_sim_reset_command.assert_called_once()
        self.assertEqual(pilot.resets, 1)
        self.assertEqual(wd.resets, 1)

    def test_gate_advance_rearms_clock(self):
        data = _data(active=1)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)
        data["active_gate_index"] = 2
        wd.tick(now=5.0)  # advance -> clock restarts at t=5
        wd.tick(now=5.0 + STALL_S - 1.0)
        controller.send_sim_reset_command.assert_not_called()
        wd.tick(now=5.0 + STALL_S + 0.5)
        controller.send_sim_reset_command.assert_called_once()

    def test_second_reset_sent_once_after_repeat_delay(self):
        data = _data(active=1)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=STALL_S)  # fires; first 31000
        self.assertEqual(controller.send_sim_reset_command.call_count, 1)
        wd.tick(now=STALL_S + RESET_REPEAT_S / 2)
        self.assertEqual(controller.send_sim_reset_command.call_count, 1)
        wd.tick(now=STALL_S + RESET_REPEAT_S)
        self.assertEqual(controller.send_sim_reset_command.call_count, 2)
        wd.tick(now=STALL_S + RESET_REPEAT_S + 1.0)
        self.assertEqual(controller.send_sim_reset_command.call_count, 2)

    def test_settle_expires_then_clock_rearms(self):
        data = _data(active=1)
        wd, controller, pilot = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=STALL_S)  # fires
        settle_end = STALL_S + RESET_REPEAT_S + SETTLE_S
        # Still settling: no re-fire even though the gate index never moved.
        wd.tick(now=settle_end - 0.1)
        self.assertEqual(pilot.resets, 1)
        wd.tick(now=settle_end)  # settle clears
        wd.tick(now=settle_end + 0.1)  # re-arms
        wd.tick(now=settle_end + STALL_S - 1.0)
        self.assertEqual(pilot.resets, 1)
        wd.tick(now=settle_end + STALL_S + 1.0)
        self.assertEqual(pilot.resets, 2)

    def test_gate1_uses_the_looser_budget(self):
        # Spawn is ~15 m before gate 0: the run to the first gate gets GATE1_S,
        # not the mid-course STALL_S.
        data = _data(active=0)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=STALL_S + 2.0)
        controller.send_sim_reset_command.assert_not_called()
        wd.tick(now=GATE1_S + 1.0)
        controller.send_sim_reset_command.assert_called_once()

    def test_gate1_softened_by_pilot_gate_count(self):
        # Pilot counted a pass the sim never registered — hold off (until the
        # 60 s hard cap inside gate1_fail).
        data = _data(active=0)
        wd, controller, _ = self._make(data, FakePilot(gates_passed=1))
        wd.tick(now=0.0)
        wd.tick(now=GATE1_S + 1.0)
        controller.send_sim_reset_command.assert_not_called()

    def test_course_complete_suppresses(self):
        data = _data(active=6, gate_count=6)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=100.0)
        controller.send_sim_reset_command.assert_not_called()

    def test_race_finish_suppresses(self):
        data = _data(active=2, finish_ns=100)
        wd, controller, _ = self._make(data)
        wd.tick(now=0.0)
        wd.tick(now=100.0)
        controller.send_sim_reset_command.assert_not_called()

    def test_outcome_recorded_before_pilot_reset(self):
        data = _data(active=1)
        wd, _, _ = self._make(data)
        with patch("simulator.gate_watchdog.run_meta") as meta:
            wd.tick(now=0.0)
            wd.tick(now=STALL_S + 0.5)
        meta.note_outcome.assert_called_once_with("gate_stall", active=1)

    def test_gate1_outcome_label(self):
        data = _data(active=0)
        wd, _, _ = self._make(data)
        with patch("simulator.gate_watchdog.run_meta") as meta:
            wd.tick(now=0.0)
            wd.tick(now=GATE1_S + 1.0)
        meta.note_outcome.assert_called_once_with("gate1_fail", active=0)

    def test_missing_flying_attribute_watches_anyway(self):
        # A pilot without the optional `flying` flag is treated as always flying.
        pilot = FakePilot()
        del pilot.flying
        data = _data(active=1)
        wd, controller, _ = self._make(data, pilot)
        wd.tick(now=0.0)
        wd.tick(now=STALL_S + 0.5)
        controller.send_sim_reset_command.assert_called_once()


class GPPilotFlyingContractTests(unittest.TestCase):
    def test_gp_pilot_exposes_flying(self):
        with patch.dict("os.environ", {"AUTO_PILOT": "gp"}, clear=False):
            from simulator.controller import Controller
            from simulator.gp_pilot import Phase

            ctrl = Controller(MagicMock(), {}, 0)
            try:
                pilot = ctrl.pilot
                self.assertFalse(pilot.flying)
                pilot.phase = Phase.FLYING
                self.assertTrue(pilot.flying)
                pilot.phase = Phase.BACKOFF
                self.assertTrue(pilot.flying)
                pilot.phase = Phase.WAIT_FOR_START
                self.assertFalse(pilot.flying)
            finally:
                ctrl.pilot.shutdown()


if __name__ == "__main__":
    unittest.main()
