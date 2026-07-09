"""GateTransitionTracker timing tests - pure state machine, no sim needed."""

import math
import unittest

from simulator.gate_transition import (
    GatePhase,
    GateTransitionConfig,
    GateTransitionTracker,
)


def make_tracker():
    return GateTransitionTracker(
        GateTransitionConfig(r_acceptance=0.5, t_min=0.5, v_max=1.0)
    )


class TestGateTransitionTracker(unittest.TestCase):
    def test_starts_in_approach(self):
        tracker = make_tracker()
        self.assertEqual(tracker.phase, GatePhase.APPROACH)
        self.assertFalse(tracker.should_commit())

    def test_enters_hovering_when_close_and_slow(self):
        tracker = make_tracker()
        phase = tracker.update(d=0.2, speed=0.3, now=0.0)
        self.assertEqual(phase, GatePhase.HOVERING)

    def test_stays_approach_when_far(self):
        tracker = make_tracker()
        phase = tracker.update(d=2.0, speed=0.3, now=0.0)
        self.assertEqual(phase, GatePhase.APPROACH)

    def test_stays_approach_when_fast(self):
        tracker = make_tracker()
        phase = tracker.update(d=0.2, speed=2.0, now=0.0)
        self.assertEqual(phase, GatePhase.APPROACH)

    def test_no_commit_before_t_min(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        phase = tracker.update(d=0.2, speed=0.3, now=0.4)
        self.assertEqual(phase, GatePhase.HOVERING)
        self.assertFalse(tracker.should_commit())

    def test_commits_after_t_min(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        phase = tracker.update(d=0.2, speed=0.3, now=0.6)
        self.assertEqual(phase, GatePhase.COMMITTED)
        self.assertTrue(tracker.should_commit())

    def test_committed_moves_to_dead_reckon_next_tick(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        tracker.update(d=0.2, speed=0.3, now=0.6)
        phase = tracker.update(d=math.inf, speed=0.3, now=0.7)
        self.assertEqual(phase, GatePhase.DEAD_RECKON)

    def test_lost_gate_resets_thover(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        tracker.update(d=0.2, speed=0.3, now=0.4)  # thover = 0.4
        tracker.update(d=math.inf, speed=0.3, now=0.5)  # gate lost
        self.assertEqual(tracker.phase, GatePhase.APPROACH)
        self.assertEqual(tracker.thover, 0.0)
        # re-acquire: dwell must restart from zero
        tracker.update(d=0.2, speed=0.3, now=0.6)
        phase = tracker.update(d=0.2, speed=0.3, now=1.0)  # only 0.4s dwell
        self.assertEqual(phase, GatePhase.HOVERING)
        self.assertFalse(tracker.should_commit())

    def test_speed_spike_resets_thover(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        tracker.update(d=0.2, speed=0.3, now=0.4)
        tracker.update(d=0.2, speed=5.0, now=0.5)  # too fast
        self.assertEqual(tracker.phase, GatePhase.APPROACH)
        self.assertEqual(tracker.thover, 0.0)

    def test_no_t_max_dead_reckon_persists(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        tracker.update(d=0.2, speed=0.3, now=0.6)  # committed
        tracker.update(d=math.inf, speed=0.5, now=0.7)  # dead reckon
        # long time with no next gate - stays in dead reckon (t_min-only design)
        phase = tracker.update(d=math.inf, speed=0.5, now=100.0)
        self.assertEqual(phase, GatePhase.DEAD_RECKON)

    def test_next_gate_visible_exits_dead_reckon(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)
        tracker.update(d=0.2, speed=0.3, now=0.6)
        tracker.update(d=math.inf, speed=0.5, now=0.7)
        tracker.on_next_gate_visible()
        self.assertEqual(tracker.phase, GatePhase.APPROACH)
        self.assertEqual(tracker.thover, 0.0)

    def test_gate_passed_resets_from_any_phase(self):
        for setup_updates in (
            [(0.2, 0.3, 0.0)],  # hovering
            [(0.2, 0.3, 0.0), (0.2, 0.3, 0.6)],  # committed
            [(0.2, 0.3, 0.0), (0.2, 0.3, 0.6), (math.inf, 0.5, 0.7)],  # dead reckon
        ):
            tracker = make_tracker()
            for d, speed, now in setup_updates:
                tracker.update(d=d, speed=speed, now=now)
            tracker.on_gate_passed()
            self.assertEqual(tracker.phase, GatePhase.APPROACH)
            self.assertEqual(tracker.thover, 0.0)
            self.assertTrue(math.isinf(tracker.d))

    def test_first_update_accumulates_no_time(self):
        tracker = make_tracker()
        # first ever update has no dt reference - must not jump straight to commit
        phase = tracker.update(d=0.2, speed=0.3, now=1000.0)
        self.assertEqual(phase, GatePhase.HOVERING)
        self.assertEqual(tracker.thover, 0.0)

    def test_on_next_gate_visible_ignored_outside_dead_reckon(self):
        tracker = make_tracker()
        tracker.update(d=0.2, speed=0.3, now=0.0)  # hovering
        tracker.on_next_gate_visible()
        self.assertEqual(tracker.phase, GatePhase.HOVERING)


if __name__ == "__main__":
    unittest.main()
