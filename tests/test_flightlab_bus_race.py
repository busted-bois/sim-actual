"""Race-GO wait helpers on the vertical harness bus (no live MAVLink)."""

from __future__ import annotations

import unittest

from flightlab.bus import Bus
from flightlab.state import StateTracker
from simulator.preflight import RACE_COUNTDOWN_MS


class BusRaceWaitTests(unittest.TestCase):
    def _bus(self, frames: list[dict]) -> Bus:
        bus = Bus.__new__(Bus)
        bus.tracker = StateTracker()
        self._frames = list(frames)
        self._idx = 0

        def drain():
            if self._idx < len(self._frames):
                race = self._frames[self._idx]
                self._idx += 1
            else:
                race = self._frames[-1]
            bus.tracker.race_status = race
            bus.tracker.data["race_status"] = race
            return bus.tracker.snapshot()

        bus.drain = drain  # type: ignore[method-assign]
        return bus

    def test_fresh_rejects_stale_already_go_restart(self):
        # Prior race already past GO; must not treat as fresh (early-arm bug).
        race_start = 1000
        bus = self._bus(
            [
                {
                    "sim_boot_time_ms": race_start + RACE_COUNTDOWN_MS + 500,
                    "race_start_boot_time_ms": race_start,
                }
            ]
        )
        self.assertFalse(
            bus.wait_for_fresh_race_start(timeout_s=0.2, is_restart=True)
        )

    def test_fresh_accepts_restart_countdown_in_progress(self):
        race_start = 2000
        bus = self._bus(
            [
                {
                    "sim_boot_time_ms": race_start + 500,
                    "race_start_boot_time_ms": race_start,
                }
            ]
        )
        bus.tracker.data["_preflight_race_start_baseline"] = 1000
        self.assertTrue(
            bus.wait_for_fresh_race_start(timeout_s=0.5, is_restart=True)
        )

    def test_race_go_first_run_blocks_until_scheduled_go(self):
        # race_start is future GO; must not add another +countdown (early/late miss).
        frames = [
            {"sim_boot_time_ms": 5000, "race_start_boot_time_ms": 8000},
            {"sim_boot_time_ms": 7000, "race_start_boot_time_ms": 8000},
            {"sim_boot_time_ms": 7999, "race_start_boot_time_ms": 8000},
        ]
        bus = self._bus(frames)
        self.assertFalse(bus.wait_for_race_go(timeout_s=0.25))

        bus = self._bus(
            frames
            + [{"sim_boot_time_ms": 8000, "race_start_boot_time_ms": 8000}]
        )
        self.assertTrue(bus.wait_for_race_go(timeout_s=1.0))

    def test_race_go_restart_blocks_until_countdown_done(self):
        race_start = 3000
        go = race_start + RACE_COUNTDOWN_MS
        bus = self._bus(
            [
                {
                    "sim_boot_time_ms": race_start + 1000,
                    "race_start_boot_time_ms": race_start,
                },
                {
                    "sim_boot_time_ms": go - 1,
                    "race_start_boot_time_ms": race_start,
                },
            ]
        )
        self.assertFalse(bus.wait_for_race_go(timeout_s=0.25, is_restart=True))

        bus = self._bus(
            [
                {
                    "sim_boot_time_ms": race_start + 1000,
                    "race_start_boot_time_ms": race_start,
                },
                {
                    "sim_boot_time_ms": go,
                    "race_start_boot_time_ms": race_start,
                },
            ]
        )
        self.assertTrue(bus.wait_for_race_go(timeout_s=1.0, is_restart=True))


if __name__ == "__main__":
    unittest.main()
