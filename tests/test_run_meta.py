"""Unit tests for the run sidecar (simulator/run_meta).

A published recording used to say nothing about what the run did. These lock in
the JSON that replaced that, and -- more importantly -- that producing it can
never interfere with flying.
"""

import json
import os
import unittest
from unittest.mock import patch

from simulator import run_meta


class _Sidecar:
    """Point run_meta at a temp file and give it a clean slate."""

    def __init__(self, case):
        self._tmp = os.path.join(
            os.path.dirname(__file__), f"_meta_tmp_{os.getpid()}"
        )
        self._case = case

    def __enter__(self):
        os.makedirs(self._tmp, exist_ok=True)
        self.path = os.path.join(self._tmp, "vision_test.json")
        self._patches = [
            patch.object(run_meta, "_DIR", self._tmp),
            patch.object(run_meta, "_PATH", self.path),
            patch.object(run_meta, "_finalized", False),
            patch.dict(
                run_meta._state,
                {"attempts": [], "video": None, "ended_utc": None},
                clear=False,
            ),
        ]
        for p in self._patches:
            p.start()
        run_meta._state["attempts"] = []
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        # The atexit hook runs at interpreter shutdown with the real _PATH
        # restored -- leave no recorded state for it to publish.
        run_meta._state["attempts"] = []
        run_meta._state["video"] = None
        for f in os.listdir(self._tmp):
            os.remove(os.path.join(self._tmp, f))
        os.rmdir(self._tmp)
        return False

    def read(self):
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)


class SidecarTests(unittest.TestCase):
    def test_attempt_lifecycle_records_gates_and_reason(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "rl/data/gp_log_20260101_000000_a1.csv")
            run_meta.note_attempt_end(gates=4, reason="reset")
            att = sc.read()["attempts"]
            self.assertEqual(len(att), 1)
            self.assertEqual(att[0]["n"], 1)
            # Basename only -- the publish step re-homes it under telemetry/.
            self.assertEqual(att[0]["telemetry"], "gp_log_20260101_000000_a1.csv")
            self.assertEqual(att[0]["gates_passed"], 4)
            self.assertEqual(att[0]["end_reason"], "reset")

    def test_several_attempts_accumulate(self):
        # One process resets and reopens its log per attempt, so the sidecar
        # needs a list, not a scalar.
        with _Sidecar(self) as sc:
            for n in (1, 2, 3):
                run_meta.note_attempt_start(n, f"gp_log_x_a{n}.csv")
                run_meta.note_attempt_end(gates=n)
            data = sc.read()
            self.assertEqual(len(data["attempts"]), 3)
            self.assertEqual(data["totals"]["attempts"], 3)
            self.assertEqual(data["totals"]["best_gates"], 3)

    def test_outcome_without_a_telemetry_log_still_records(self):
        # make auto flies IBVS, which writes no CSV -- the outcome must still
        # land somewhere, since otherwise it only ever reached stdout.
        with _Sidecar(self) as sc:
            run_meta.note_outcome("gate1_fail", attempt=1, active=0)
            att = sc.read()["attempts"]
            self.assertEqual(len(att), 1)
            self.assertEqual(att[0]["outcome"], "gate1_fail")
            self.assertIsNone(att[0]["telemetry"])

    def test_outcome_attaches_to_the_matching_open_attempt(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "a.csv")
            run_meta.note_outcome("success", attempt=1, lap_s=42.123, active=6)
            att = sc.read()["attempts"]
            self.assertEqual(len(att), 1, "should not fork a second attempt")
            self.assertEqual(att[0]["outcome"], "success")
            self.assertEqual(att[0]["lap_s"], 42.12)

    def test_video_stats_land_with_byte_size(self):
        with _Sidecar(self) as sc:
            mp4 = os.path.join(sc._tmp, "v.mp4")
            with open(mp4, "wb") as fh:
                fh.write(b"x" * 1234)
            run_meta.note_video(
                {"path": mp4, "fps_nominal": 30.0, "fps_actual": 22.7, "frames": 100}
            )
            vid = sc.read()["video"]
            self.assertEqual(vid["bytes"], 1234)
            self.assertEqual(vid["fps_actual"], 22.7)
            self.assertNotIn("path", vid, "absolute local path must not be published")

    def test_finalize_stamps_end_and_process_identity(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "a.csv")
            run_meta.finalize()
            data = sc.read()
            self.assertIsNotNone(data["ended_utc"])
            self.assertIsNotNone(data["entry"])
            self.assertIn("sha", data["git"])

    def test_finalize_is_idempotent(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "a.csv")
            run_meta.finalize()
            first = sc.read()["ended_utc"]
            run_meta.finalize()
            self.assertEqual(sc.read()["ended_utc"], first)

    def test_finalize_writes_nothing_when_no_run_happened(self):
        # Merely importing this module (a test, a lint pass) must not litter
        # runs/videos/ with sidecars for processes that never flew.
        with _Sidecar(self) as sc:
            run_meta.finalize()
            self.assertFalse(os.path.exists(sc.path))

    def test_write_is_atomic_so_a_killed_run_stays_parseable(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "a.csv")
            # No .tmp left behind, and what is on disk parses.
            self.assertEqual(
                [f for f in os.listdir(sc._tmp) if f.endswith(".tmp")], []
            )
            self.assertIsInstance(sc.read(), dict)


class NeverGroundsThePilotTests(unittest.TestCase):
    """The contract that actually matters: sidecar trouble is swallowed."""

    def test_unwritable_target_does_not_raise(self):
        with _Sidecar(self):
            with patch.object(
                run_meta, "_write", side_effect=OSError("read-only filesystem")
            ):
                run_meta.note_attempt_start(1, "a.csv")
                run_meta.note_attempt_end(gates=1)
                run_meta.note_outcome("gate_stall", attempt=1)
                run_meta.note_video({"path": None, "frames": 1})
                run_meta.finalize()

    def test_garbage_input_does_not_raise(self):
        with _Sidecar(self):
            run_meta.note_attempt_start(None, None)
            run_meta.note_outcome("x", attempt=None, lap_s="not-a-number")
            run_meta.note_video({"path": "/does/not/exist", "frames": 1})

    def test_note_video_ignores_an_empty_stats_dict(self):
        # display.stats() returns {} when nothing was recorded.
        with _Sidecar(self) as sc:
            run_meta.note_video({})
            self.assertFalse(os.path.exists(sc.path))


if __name__ == "__main__":
    unittest.main()
