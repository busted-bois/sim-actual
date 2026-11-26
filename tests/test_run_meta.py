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
            # Identity is captured once per process; reset it per test so each
            # gets a clean slate the way a fresh flight process would.
            patch.object(run_meta, "_identified", False),
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


class ConfigCaptureTests(unittest.TestCase):
    """What made two runs at the same sha indistinguishable, and now doesn't."""

    def test_finalize_captures_tuning_env_by_prefix(self):
        with _Sidecar(self) as sc:
            env = {"GP_PEEK": "1", "BL_CRUISE_KMH": "9", "PATH": "/nope"}
            with patch.dict(os.environ, env, clear=True):
                run_meta.note_attempt_start(1, "x.csv")
                run_meta.finalize()
            cfg = sc.read()["config"]
            self.assertEqual(cfg["env"], {"GP_PEEK": "1", "BL_CRUISE_KMH": "9"})
            # Unrelated vars must not bloat the sidecar.
            self.assertNotIn("PATH", cfg["env"])
            self.assertIsInstance(cfg["argv"], list)

    def test_unset_knobs_are_absent_not_defaulted(self):
        # Defaults live in code, which sha+diff_sha already pin. Recording a
        # guessed default would claim a value the run never actually used.
        with _Sidecar(self) as sc:
            with patch.dict(os.environ, {}, clear=True):
                run_meta.note_attempt_start(1, "x.csv")
                run_meta.finalize()
            self.assertEqual(sc.read()["config"]["env"], {})

    def test_git_identifies_a_dirty_tree(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "x.csv")
            run_meta.finalize()
            git = sc.read()["git"]
            # `dirty` alone was useless: every recorded run flew dirty, so the
            # sha identified nothing. These are what make it identifiable.
            for key in ("sha", "branch", "dirty", "diff_sha", "untracked", "submodule"):
                self.assertIn(key, git)

    def test_identity_survives_a_kill_that_never_reaches_finalize(self):
        # Run 20260801_134651 was hard-killed and its sidecar has git, config,
        # entry, pilot and target ALL null -- 18 attempts of telemetry that
        # cannot be attributed to a commit or a config, so unusable as a
        # baseline. Identity is immutable and known at import; capture it on
        # the first write, not at finalize.
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "x.csv")  # no finalize()
            got = sc.read()
            self.assertIsNotNone(got["config"])
            self.assertIsNotNone(got["git"])
            self.assertIsNotNone(got["entry"])
            self.assertIsNone(got["ended_utc"])  # finalize alone stamps this

    def test_identity_is_captured_once_not_per_write(self):
        with _Sidecar(self) as sc:
            with patch.object(run_meta, "_git", return_value={"sha": "x"}) as g:
                run_meta.note_attempt_start(1, "x.csv")
                run_meta.note_attempt_end(gates=1)
                run_meta.note_log_columns(["t"])
            self.assertEqual(g.call_count, 1)  # not once per _write()
            self.assertEqual(sc.read()["git"], {"sha": "x"})

    def test_a_broken_git_does_not_discard_the_rest(self):
        # _git() shells out; @_guard wraps all of finalize, so an exception
        # there used to take config/entry/ended_utc down with it.
        with _Sidecar(self) as sc:
            with patch.object(run_meta, "_git", side_effect=OSError("no git")):
                run_meta.note_attempt_start(1, "x.csv")
                run_meta.finalize()
            got = sc.read()
            self.assertIsNone(got["git"])
            self.assertIsNotNone(got["config"])
            self.assertIsNotNone(got["ended_utc"])

    def test_track_records_the_course_flown(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "x.csv")
            run_meta.note_track(
                [
                    {"gate_id": 0, "position_ned": (1.234, -2.0, 3.0)},
                    {"gate_id": 1, "position_ned": (9.0, 9.0, 9.0)},
                ]
            )
            track = sc.read()["track"]
            self.assertEqual(track["n_gates"], 2)
            self.assertEqual(track["gate0_ned"], [1.23, -2.0, 3.0])

    def test_track_ignores_an_empty_burst(self):
        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "x.csv")
            run_meta.note_track([])
            self.assertIsNone(sc.read()["track"])

    def test_log_columns_let_a_reader_reject_a_schema_unopened(self):
        from simulator.gp_pilot import LOG_COLUMNS

        with _Sidecar(self) as sc:
            run_meta.note_attempt_start(1, "x.csv")
            run_meta.note_log_columns(LOG_COLUMNS)
            self.assertEqual(sc.read()["log_columns"], list(LOG_COLUMNS))


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
