"""Unit tests for the cross-run control scorer (scripts/gp_score.py).

The scorer is the thing that decides whether a pilot change helped, so its two
load-bearing behaviours are locked in here: it must refuse a log whose schema
is not the current one, and it must never pool runs that flew different code
or different config. Both were how "did this help?" got answered wrongly
before -- 6a10771 shipped on sound reasoning and needed a flight to disprove.

No live sim and no flight needed: the CSVs are synthesised against the pilot's
own LOG_COLUMNS, so this also proves the per-tick pipeline works before the
first real flight ever writes a current-schema log.
"""

import csv
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

from simulator.gp_pilot import LOG_COLUMNS

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(os.path.dirname(_HERE), "scripts", "gp_score.py")

_spec = importlib.util.spec_from_file_location("gp_score", _SCRIPT)
gp_score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp_score)


def _row(**over):
    """One tick of straight-and-level flight; override what a test cares about."""
    base = dict.fromkeys(LOG_COLUMNS, "0")
    base.update(
        {
            "t": "1000.000",
            "up": "0.980",
            "thrust": "0.2700",
            "mode": "yolo",
            "source": "yolo",
            "reliable": "1",
            "vision_valid": "1",
            "agl": "1.500",
            "infer_ms": "150.0",
            "bl_gap": "0",
            "occ_gap": "-1",
        }
    )
    base.update({k: str(v) for k, v in over.items()})
    return base


class _Workspace:
    """A temp cwd holding runs/videos/*.json + rl/data/*.csv."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="gpscore_")

    def __enter__(self):
        os.makedirs(os.path.join(self.root, "runs", "videos"))
        os.makedirs(os.path.join(self.root, "rl", "data"))
        self._prev = os.getcwd()
        os.chdir(self.root)
        return self

    def __exit__(self, *exc):
        os.chdir(self._prev)
        shutil.rmtree(self.root, ignore_errors=True)
        return False

    def add(self, run_id, rows, header=None, git=None, env=None, gates=2, schema=2):
        name = f"gp_log_{run_id}_a1.csv"
        with open(os.path.join("rl", "data", name), "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(header or list(LOG_COLUMNS))
            for r in rows:
                wr.writerow([r[c] for c in (header or LOG_COLUMNS)])
        meta = {
            "schema": schema,
            "run_id": run_id,
            "git": git if git is not None else {
                "sha": "abc1234", "branch": "b", "dirty": False,
                "diff_sha": None, "untracked": None, "submodule": "s",
            },
            "config": {"env": env if env is not None else {}, "argv": []},
            "track": {"n_gates": 6, "gate0_ned": [0, 0, 0]},
            "log_columns": list(header or LOG_COLUMNS),
            "attempts": [
                {"n": 1, "telemetry": name, "gates_passed": gates,
                 "outcome": "gate_stall", "lap_s": None, "end_reason": "reset"}
            ],
        }
        with open(os.path.join("runs", "videos", f"vision_{run_id}.json"), "w") as fh:
            json.dump(meta, fh)


class SchemaGateTests(unittest.TestCase):
    def test_current_schema_is_read(self):
        with _Workspace() as ws:
            ws.add("20260101_000000", [_row(), _row(t="1000.031")])
            groups, skipped = gp_score.collect(
                ["runs/videos/vision_20260101_000000.json"]
            )
            self.assertEqual(skipped, 0)
            self.assertEqual(sum(len(g["rows"]) for g in groups.values()), 1)

    def test_legacy_schema_is_skipped_not_misread(self):
        # The real hazard: a 33-column log silently read as if it were the
        # 37-column one would misattribute every column after the drift point.
        with _Workspace() as ws:
            legacy = list(LOG_COLUMNS)[:-3]
            ws.add("20260101_000001", [_row()], header=legacy)
            groups, skipped = gp_score.collect(
                ["runs/videos/vision_20260101_000001.json"]
            )
            self.assertEqual(skipped, 1)
            self.assertEqual(sum(len(g["rows"]) for g in groups.values()), 0)

    def test_a_reorder_is_caught_which_a_version_int_would_miss(self):
        with _Workspace() as ws:
            swapped = list(LOG_COLUMNS)
            i = swapped.index("roll")
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            ws.add("20260101_000002", [_row()], header=swapped)
            _groups, skipped = gp_score.collect(
                ["runs/videos/vision_20260101_000002.json"]
            )
            self.assertEqual(skipped, 1)


class GroupingTests(unittest.TestCase):
    def test_different_config_is_not_pooled(self):
        with _Workspace() as ws:
            ws.add("20260101_000010", [_row()], env={"GP_PEEK": "1"})
            ws.add("20260101_000011", [_row()], env={"GP_PEEK": "0"})
            groups, _ = gp_score.collect(sorted(
                __import__("glob").glob("runs/videos/vision_*.json")
            ))
            self.assertEqual(len(groups), 2)

    def test_force_pools_everything(self):
        with _Workspace() as ws:
            ws.add("20260101_000012", [_row()], env={"GP_PEEK": "1"})
            ws.add("20260101_000013", [_row()], env={"GP_PEEK": "0"})
            groups, _ = gp_score.collect(
                sorted(__import__("glob").glob("runs/videos/vision_*.json")),
                force=True,
            )
            self.assertEqual(len(groups), 1)

    def test_legacy_sidecar_is_not_called_clean(self):
        # schema-1 sidecars have no diff_sha key; every one of those runs flew
        # dirty, so reporting "clean" would assert something never observed.
        key = gp_score.config_key({"git": {"sha": "c19a5ae", "dirty": True}})
        self.assertEqual(key[1], "?")
        self.assertIn("diff=?", gp_score.describe_key(key))

    def test_clean_tree_is_distinguished_from_unknown(self):
        key = gp_score.config_key({"git": {"sha": "x", "diff_sha": None}})
        self.assertEqual(key[1], "clean")


class TickMetricTests(unittest.TestCase):
    def test_saturation_is_measured_against_the_pilots_own_clamps(self):
        from simulator.gp_pilot import ROLL_WIRE_MAX_DEG

        rows = [_row(cmd_roll_deg=ROLL_WIRE_MAX_DEG) for _ in range(3)]
        rows += [_row(cmd_roll_deg=1.0) for _ in range(1)]
        m = gp_score.tick_metrics([rows])
        self.assertAlmostEqual(m["roll_sat"], 0.75, places=6)

    def test_upset_ticks_counted_from_unclipped_up(self):
        rows = [_row(up=0.98), _row(up=-0.31), _row(up=0.2)]
        m = gp_score.tick_metrics([rows])
        self.assertEqual(m["upset_ticks"], 2)
        self.assertAlmostEqual(m["up_min"], -0.31, places=6)

    def test_guard_firing_is_tracked_separately_from_being_upset(self):
        # They diverge when the guard is disabled or the threshold is retuned,
        # and that divergence is the point of having both.
        rows = [_row(up=-0.31, upset=0), _row(up=-0.31, upset=1)]
        m = gp_score.tick_metrics([rows])
        self.assertEqual(m["upset_ticks"], 2)
        self.assertEqual(m["guard_fired"], 1)

    def test_loop_rate_ignores_the_gap_between_attempts(self):
        # Two attempts minutes apart must not contribute a 100 s "dt".
        a1 = [_row(t=1000.0), _row(t=1000.031), _row(t=1000.062)]
        a2 = [_row(t=1200.0), _row(t=1200.031), _row(t=1200.062)]
        m = gp_score.tick_metrics([a1, a2])
        self.assertAlmostEqual(m["hz_median"], 1.0 / 0.031, places=3)

    def test_longest_blind_run_spans_only_consecutive_ticks(self):
        rows = [
            _row(t=1000.0, vision_valid=1),
            _row(t=1000.5, vision_valid=0),
            _row(t=1001.0, vision_valid=0),
            _row(t=1001.5, vision_valid=1),
            _row(t=1002.0, vision_valid=0),
            _row(t=1002.2, vision_valid=1),
        ]
        m = gp_score.tick_metrics([rows])
        self.assertAlmostEqual(m["blind_longest_s"], 1.0, places=6)

    def test_blank_columns_do_not_crash_the_scorer(self):
        # infer_ms is blank whenever YOLO published nothing for that tick.
        m = gp_score.tick_metrics([[_row(infer_ms=""), _row(infer_ms="")]])
        self.assertEqual(m["n_ticks"], 2)

    def test_mode_dwell_sums_to_one(self):
        rows = [_row(mode="SEARCH"), _row(mode="SEARCH"), _row(mode="yolo")]
        m = gp_score.tick_metrics([rows])
        self.assertAlmostEqual(sum(m["dwell"].values()), 1.0, places=9)
        self.assertAlmostEqual(m["dwell"]["SEARCH"], 2 / 3, places=6)


class OutcomeTests(unittest.TestCase):
    def test_counts_are_raw_never_percentages(self):
        atts = [
            {"gates_passed": 0, "outcome": "gate_stall"},
            {"gates_passed": 2, "outcome": "gate_stall"},
            {"gates_passed": 0, "outcome": "gate1_fail"},
        ]
        m = gp_score.outcome_metrics(atts)
        self.assertEqual((m["reached_1"], m["n_scored"]), (1, 3))
        self.assertEqual(m["best_gates"], 2)


if __name__ == "__main__":
    unittest.main()
