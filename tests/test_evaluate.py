"""Focused tests for the T11 deterministic evaluation harness."""

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl.core.config import EnvConfig, RLConfig
from rl.environment.env import CURRICULUM
from rl.training.evaluate import CSV_COLUMNS, evaluate

POLICY_PT = "rl/data/policy.pt"
REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR = REPO_ROOT / "rl" / "data" / "policy.pt"
BEST_DIR = REPO_ROOT / "rl" / "data" / "best"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


class PolicyFlowTests(unittest.TestCase):
    def test_policy_smoke(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "eval.csv"
            summary = evaluate(
                policy_path=POLICY_PT,
                episodes=1,
                seeds=[0],
                out_csv=str(out),
            )
        self.assertEqual(summary["episodes"], 1)
        self.assertIn("gates_cleared_mean", summary)
        self.assertIn("return_std", summary)
        for key in (
            "gates_cleared_mean",
            "gates_cleared_std",
            "return_mean",
            "return_std",
        ):
            self.assertTrue(np.isfinite(summary[key]))


class ExpertFlowTests(unittest.TestCase):
    def test_expert_smoke(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "expert.csv"
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                out_csv=str(out),
            )
            self.assertTrue(out.exists())
            header, rows = _read_csv(out)
        self.assertEqual(header, CSV_COLUMNS)
        self.assertEqual(len(rows), 1)
        gates = int(rows[0][2])
        self.assertGreaterEqual(gates, 0)


class DeterminismTests(unittest.TestCase):
    def test_repeat_is_byte_identical(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a.csv"
            out_b = Path(tmp) / "b.csv"
            evaluate(
                expert="gp_expert",
                episodes=2,
                seeds=[0, 1],
                out_csv=str(out_a),
            )
            evaluate(
                expert="gp_expert",
                episodes=2,
                seeds=[0, 1],
                out_csv=str(out_b),
            )
            self.assertEqual(out_a.read_bytes(), out_b.read_bytes())

    def test_different_seed_schedule_diverges(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a.csv"
            out_b = Path(tmp) / "b.csv"
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                out_csv=str(out_a),
            )
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[7],
                out_csv=str(out_b),
            )
            _, rows_a = _read_csv(out_a)
            _, rows_b = _read_csv(out_b)
        self.assertEqual(int(rows_a[0][0]), 0)
        self.assertEqual(int(rows_b[0][0]), 7)
        self.assertNotEqual(rows_a, rows_b)


class CsvSchemaTests(unittest.TestCase):
    def test_header_order_and_row_count_and_schedule(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "schema.csv"
            evaluate(
                expert="gp_expert",
                episodes=3,
                seeds=[0, 1],
                out_csv=str(out),
            )
            header, rows = _read_csv(out)
        self.assertEqual(header, CSV_COLUMNS)
        self.assertEqual(len(rows), 6)
        seed_col = [int(r[0]) for r in rows]
        ep_col = [int(r[1]) for r in rows]
        self.assertEqual(seed_col, [0, 0, 0, 1, 1, 1])
        self.assertEqual(ep_col, [0, 1, 2, 0, 1, 2])

    def test_success_is_zero_or_one(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "success.csv"
            evaluate(
                expert="gp_expert",
                episodes=2,
                seeds=[0],
                out_csv=str(out),
            )
            _, rows = _read_csv(out)
        for row in rows:
            self.assertIn(row[4], {"0", "1"})


class SummaryMathTests(unittest.TestCase):
    def test_summary_matches_population_stats(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "math.csv"
            summary = evaluate(
                expert="gp_expert",
                episodes=2,
                seeds=[0, 1],
                out_csv=str(out),
            )
            _, rows = _read_csv(out)
        gates = np.asarray([int(r[2]) for r in rows], dtype=np.float64)
        rets = np.asarray([float(r[3]) for r in rows], dtype=np.float64)
        succ = np.asarray([int(r[4]) for r in rows], dtype=np.float64)
        np.testing.assert_allclose(summary["gates_cleared_mean"], gates.mean())
        np.testing.assert_allclose(summary["gates_cleared_std"], np.std(gates, ddof=0))
        np.testing.assert_allclose(summary["return_mean"], rets.mean())
        np.testing.assert_allclose(summary["return_std"], np.std(rets, ddof=0))
        np.testing.assert_allclose(summary["success_rate"], succ.mean())
        self.assertEqual(summary["episodes"], len(rows))
        self.assertEqual(summary["seeds"], [0, 1])


class ValidationTests(unittest.TestCase):
    def test_missing_both_modes_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(episodes=1, seeds=[0], out_csv="/tmp/x.csv")

    def test_both_modes_raises(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluate(
                    policy_path=POLICY_PT,
                    expert="gp_expert",
                    out_csv=str(Path(tmp) / "x.csv"),
                )

    def test_nonpositive_episodes_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(expert="gp_expert", episodes=0, seeds=[0], out_csv="/tmp/x.csv")

    def test_empty_seeds_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(expert="gp_expert", episodes=1, seeds=[], out_csv="/tmp/x.csv")

    def test_duplicate_seeds_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0, 0],
                out_csv="/tmp/x.csv",
            )

    def test_non_integer_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0, 1.5],
                out_csv="/tmp/x.csv",
            )

    def test_negative_seed_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[-1],
                out_csv="/tmp/x.csv",
            )

    def test_unsupported_expert_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="bogus",
                episodes=1,
                seeds=[0],
                out_csv="/tmp/x.csv",
            )

    def test_missing_policy_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                policy_path="/nonexistent/policy.pt",
                episodes=1,
                seeds=[0],
                out_csv="/tmp/x.csv",
            )

    def test_bad_env_cfg_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                env_cfg={"curriculum_stage": 2},
                out_csv="/tmp/x.csv",
            )

    def test_envconfig_typed_carrier_accepted(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cfg.csv"
            summary = evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                env_cfg=EnvConfig(curriculum_stage=0, max_steps=200),
                out_csv=str(out),
            )
        self.assertEqual(summary["episodes"], 1)

    def test_envconfig_stage_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                env_cfg=EnvConfig(curriculum_stage=len(CURRICULUM), max_steps=200),
                out_csv="/tmp/x.csv",
            )

    def test_envconfig_nonpositive_max_steps_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                env_cfg=EnvConfig(curriculum_stage=0, max_steps=0),
                out_csv="/tmp/x.csv",
            )

    def test_rlconfig_typed_carrier_accepted(self) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rlcfg.csv"
            summary = evaluate(
                expert="gp_expert",
                episodes=1,
                seeds=[0],
                env_cfg=RLConfig(env=EnvConfig(curriculum_stage=0, max_steps=200)),
                out_csv=str(out),
            )
        self.assertEqual(summary["episodes"], 1)


class AnchorImmutabilityTests(unittest.TestCase):
    def test_policy_pt_byte_identical_after_run(self) -> None:

        before = _sha256(ANCHOR)
        policy_default_dir = BEST_DIR / "policy"
        existed_before = policy_default_dir.exists()
        with tempfile.TemporaryDirectory() as tmp:
            evaluate(
                policy_path=POLICY_PT,
                episodes=1,
                seeds=[0],
                out_csv=str(Path(tmp) / "x.csv"),
            )
        after = _sha256(ANCHOR)
        self.assertEqual(before, after, "policy.pt was mutated by the harness")
        if not existed_before:
            self.assertFalse(
                policy_default_dir.exists(),
                "default rl/data/best/policy/ must not be created when out_csv "
                "points elsewhere",
            )


if __name__ == "__main__":
    unittest.main()
