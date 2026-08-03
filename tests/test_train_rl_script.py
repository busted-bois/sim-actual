"""Tests for the cross-platform end-to-end RL training script."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_rl.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _paths(text: str) -> str:
    r"""Normalize path separators before matching.

    The script builds paths with os.path.join, so on Windows it prints
    rl\data\best\ppo\policy.pt while these assertions are written with POSIX
    separators. That is a difference in separator style, not in behaviour, so
    normalize rather than asserting one platform's spelling.
    """
    return text.replace("\\", "/")


class TrainRlScriptTests(unittest.TestCase):
    def test_default_dry_run_has_ordered_complete_pipeline(self):
        result = _run("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        stages = (
            "Generate GP demos",
            "Train behavior cloning",
            "Train curriculum PPO",
            "Verify candidate",
            "Evaluate candidate",
            "Evaluate GP expert",
        )
        positions = [result.stdout.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("rl/data/best/ppo/policy.pt", _paths(result.stdout))
        self.assertIn("--bc-init rl/data/policy_bc.pt", _paths(result.stdout))
        self.assertIn("course_gates=17", result.stdout)
        self.assertNotIn("[demos]", result.stdout)

    def test_overrides_reach_commands(self):
        result = _run(
            "--dry-run",
            "--run-name",
            "windows-run",
            "--steps",
            "2000",
            "--envs",
            "2",
            "--episodes",
            "3",
            "--seeds",
            "0,4,9",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "--run-name windows-run",
            "--steps 2000",
            "--envs 2",
            "--episodes 3",
            "--seeds 0 4 9",
            "rl/data/best/windows-run/expert-eval.csv",
        ):
            self.assertIn(expected, _paths(result.stdout))

    def test_custom_config_resolves_checkpoint_directory(self):
        result = _run("--dry-run", "--config", "configs/default.yaml")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("checkpoint.dir=rl/data/best", _paths(result.stdout))

    def test_help(self):
        result = _run("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--run-name", result.stdout)
        self.assertIn("--dry-run", result.stdout)

    def test_invalid_arguments_fail_before_training(self):
        cases = (
            ("--run-name", "../escape"),
            ("--run-name", "two words"),
            ("--steps", "0"),
            ("--envs", "nope"),
            ("--episodes", "-1"),
            ("--seeds", "0,-1"),
            ("--seeds", "1,1"),
            ("--unknown",),
        )
        for case in cases:
            with self.subTest(case=case):
                result = _run(*case)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Generate GP demos", result.stdout)


if __name__ == "__main__":
    unittest.main()
