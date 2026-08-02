"""Regression coverage for the frozen RL environment dynamics and API."""

import json
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from rl.core import spec
from rl.environment.env import CURRICULUM, G_WORLD, GateRacingEnv


BASELINE_PATH = Path(__file__).parents[1] / "rl" / "data" / "baseline.json"


class EnvironmentRegressionTests(unittest.TestCase):
    def test_wave_zero_golden_trajectory(self) -> None:
        golden = json.loads(BASELINE_PATH.read_text())["golden_trajectory"]
        env = GateRacingEnv(stage=golden["stage"])
        observation, _ = env.reset(seed=golden["seed"])
        action = np.asarray(golden["action"], dtype=np.float32)

        np.testing.assert_array_equal(action, golden["action"])
        for expected in golden["steps"]:
            observation, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_allclose(env.p, expected["position"], atol=1e-12)
            np.testing.assert_allclose(env.v, expected["velocity"], atol=1e-12)
            np.testing.assert_allclose(env.q, expected["quaternion"], atol=1e-12)
            np.testing.assert_allclose(
                env.omega, expected["angular_velocity"], atol=1e-12
            )
            self.assertAlmostEqual(reward, expected["reward"], places=12)
            self.assertEqual(terminated, expected["terminated"])
            self.assertEqual(truncated, expected["truncated"])
            self.assertEqual(info["gates_cleared"], expected["gate_index"])
            self.assertEqual(
                {key: value for key, value in info.items() if key != "gates_cleared"},
                expected["info"],
            )

        np.testing.assert_allclose(observation, golden["final_observation"], atol=1e-7)

    def test_curriculum_contract(self) -> None:
        self.assertEqual(
            CURRICULUM,
            [
                {
                    "num_gates": 1,
                    "spawn_dist": 5.0,
                    "jitter": 0.5,
                    "max_seconds": 20.0,
                },
                {
                    "num_gates": 2,
                    "spawn_dist": 6.0,
                    "jitter": 1.0,
                    "max_seconds": 20.0,
                },
                {
                    "num_gates": 6,
                    "spawn_dist": 7.0,
                    "jitter": 2.0,
                    "max_seconds": 30.0,
                },
                {
                    "num_gates": 17,
                    "spawn_dist": 7.0,
                    "jitter": 2.0,
                    "max_seconds": 70.0,
                },
            ],
        )
        for stage, gate_count in enumerate((1, 2, 6, 17)):
            env = GateRacingEnv(stage=stage, seed=0)
            self.assertEqual(env.stage, stage)
            self.assertEqual(len(env.gate_map), gate_count)
            self.assertEqual(env.max_steps, int(CURRICULUM[stage]["max_seconds"] * 50))
        self.assertEqual(GateRacingEnv(stage=-1).stage, 0)
        self.assertEqual(GateRacingEnv(stage=99).stage, 3)

    def test_full_stage_uses_all_17_supplied_gates(self) -> None:
        gate_map = [
            {"pos": [float(index + 1), 0.0, -3.0], "quat": [1.0, 0.0, 0.0, 0.0]}
            for index in range(17)
        ]
        env = GateRacingEnv(stage=3, gate_map=gate_map, seed=0)
        self.assertEqual(len(env.gate_map), 17)
        self.assertEqual(env.gate_map[-1]["pos"], [17.0, 0.0, -3.0])

    def test_physics_gravity_contract(self) -> None:
        np.testing.assert_array_equal(G_WORLD, np.array([0.0, 0.0, spec.GRAVITY]))

    def test_gymnasium_contract(self) -> None:
        check_env(GateRacingEnv(stage=0, seed=0), warn=True)


if __name__ == "__main__":
    unittest.main()
