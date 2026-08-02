"""Focused tests for the thin PPO algorithm adapter."""

from __future__ import annotations

import unittest
from unittest import mock

from rl.algorithms import PPO
from rl.core.config import default_config


class PPOAdapterTests(unittest.TestCase):
    def test_train_rejects_non_none_env(self):
        with self.assertRaises(ValueError):
            PPO().train(object(), default_config())

    def test_train_delegates_to_runner_with_cfg_only(self):
        cfg = default_config()
        sentinel = object()
        with mock.patch("rl.algorithms.ppo._run_ppo", return_value=sentinel) as runner:
            result = PPO().train(None, cfg)
        runner.assert_called_once_with(cfg)
        self.assertIs(result, sentinel)

    def test_load_delegates_to_sb3_native(self):
        with mock.patch(
            "rl.algorithms.ppo._SB3PPO.load", return_value="model"
        ) as loader:
            result = PPO().load("/tmp/whatever.zip")
        loader.assert_called_once_with("/tmp/whatever.zip", force_reset=True)
        self.assertEqual(result, "model")

    def test_save_delegates_to_atomic_saver(self):
        policy = object()
        with mock.patch("rl.algorithms.ppo.save_sb3_atomic") as saver:
            PPO().save(policy, "/tmp/out.zip")
        saver.assert_called_once_with(policy, "/tmp/out.zip")


if __name__ == "__main__":
    unittest.main()
