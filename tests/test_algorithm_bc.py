"""Focused tests for the behavior-cloning algorithm adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import torch

from rl.algorithms import BC
from rl.core import spec
from rl.core.config import default_config
from rl.training.train_ppo import NET_ARCH, StandalonePolicy


class BCAdapterTests(unittest.TestCase):
    def _make_policy(self) -> StandalonePolicy:
        return StandalonePolicy(
            arch=NET_ARCH, obs_dim=spec.OBS_DIM, act_dim=spec.ACTION_DIM
        )

    def _checkpoint_path(self, directory: str) -> str:
        path = os.path.join(directory, "policy_bc.pt")
        BC().save(self._make_policy(), path)
        return path

    def test_train_rejects_non_none_env(self):
        with self.assertRaises(ValueError):
            BC().train(object(), default_config())

    def test_train_delegates_to_runner_with_cfg_only(self):
        cfg = default_config()
        sentinel = object()
        with mock.patch("rl.algorithms.bc._run_bc", return_value=sentinel) as runner:
            result = BC().train(None, cfg)
        runner.assert_called_once_with(cfg)
        self.assertIs(result, sentinel)

    def test_save_delegates_to_existing_save_policy(self):
        policy = self._make_policy()
        with mock.patch("rl.algorithms.bc._save_policy") as saver:
            BC().save(policy, "/tmp/x.pt")
        saver.assert_called_once_with(policy, "/tmp/x.pt")

    def test_save_then_load_roundtrip_state_dict_parity(self):
        policy = self._make_policy()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy_bc.pt")
            BC().save(policy, path)
            loaded = BC().load(path)
        self.assertIsInstance(loaded, StandalonePolicy)
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)

    def test_load_sets_eval_mode(self):
        policy = self._make_policy()
        policy.train()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy_bc.pt")
            BC().save(policy, path)
            loaded = BC().load(path)
        self.assertFalse(loaded.training)

    def test_load_rejects_bad_dimensions(self):
        for field in ("obs_dim", "act_dim"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = self._checkpoint_path(directory)
                checkpoint = torch.load(path, weights_only=True)
                checkpoint[field] = 999
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, field):
                    BC().load(path)

    def test_load_rejects_invalid_architecture(self):
        for arch in ([64, True, 64], [], [64, 0, 64]):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as directory:
                path = self._checkpoint_path(directory)
                checkpoint = torch.load(path, weights_only=True)
                checkpoint["arch"] = arch
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "arch"):
                    BC().load(path)

    def test_load_rejects_missing_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._checkpoint_path(directory)
            checkpoint = torch.load(path, weights_only=True)
            del checkpoint["state_dict"]
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "state_dict"):
                BC().load(path)


if __name__ == "__main__":
    unittest.main()
