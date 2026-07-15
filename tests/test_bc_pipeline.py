"""Demo logging -> BC pretrain -> PPO warm-start pipeline (offline)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from rl import spec
from rl.env import GateRacingEnv
from rl.gp_expert import GPExpert
from rl.log_demos import _episode, save
from rl.train_bc import save_policy, train_bc
from rl.train_ppo import NET_ARCH, StandalonePolicy


class DemoLoggingTests(unittest.TestCase):
    def test_episode_shapes_and_bounds(self):
        env = GateRacingEnv(stage=0, seed=200)
        obs, act = _episode(env, GPExpert())
        self.assertGreater(len(obs), 0)
        self.assertEqual(obs.shape[1], spec.OBS_DIM)
        self.assertEqual(act.shape[1], spec.ACTION_DIM)
        self.assertEqual(len(obs), len(act))
        self.assertTrue(np.all(np.abs(act) <= 1.0))
        self.assertEqual(obs.dtype, np.float32)

    def test_save_roundtrip(self):
        demos = {
            "obs": np.zeros((7, spec.OBS_DIM), np.float32),
            "act": np.zeros((7, spec.ACTION_DIM), np.float32),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gp_demos.npz")
            save(demos, path)
            with np.load(path) as loaded:
                self.assertEqual(loaded["obs"].shape, (7, spec.OBS_DIM))
                self.assertEqual(loaded["act"].shape, (7, spec.ACTION_DIM))


class BCTrainTests(unittest.TestCase):
    def test_loss_drops_and_schema(self):
        rng = np.random.default_rng(5)
        obs = torch.from_numpy(
            rng.uniform(-1, 1, (128, spec.OBS_DIM)).astype(np.float32)
        )
        w = torch.from_numpy(
            rng.uniform(-0.2, 0.2, (spec.OBS_DIM, spec.ACTION_DIM)).astype(np.float32)
        )
        act = torch.clamp(obs @ w, -1, 1)
        policy, losses = train_bc(obs, act, epochs=10, batch_size=32)
        self.assertLess(losses[-1], losses[0])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy_bc.pt")
            save_policy(policy, path)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["arch"], NET_ARCH)
        self.assertEqual(ckpt["obs_dim"], spec.OBS_DIM)
        self.assertEqual(ckpt["act_dim"], spec.ACTION_DIM)
        std = StandalonePolicy()
        std.load_state_dict(ckpt["state_dict"])  # same schema as policy.pt


class PPOWarmStartTests(unittest.TestCase):
    def test_bc_weights_land_in_ppo_actor(self):
        import torch.nn as nn
        from stable_baselines3 import PPO

        from rl.train_ppo import _vec_env, load_bc_init

        policy = StandalonePolicy()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy_bc.pt")
            save_policy(policy, path)
            model = PPO(
                "MlpPolicy",
                _vec_env(0, 1, 0),
                policy_kwargs=dict(
                    net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh
                ),
                n_steps=32,
                batch_size=16,
                device="cpu",
                seed=0,
            )
            load_bc_init(model, path)

        obs = (
            np.random.default_rng(1)
            .uniform(-1, 1, (16, spec.OBS_DIM))
            .astype(np.float32)
        )
        sb3_act, _ = model.predict(obs, deterministic=True)
        policy.eval()
        with torch.no_grad():
            bc_act = policy(torch.from_numpy(obs)).numpy()
        err = np.abs(np.clip(sb3_act, -1, 1) - np.clip(bc_act, -1, 1)).max()
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    unittest.main()
