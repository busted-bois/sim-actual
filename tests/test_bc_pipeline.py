"""Demo logging -> BC pretrain -> PPO warm-start pipeline (offline)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from rl.core import spec
from rl.environment.env import GateRacingEnv
from rl.experts.gp_expert import GPExpert
from rl.training.log_demos import _episode, save
from rl.training.train_bc import save_policy, train_bc
from rl.training.train_ppo import NET_ARCH, StandalonePolicy


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
                # Plant stamp checked by rl.train_bc.load_demos.
                np.testing.assert_allclose(
                    loaded["action_scale"],
                    [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE],
                )
                np.testing.assert_allclose(loaded["hover_thrust"], spec.HOVER_THRUST)

    def test_load_demos_roundtrips_current_stamp(self):
        from rl.training.train_bc import load_demos

        demos = {
            "obs": np.zeros((5, spec.OBS_DIM), np.float32),
            "act": np.zeros((5, spec.ACTION_DIM), np.float32),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gp_demos.npz")
            save(demos, path)
            obs, act = load_demos(path)
        self.assertEqual(obs.shape, (5, spec.OBS_DIM))
        self.assertEqual(act.shape, (5, spec.ACTION_DIM))

    def test_load_demos_refuses_stale_or_unstamped(self):
        from rl.training.train_bc import load_demos

        with tempfile.TemporaryDirectory() as d:
            # Unstamped (pre-caps-change) demo file.
            path = os.path.join(d, "old.npz")
            np.savez_compressed(
                path,
                obs=np.zeros((3, spec.OBS_DIM), np.float32),
                act=np.zeros((3, spec.ACTION_DIM), np.float32),
            )
            with self.assertRaises(SystemExit) as ctx:
                load_demos(path)
            self.assertIn("log-demos", str(ctx.exception))

            # Stamped under different caps.
            path2 = os.path.join(d, "stale.npz")
            np.savez_compressed(
                path2,
                obs=np.zeros((3, spec.OBS_DIM), np.float32),
                act=np.zeros((3, spec.ACTION_DIM), np.float32),
                action_scale=np.array([4.0, 4.0, 3.0]),
                hover_thrust=np.array(0.5),
            )
            with self.assertRaises(SystemExit):
                load_demos(path2)

    def test_load_demos_missing_file_points_at_make_target(self):
        from rl.training.train_bc import load_demos

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as ctx:
                load_demos(os.path.join(d, "nope.npz"))
        self.assertIn("log-demos", str(ctx.exception))


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
        # Training-plant contract consumed by rl.deploy.live_scale_action.
        self.assertEqual(ckpt["train_hover_thrust"], spec.HOVER_THRUST)
        self.assertEqual(
            list(ckpt["action_scale"]),
            [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE],
        )
        std = StandalonePolicy()
        std.load_state_dict(ckpt["state_dict"])  # same schema as policy.pt


class PPOWarmStartTests(unittest.TestCase):
    def test_bc_weights_land_in_ppo_actor(self):
        import torch.nn as nn
        from stable_baselines3 import PPO

        from rl.training.train_ppo import _vec_env, load_bc_init

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

        # PPO export carries the same training-plant metadata as BC.
        from rl.training.train_ppo import export_policy

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "policy.pt")
            export_policy(model, out)
            ckpt = torch.load(out, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["train_hover_thrust"], spec.HOVER_THRUST)
        self.assertEqual(
            list(ckpt["action_scale"]),
            [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE],
        )


if __name__ == "__main__":
    unittest.main()
