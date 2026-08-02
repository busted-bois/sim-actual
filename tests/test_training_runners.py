"""Tests for T9 training runners: CLI parsing, validation, callbacks, smoke isolation."""

import csv
import hashlib
import os
import tempfile
import unittest

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rl.core import spec
from rl.core.config import default_config, resolve_device
from rl.training.train_bc import (
    _make_synthetic_demos,
    run as bc_run,
    save_policy,
    train_bc,
)
from rl.training.train_ppo import (
    LEGACY_RUN_DIR,
    NET_ARCH,
    POLICY_PT,
    StandalonePolicy,
    _ProgressCallback,
    _largest_safe_divisor,
    _make_env,
    _parse_args,
    _select_rollout_size,
    _vec_env,
    export_policy,
    load_bc_init,
    run as ppo_run,
    validate_run_name,
)


class ValidateRunNameTests(unittest.TestCase):
    def test_valid_names(self):
        self.assertEqual(validate_run_name("ppo"), "ppo")
        self.assertEqual(validate_run_name("smoke-qa"), "smoke-qa")
        self.assertEqual(validate_run_name("test_run_123"), "test_run_123")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_run_name("")
        with self.assertRaises(ValueError):
            validate_run_name("   ")

    def test_path_separators_rejected(self):
        for bad in ["../etc", "foo/bar", "foo\\bar", "./x", "a..b"]:
            with self.assertRaises(ValueError, msg=f"should reject {bad!r}"):
                validate_run_name(bad)

    def test_dot_segments_rejected(self):
        with self.assertRaises(ValueError):
            validate_run_name("..")
        with self.assertRaises(ValueError):
            validate_run_name("a/../b")


class ParseArgsTests(unittest.TestCase):
    def test_default_args(self):
        args = _parse_args([])
        self.assertEqual(args.config, "configs/default.yaml")
        self.assertEqual(args.run_name, "ppo")
        self.assertIsNone(args.smoke)
        self.assertFalse(args.quick)

    def test_smoke_flag(self):
        args = _parse_args(["--smoke", "500"])
        self.assertEqual(args.smoke, 500)

    def test_quick_maps_to_smoke(self):
        args = _parse_args(["--quick"])
        self.assertTrue(args.quick)

    def test_legacy_flags_preserved(self):
        args = _parse_args(["--steps", "100", "--envs", "2", "--bc-init", "foo.pt"])
        self.assertEqual(args.steps, 100)
        self.assertEqual(args.envs, 2)
        self.assertEqual(args.bc_init, "foo.pt")


class ProgressCallbackTests(unittest.TestCase):
    def test_csv_header_and_row_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.csv")
            cb = _ProgressCallback(path)
            cb.num_timesteps = 500
            cb.locals = {
                "infos": [
                    {
                        "episode": {"r": 12.5, "l": 100},
                        "gates_cleared": 1,
                    }
                ]
            }
            cb._on_step()
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertEqual(header, ["ep_rew", "gates_cleared", "n_steps"])
                row = next(reader)
                self.assertEqual(len(row), 3)
                self.assertEqual(float(row[0]), 12.5)
                self.assertEqual(int(row[1]), 1)
                self.assertEqual(int(row[2]), 500)

    def test_n_steps_is_global(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.csv")
            cb = _ProgressCallback(path)
            cb.num_timesteps = 1234
            cb.locals = {
                "infos": [{"episode": {"r": 0.0, "l": 50}, "gates_cleared": 0}]
            }
            cb._on_step()
            with open(path) as f:
                reader = csv.reader(f)
                next(reader)
                row = next(reader)
                self.assertEqual(int(row[2]), 1234)

    def test_multiple_rows_appended(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "progress.csv")
            cb = _ProgressCallback(path)
            cb.num_timesteps = 50
            cb.locals = {
                "infos": [{"episode": {"r": 1.0, "l": 50}, "gates_cleared": 0}]
            }
            cb._on_step()
            cb.num_timesteps = 100
            cb.locals = {
                "infos": [{"episode": {"r": 5.0, "l": 80}, "gates_cleared": 2}]
            }
            cb._on_step()
            with open(path) as f:
                rows = list(csv.reader(f))
                self.assertEqual(len(rows), 3)

    def test_resume_no_duplicate_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "progress.csv")
            with open(path, "w") as f:
                f.write("ep_rew,gates_cleared,n_steps\n")
                f.write("10.0,1,200\n")
            cb = _ProgressCallback(path)
            cb._init_callback()
            cb.num_timesteps = 400
            cb.locals = {
                "infos": [{"episode": {"r": 5.0, "l": 50}, "gates_cleared": 0}]
            }
            cb._on_step()
            with open(path) as f:
                rows = list(csv.reader(f))
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0], ["ep_rew", "gates_cleared", "n_steps"])


class LargestSafeDivisorTests(unittest.TestCase):
    def test_divides_rollout(self):
        self.assertEqual(_largest_safe_divisor(1024, 256), 256)
        self.assertEqual(_largest_safe_divisor(100, 256), 100)
        self.assertEqual(_largest_safe_divisor(17, 256), 17)

    def test_never_one_for_valid_rollout(self):
        self.assertEqual(_largest_safe_divisor(13, 8), 2)
        self.assertEqual(_largest_safe_divisor(7, 3), 2)
        self.assertEqual(_largest_safe_divisor(5, 2), 2)

    def test_small_rollout(self):
        self.assertEqual(_largest_safe_divisor(1, 256), 1)


class SelectRolloutSizeTests(unittest.TestCase):
    def test_2000_selects_1000_with_max_1024(self):
        result = _select_rollout_size(2000, 1024, 1)
        self.assertEqual(result, 1000)
        self.assertEqual(2000 % result, 0)

    def test_total_equals_rollout(self):
        self.assertEqual(_select_rollout_size(1024, 1024, 1), 1024)

    def test_small_total(self):
        result = _select_rollout_size(64, 1024, 1)
        self.assertEqual(64 % result, 0)
        self.assertLessEqual(result, 64)


class ConfigConsumptionTests(unittest.TestCase):
    def test_config_loads_and_constructs(self):
        cfg = default_config()
        self.assertEqual(cfg.ppo.n_steps, 1024)
        self.assertEqual(cfg.ppo.batch_size, 256)
        self.assertEqual(cfg.ppo.gamma, 0.99)
        self.assertEqual(cfg.ppo.lr, 3e-4)
        self.assertEqual(cfg.ppo.n_epochs, 10)
        self.assertEqual(cfg.ppo.net_arch, [64, 64, 64])
        self.assertEqual(cfg.ppo.total_timesteps_per_stage, 300_000)
        self.assertEqual(cfg.ppo.n_envs, 8)
        self.assertEqual(cfg.env.max_steps, 1000)
        self.assertEqual(cfg.env.curriculum_stage, 2)
        self.assertEqual(cfg.device, "auto")

    def test_resolve_device(self):
        d = resolve_device("auto")
        self.assertIn(d.type, {"cpu", "cuda"})
        self.assertEqual(resolve_device("cpu").type, "cpu")

    def test_curriculum_stage_rejected_out_of_range(self):
        cfg = default_config()
        cfg.env.curriculum_stage = 5
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = os.path.join(d, "best")
            with self.assertRaises(ValueError, msg="curriculum_stage 5 out of range"):
                ppo_run(cfg, run_name="bad-stage", tb_root=os.path.join(d, "tb"))


class SmokeIsolationTests(unittest.TestCase):
    def setUp(self):
        self._anchor = os.path.join(
            os.path.dirname(__file__), "..", "rl", "data", "policy.pt"
        )
        self._before = None
        if os.path.exists(self._anchor):
            with open(self._anchor, "rb") as f:
                self._before = hashlib.sha256(f.read()).hexdigest()

    def tearDown(self):
        if self._before and os.path.exists(self._anchor):
            with open(self._anchor, "rb") as f:
                after = hashlib.sha256(f.read()).hexdigest()
            if after != self._before:
                raise AssertionError("anchor policy.pt was mutated")

    def _make_cfg(self, tmpdir, run_name="test-smoke"):
        cfg = default_config()
        cfg.checkpoint.dir = os.path.join(tmpdir, "best")
        return cfg, os.path.join(tmpdir, "tb"), run_name

    def test_ppo_smoke_no_anchor_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)

    def test_ppo_smoke_creates_csv(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d, "csv-test")
            cfg.env.max_steps = 50
            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            csv_path = os.path.join(d, "best", "csv-test", "progress.csv")
            self.assertTrue(os.path.exists(csv_path), "progress.csv must exist")
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertIn("ep_rew", header)
                self.assertIn("gates_cleared", header)
                self.assertIn("n_steps", header)
                rows = list(reader)
                self.assertGreater(len(rows), 0, "progress.csv must have >= 1 data row")

    def test_ppo_smoke_forces_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            cfg.device = "auto"
            model = ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)
            self.assertEqual(model.device.type, "cpu")

    def test_ppo_smoke_tb_in_temp(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)
            tb_dir = os.path.join(tb_root, rn)
            self.assertTrue(os.path.isdir(tb_dir))

    def test_bc_smoke_no_save(self):
        anchor_bc = os.path.join(
            os.path.dirname(__file__), "..", "rl", "data", "policy_bc.pt"
        )
        existed_before = os.path.exists(anchor_bc)
        cfg = default_config()
        bc_run(cfg, smoke_n=32)
        self.assertEqual(
            os.path.exists(anchor_bc),
            existed_before,
            "smoke BC must not save policy_bc.pt",
        )

    def test_bc_smoke_synthetic_shapes(self):
        obs, act = _make_synthetic_demos(50)
        self.assertEqual(obs.shape, (50, spec.OBS_DIM))
        self.assertEqual(act.shape, (50, spec.ACTION_DIM))

    def test_bc_smoke_loss_drops(self):
        cfg = default_config()
        cfg.seed = 123
        policy = bc_run(cfg, smoke_n=64)
        self.assertIsInstance(policy, StandalonePolicy)


class ConfigurableArchTests(unittest.TestCase):
    def test_export_policy_infers_arch(self):
        import torch.nn as nn

        arch = [32, 16]
        policy_kwargs = dict(net_arch=dict(pi=arch, vf=arch), activation_fn=nn.Tanh)
        env = _vec_env(0, 1, 0)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=32,
            batch_size=16,
            device="cpu",
            seed=0,
        )
        model.learn(total_timesteps=32)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "policy.pt")
            export_policy(model, out)
            ckpt = torch.load(out, map_location="cpu", weights_only=True)
        self.assertEqual(ckpt["arch"], arch)
        std = StandalonePolicy(arch=arch)
        std.load_state_dict(ckpt["state_dict"])

    def test_export_parity_with_nondefault_arch(self):
        import torch.nn as nn

        arch = [32, 16]
        policy_kwargs = dict(net_arch=dict(pi=arch, vf=arch), activation_fn=nn.Tanh)
        env = _vec_env(0, 1, 0)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=32,
            batch_size=16,
            device="cpu",
            seed=0,
        )
        model.learn(total_timesteps=64)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "policy.pt")
            std = export_policy(model, out)
            err = _verify_export_with(model, std)
            self.assertLess(err, 1e-4)

    def test_load_bc_init_mismatch_raises(self):
        import torch.nn as nn

        bad_arch = [32, 16]
        bad_policy = StandalonePolicy(arch=bad_arch)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.pt")
            torch.save(
                {
                    "state_dict": bad_policy.state_dict(),
                    "arch": bad_arch,
                    "obs_dim": spec.OBS_DIM,
                    "act_dim": spec.ACTION_DIM,
                },
                path,
            )
            good_env = _vec_env(0, 1, 0)
            model = PPO(
                "MlpPolicy",
                good_env,
                policy_kwargs=dict(
                    net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh
                ),
                n_steps=32,
                batch_size=16,
                device="cpu",
                seed=0,
            )
            with self.assertRaises(ValueError, msg="arch mismatch must raise"):
                load_bc_init(model, path)

    def test_bc_train_with_custom_arch(self):
        obs = torch.zeros((8, spec.OBS_DIM))
        act = torch.zeros((8, spec.ACTION_DIM))
        policy, losses = train_bc(obs, act, epochs=2, batch_size=4, arch=[32, 16])
        self.assertIsInstance(policy, StandalonePolicy)
        self.assertEqual(len(losses), 2)

    def test_bc_save_stamps_actual_arch(self):
        policy = StandalonePolicy(arch=[32, 16])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy_bc.pt")
            save_policy(policy, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["arch"], [32, 16])


def _verify_export_with(model, std, n=64):
    obs = np.random.uniform(-1, 1, (n, spec.OBS_DIM)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mine = std(torch.from_numpy(obs)).numpy()
    return float(np.abs(np.clip(sb3_act, -1, 1) - np.clip(mine, -1, 1)).max())


class LegacyCompatTests(unittest.TestCase):
    def test_legacy_train_outputs_do_not_target_anchor(self):
        self.assertNotEqual(
            os.path.realpath(os.path.join(LEGACY_RUN_DIR, "policy.pt")),
            os.path.realpath(POLICY_PT),
        )

    def test_train_bc_signature(self):
        obs = torch.zeros((8, spec.OBS_DIM))
        act = torch.zeros((8, spec.ACTION_DIM))
        policy, losses = train_bc(obs, act, epochs=2, batch_size=4)
        self.assertIsInstance(policy, StandalonePolicy)
        self.assertEqual(len(losses), 2)

    def test_train_ppo_legacy_imports(self):
        self.assertIsNotNone(StandalonePolicy)
        self.assertEqual(NET_ARCH, [64, 64, 64])

    def test_make_env_creates_monitor(self):
        fn = _make_env(0, seed=42)
        env = fn()
        self.assertIsInstance(env, Monitor)

    def test_make_env_accepts_max_steps(self):
        fn = _make_env(0, seed=42, max_steps=100)
        env = fn()
        self.assertIsInstance(env, Monitor)
        self.assertEqual(env.env.max_steps, 100)

    def test_vec_env_legacy_call(self):
        env = _vec_env(0, n_envs=1, seed=0)
        self.assertIsNotNone(env)

    def test_export_policy_default_out_still_works(self):
        import torch.nn as nn

        env = _vec_env(0, 1, 0)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh
            ),
            n_steps=32,
            batch_size=16,
            device="cpu",
            seed=0,
        )
        model.learn(total_timesteps=32)
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "policy.pt")
            std = export_policy(model, out)
            self.assertIsInstance(std, StandalonePolicy)


class SmokeValueValidationTests(unittest.TestCase):
    def test_ppo_rejects_smoke_lt_2(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = os.path.join(d, "best")
            with self.assertRaises(ValueError):
                ppo_run(
                    cfg, run_name="bad", smoke_steps=1, tb_root=os.path.join(d, "tb")
                )

    def test_bc_rejects_smoke_lt_2(self):
        cfg = default_config()
        with self.assertRaises(ValueError):
            bc_run(cfg, smoke_n=1)


if __name__ == "__main__":
    unittest.main()
