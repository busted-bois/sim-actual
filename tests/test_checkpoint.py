"""Tests for T10 atomic versioned PPO checkpoints and curriculum resume."""

import csv
import hashlib
import json
import os
import tempfile
import unittest
import zipfile

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from rl.core.config import default_config
from rl.training.checkpoint import (
    BUNDLE_SUFFIX,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointVersionError,
    _MODEL_MEMBER,
    _META_MEMBER,
    load,
    resume_latest,
    save_atomic,
    save_sb3_atomic,
)
from rl.training.train_ppo import (
    NET_ARCH,
    _CheckpointCallback,
    _vec_env,
    run as ppo_run,
)

ANCHOR = os.path.join(os.path.dirname(__file__), "..", "rl", "data", "policy.pt")


def _anchor_hash():
    with open(ANCHOR, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _make_model(timesteps=128, seed=0):
    env = _vec_env(0, n_envs=1, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh
        ),
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)
    env.close()
    return model


class AnchorProtectionMixin:
    """Verify rl/data/policy.pt is never mutated by any test."""

    def setUp(self):
        self._before = _anchor_hash() if os.path.exists(ANCHOR) else None

    def tearDown(self):
        if self._before and os.path.exists(ANCHOR):
            after = _anchor_hash()
            if after != self._before:
                raise AssertionError("anchor policy.pt was mutated")


class OutputParityTests(AnchorProtectionMixin, unittest.TestCase):
    def test_save_load_prediction_parity(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(128)
            model.num_timesteps = 5000
            path = os.path.join(d, "test.ckpt")
            save_atomic(model, path, cfg, stage=0, stage_step=5000)

            loaded, meta = load(path, cfg, device="cpu")
            self.assertEqual(meta["training_step"], 5000)

            obs = np.random.RandomState(0).uniform(-1, 1, (32, 24)).astype(np.float32)
            a1, _ = model.predict(obs, deterministic=True)
            a2, _ = loaded.predict(obs, deterministic=True)
            self.assertTrue(np.array_equal(a1, a2))

    def test_num_timesteps_restored(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            model.num_timesteps = 4242
            path = os.path.join(d, "ts.ckpt")
            save_atomic(model, path, cfg, stage=1, stage_step=100)
            loaded, _ = load(path, cfg, device="cpu")
            self.assertEqual(loaded.num_timesteps, 4242)

    def test_optimizer_state_restored(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(192)
            path = os.path.join(d, "optim.ckpt")
            save_atomic(model, path, cfg)
            loaded, _ = load(path, cfg, device="cpu")

            opt_a = model.policy.optimizer.state_dict()["state"]
            opt_b = loaded.policy.optimizer.state_dict()["state"]
            self.assertTrue(len(opt_a) > 0, "source optim should have momentum state")
            self.assertEqual(set(opt_a.keys()), set(opt_b.keys()))
            for k in opt_a:
                for buf_name in opt_a[k]:
                    va = opt_a[k][buf_name]
                    vb = opt_b[k][buf_name]
                    if isinstance(va, torch.Tensor):
                        self.assertTrue(
                            torch.equal(va, vb), f"optim state[{k}][{buf_name}]"
                        )


class SaveValidationTests(AnchorProtectionMixin, unittest.TestCase):
    def test_training_step_mismatch_rejected_at_save(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "bad.ckpt")
            with self.assertRaises(CheckpointError):
                save_atomic(model, path, cfg, training_step=999)

    def test_extra_key_collision_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "extra.ckpt")
            with self.assertRaises(CheckpointError):
                save_atomic(model, path, cfg, extra={"stage": 99})

    def test_extra_schema_min_version_collision_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "extra2.ckpt")
            with self.assertRaises(CheckpointError):
                save_atomic(model, path, cfg, extra={"schema_min_version": 0})

    def test_load_rejects_metadata_model_mismatch(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(128)
            path = os.path.join(d, "mismatch.ckpt")
            save_atomic(model, path, cfg)
            _patch_metadata(path, {"training_step": 999999})
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")


class SchemaRejectionTests(AnchorProtectionMixin, unittest.TestCase):
    def test_version_zero_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "v0.ckpt")
            save_atomic(model, path, cfg)
            _patch_metadata(path, {"config_version": 0})
            with self.assertRaises(CheckpointVersionError):
                load(path, cfg, device="cpu")

    def test_old_version_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "old.ckpt")
            save_atomic(model, path, cfg)

            cfg_strict = default_config()
            cfg_strict.checkpoint.schema_min_version = 2
            with self.assertRaises(CheckpointVersionError):
                load(path, cfg_strict, device="cpu")

    def test_missing_required_key_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "missing.ckpt")
            save_atomic(model, path, cfg)
            with zipfile.ZipFile(path, "r") as zf:
                model_bytes = zf.read(_MODEL_MEMBER)
            meta = _read_meta(path)
            del meta["training_step"]
            _rewrite_bundle(path, meta, model_bytes)
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")

    def test_wrong_type_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "badtype.ckpt")
            save_atomic(model, path, cfg)
            meta = _read_meta(path)
            meta["training_step"] = "not-an-int"
            with zipfile.ZipFile(path, "r") as zf:
                model_bytes = zf.read(_MODEL_MEMBER)
            _rewrite_bundle(path, meta, model_bytes)
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")

    def test_negative_value_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "neg.ckpt")
            save_atomic(model, path, cfg)
            _patch_metadata(path, {"training_step": -1})
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")


class CorruptionRejectionTests(AnchorProtectionMixin, unittest.TestCase):
    def test_truncated_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "trunc.ckpt")
            save_atomic(model, path, cfg)
            size = os.path.getsize(path)
            with open(path, "r+b") as f:
                f.truncate(size // 2)
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")

    def test_random_bytes_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            path = os.path.join(d, "garbage.ckpt")
            with open(path, "wb") as f:
                f.write(os.urandom(1024))
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")

    def test_missing_model_member_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "nobytes.ckpt")
            save_atomic(model, path, cfg)
            meta_bytes = json.dumps(
                {
                    "config_version": 1,
                    "training_step": model.num_timesteps,
                    "seed": 0,
                    "stage": 0,
                    "stage_step": 0,
                }
            ).encode()
            os.replace(path, path + ".orig")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr(_META_MEMBER, meta_bytes)
            with self.assertRaises(CheckpointError):
                load(path, cfg, device="cpu")


class AtomicWriteTests(AnchorProtectionMixin, unittest.TestCase):
    def test_no_temp_files_after_save(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "clean.ckpt")
            save_atomic(model, path, cfg)
            leftovers = [f for f in os.listdir(d) if f.startswith(".clean.ckpt")]
            self.assertEqual(leftovers, [])

    def test_overwrite_replaces_atomically(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "replace.ckpt")
            model.num_timesteps = 100
            save_atomic(model, path, cfg)
            model.num_timesteps = 200
            save_atomic(model, path, cfg)
            self.assertTrue(os.path.exists(path))
            leftovers = [f for f in os.listdir(d) if f.startswith(".replace.ckpt")]
            self.assertEqual(leftovers, [])

    def test_bundle_contains_required_members(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            model.num_timesteps = 42
            path = os.path.join(d, "members.ckpt")
            save_atomic(model, path, cfg, stage=2, stage_step=42)
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                self.assertIn(_META_MEMBER, names)
                self.assertIn(_MODEL_MEMBER, names)
                meta = json.loads(zf.read(_META_MEMBER))
                for key in (
                    "config_version",
                    "training_step",
                    "seed",
                    "stage",
                    "stage_step",
                ):
                    self.assertIn(key, meta)
                self.assertEqual(meta["stage"], 2)
                self.assertEqual(meta["stage_step"], 42)
                self.assertEqual(meta["training_step"], 42)

    def test_sb3_save_atomic_no_temp(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "policy.zip")
            save_sb3_atomic(model, path)
            self.assertTrue(os.path.exists(path))
            leftovers = [f for f in os.listdir(d) if f.startswith(".policy.zip")]
            self.assertEqual(leftovers, [])


class ResumeLatestTests(AnchorProtectionMixin, unittest.TestCase):
    def test_selects_highest_training_step_not_mtime(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            for step in (100, 300, 200):
                model.num_timesteps = step
                p = os.path.join(d, f"cp_{step:08d}.ckpt")
                save_atomic(model, p, cfg)
            import time

            old = time.time() - 86400
            os.utime(os.path.join(d, "cp_00000300.ckpt"), (old, old))
            os.utime(os.path.join(d, "cp_00000100.ckpt"), (time.time(), time.time()))

            _, meta = resume_latest(d, cfg, device="cpu")
            self.assertEqual(meta["training_step"], 300)

    def test_no_checkpoints_raises_not_found(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(CheckpointNotFoundError):
                resume_latest(d, cfg, device="cpu")

    def test_newest_corrupt_rejected_loudly(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            model.num_timesteps = 100
            p_old = os.path.join(d, "cp_00000100.ckpt")
            save_atomic(model, p_old, cfg)
            model.num_timesteps = 200
            p_new = os.path.join(d, "cp_00000200.ckpt")
            save_atomic(model, p_new, cfg)
            with open(p_new, "r+b") as f:
                f.truncate(os.path.getsize(p_new) // 2)
            with self.assertRaises(CheckpointError):
                resume_latest(d, cfg, device="cpu")

    def test_any_corrupt_candidate_rejected(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            model.num_timesteps = 300
            save_atomic(model, os.path.join(d, "cp_00000300.ckpt"), cfg)
            with open(os.path.join(d, "garbage.ckpt"), "wb") as f:
                f.write(os.urandom(512))
            with self.assertRaises(CheckpointError):
                resume_latest(d, cfg, device="cpu")

    def test_old_schema_preserves_version_error(self):
        cfg = default_config()
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            model = _make_model(64)
            path = os.path.join(d, "cp_00000100.ckpt")
            save_atomic(model, path, cfg)
            _patch_metadata(path, {"config_version": 0})
            with self.assertRaises(CheckpointVersionError):
                resume_latest(d, cfg, device="cpu")


class TrainRunResumeTests(AnchorProtectionMixin, unittest.TestCase):
    def _make_cfg(self, tmpdir, run_name="resume-test"):
        cfg = default_config()
        cfg.checkpoint.dir = os.path.join(tmpdir, "best")
        cfg.checkpoint.save_every_steps = 1000000
        return cfg, os.path.join(tmpdir, "tb"), run_name

    def test_first_run_no_checkpoint_starts_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            model = ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            self.assertIsNotNone(model)
            run_dir = os.path.join(d, "best", rn)
            ckpts = [f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX)]
            self.assertGreater(len(ckpts), 0)

    def test_corrupt_checkpoint_aborts_not_fresh_starts(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)
            for ckpt in os.listdir(run_dir):
                if ckpt.endswith(BUNDLE_SUFFIX):
                    p = os.path.join(run_dir, ckpt)
                    with open(p, "r+b") as f:
                        f.truncate(os.path.getsize(p) // 2)
            with self.assertRaises(CheckpointError):
                ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)

    def test_smoke_writes_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)
            ckpts = [f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX)]
            self.assertGreater(len(ckpts), 0, "final checkpoint must exist")

    def test_resume_skips_completed_work(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            model = ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            first_ts = model.num_timesteps

            model2 = ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            self.assertEqual(model2.num_timesteps, first_ts)

    def test_repeated_completed_run_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)

            ckpts_before = sorted(
                f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX)
            )
            path_before = os.path.join(run_dir, ckpts_before[-1])
            _, meta_before = load(path_before, cfg, device="cpu")

            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)

            ckpts_after = sorted(
                f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX)
            )
            self.assertEqual(
                len(ckpts_after),
                len(ckpts_before),
                "no new checkpoints on completed re-entry",
            )
            path_after = os.path.join(run_dir, ckpts_after[-1])
            _, meta_after = load(path_after, cfg, device="cpu")
            self.assertEqual(meta_before["stage"], meta_after["stage"])
            self.assertEqual(meta_before["stage_step"], meta_after["stage_step"])
            self.assertEqual(meta_before["training_step"], meta_after["training_step"])

    def test_resume_updates_tensorboard_log_directory(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)

            resumed_tb_root = os.path.join(d, "resumed-tb")
            model = ppo_run(
                cfg,
                run_name=rn,
                smoke_steps=64,
                tb_root=resumed_tb_root,
            )

            self.assertEqual(
                model.tensorboard_log,
                os.path.join(resumed_tb_root, rn),
            )

    def test_no_duplicate_progress_header(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            csv_path = os.path.join(d, "best", rn, "progress.csv")
            with open(csv_path) as f:
                rows = list(csv.reader(f))
            headers = [r for r in rows if r == ["ep_rew", "gates_cleared", "n_steps"]]
            self.assertEqual(len(headers), 1)

            ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            with open(csv_path) as f:
                rows2 = list(csv.reader(f))
            headers2 = [r for r in rows2 if r == ["ep_rew", "gates_cleared", "n_steps"]]
            self.assertEqual(len(headers2), 1, "resume must not duplicate header")

    def test_periodic_save_during_training(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            cfg.ppo.n_steps = 64
            cfg.checkpoint.save_every_steps = 64
            ppo_run(cfg, run_name=rn, smoke_steps=256, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)
            ckpts = sorted(f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX))
            self.assertGreaterEqual(
                len(ckpts), 2, "should have periodic + final checkpoints"
            )

    def test_resume_continues_partial_stage(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            cfg.checkpoint.save_every_steps = 64
            model = ppo_run(cfg, run_name=rn, smoke_steps=128, tb_root=tb_root)
            first_ts = model.num_timesteps

            model2 = ppo_run(cfg, run_name=rn, smoke_steps=256, tb_root=tb_root)
            self.assertGreater(model2.num_timesteps, first_ts)

    def test_resume_metadata_tracks_stage(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)
            ckpts = [f for f in os.listdir(run_dir) if f.endswith(BUNDLE_SUFFIX)]
            self.assertGreater(len(ckpts), 0)
            path = os.path.join(run_dir, sorted(ckpts)[-1])
            _, meta = load(path, cfg, device="cpu")
            self.assertEqual(meta["stage"], 0)
            self.assertGreaterEqual(meta["stage_step"], 0)

    def test_policy_ppo_zip_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, tb_root, rn = self._make_cfg(d)
            ppo_run(cfg, run_name=rn, smoke_steps=64, tb_root=tb_root)
            run_dir = os.path.join(d, "best", rn)
            zip_path = os.path.join(run_dir, "policy_ppo.zip")
            self.assertTrue(os.path.exists(zip_path))
            leftovers = [f for f in os.listdir(run_dir) if ".policy_ppo.zip." in f]
            self.assertEqual(leftovers, [])


class CheckpointCallbackTests(AnchorProtectionMixin, unittest.TestCase):
    def test_callback_saves_at_interval(self):
        cfg = default_config()
        cfg.checkpoint.save_every_steps = 64
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            cb = _CheckpointCallback(cfg, d, initial_step=0)
            model = _make_model(64)
            cb.init_callback(model)
            cb.set_stage(0, 0)

            cb.model = model
            model.num_timesteps = 70
            cb._on_rollout_end()
            ckpts = [f for f in os.listdir(d) if f.endswith(BUNDLE_SUFFIX)]
            self.assertEqual(len(ckpts), 1)

            model.num_timesteps = 80
            cb._on_rollout_end()
            ckpts = [f for f in os.listdir(d) if f.endswith(BUNDLE_SUFFIX)]
            self.assertEqual(len(ckpts), 1)

            model.num_timesteps = 140
            cb._on_rollout_end()
            ckpts = [f for f in os.listdir(d) if f.endswith(BUNDLE_SUFFIX)]
            self.assertEqual(len(ckpts), 2)

    def test_callback_save_now_dedup(self):
        cfg = default_config()
        cfg.checkpoint.save_every_steps = 1000000
        with tempfile.TemporaryDirectory() as d:
            cfg.checkpoint.dir = d
            cb = _CheckpointCallback(cfg, d, initial_step=30)
            model = _make_model(64)
            cb.init_callback(model)
            cb.set_stage(0, 0)
            cb.model = model
            model.num_timesteps = 30
            cb.save_now()
            ckpts = [f for f in os.listdir(d) if f.endswith(BUNDLE_SUFFIX)]
            self.assertEqual(len(ckpts), 0, "should skip when ts == initial_step")

            model.num_timesteps = 60
            cb.save_now()
            ckpts = [f for f in os.listdir(d) if f.endswith(BUNDLE_SUFFIX)]
            self.assertEqual(len(ckpts), 1)


class ConfigValidationTests(unittest.TestCase):
    def test_save_every_steps_default(self):
        cfg = default_config()
        self.assertEqual(cfg.checkpoint.save_every_steps, 10_000)

    def test_save_every_steps_must_be_positive(self):
        from rl.core.config import ConfigError, save_config

        cfg = default_config()
        cfg.checkpoint.save_every_steps = 0
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ConfigError):
                save_config(cfg, os.path.join(d, "bad.yaml"))

    def test_default_yaml_has_save_every_steps(self):
        from rl.core.config import load_config

        cfg = load_config("configs/default.yaml")
        self.assertEqual(cfg.checkpoint.save_every_steps, 10_000)


def _read_meta(path):
    with zipfile.ZipFile(path, "r") as zf:
        return json.loads(zf.read(_META_MEMBER))


def _patch_metadata(path, patches):
    with zipfile.ZipFile(path, "r") as zf:
        model_bytes = zf.read(_MODEL_MEMBER)
    meta = _read_meta(path)
    meta.update(patches)
    _rewrite_bundle(path, meta, model_bytes)


def _rewrite_bundle(path, meta, model_bytes):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr(_META_MEMBER, json.dumps(meta))
                zf.writestr(_MODEL_MEMBER, model_bytes)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


if __name__ == "__main__":
    unittest.main()
