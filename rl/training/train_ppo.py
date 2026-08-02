"""Module 8 (training) — PPO with a 3x64 MLP policy + curriculum.

Trains an SB3 PPO agent over the 24-D observation across the 3 curriculum
stages (single gate -> two gates -> full 6-gate course), reusing weights
between stages. Exports a dependency-light ``policy.pt`` (pure-torch
deterministic actor) for deployment, plus the full SB3 zip for resuming.

    uv run -m rl.training.train_ppo --config configs/default.yaml --smoke 2000 --run-name my-run
    uv run -m rl.training.train_ppo                 # full curriculum
    uv run -m rl.training.train_ppo --quick         # tiny smoke run (deprecated alias)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.core.config import RLConfig, load_config, resolve_device
from rl.core import spec
from rl.environment.env import CURRICULUM, DECISION_HZ, GateRacingEnv
from rl.training.checkpoint import (
    CheckpointNotFoundError,
    save_atomic,
    save_sb3_atomic,
    resume_latest,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
ZIP_PATH = os.path.join(DATA_DIR, "policy_ppo.zip")
POLICY_PT = os.path.join(DATA_DIR, "policy.pt")
POLICY_BC_PT = os.path.join(DATA_DIR, "policy_bc.pt")
LEGACY_RUN_DIR = os.path.join(DATA_DIR, "best", "legacy")

NET_ARCH = [64, 64, 64]

_UNSAFE_RUN_NAME = re.compile(r"(?:[./\\]|\.\.)")


def validate_run_name(name: str) -> str:
    if not name or name.strip() != name:
        raise ValueError(
            f"run_name must be non-empty and not start/end with whitespace: {name!r}"
        )
    if _UNSAFE_RUN_NAME.search(name):
        raise ValueError(f"run_name must not contain path separators or '..': {name!r}")
    return name


def _infer_arch_from_sb3(model: PPO) -> list[int]:
    layers = list(model.policy.mlp_extractor.policy_net)
    arch = []
    for layer in layers:
        if isinstance(layer, nn.Linear):
            arch.append(layer.out_features)
    return arch


class _ProgressCallback(BaseCallback):
    def __init__(self, csv_path: str):
        super().__init__()
        self._csv_path = csv_path
        self._header_written = False
        self._file_size_at_init = 0

    def _init_callback(self) -> None:
        if os.path.exists(self._csv_path):
            self._file_size_at_init = os.path.getsize(self._csv_path)
            if self._file_size_at_init > 0:
                self._header_written = True
        else:
            self._file_size_at_init = 0
            self._header_written = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                ep_rew = float(ep["r"])
                gates = int(info.get("gates_cleared", 0))
                global_steps = self.num_timesteps
                Path(self._csv_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self._csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not self._header_written:
                        writer.writerow(["ep_rew", "gates_cleared", "n_steps"])
                        self._header_written = True
                    writer.writerow([ep_rew, gates, global_steps])
        return True


class _CheckpointCallback(BaseCallback):
    """Periodic atomic checkpoint saver.

    Saves at rollout boundaries (``_on_rollout_end``) using true
    ``model.num_timesteps`` so it is immune to the n_env-frequency bugs that
    affect SB3's ``CheckpointCallback``.  The final save is guaranteed via
    :meth:`save_now` from the runner.
    """

    def __init__(self, cfg: RLConfig, ckpt_dir: str, initial_step: int = 0):
        super().__init__()
        self._cfg = cfg
        self._ckpt_dir = ckpt_dir
        self._save_every = cfg.checkpoint.save_every_steps
        self._last_save = initial_step
        self._stage = 0
        self._stage_start = 0

    def set_stage(self, stage: int, stage_start_ts: int) -> None:
        self._stage = stage
        self._stage_start = stage_start_ts

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        ts = int(self.model.num_timesteps)
        if ts - self._last_save >= self._save_every:
            self._write(ts)
            self._last_save = ts

    def save_now(self) -> None:
        ts = int(self.model.num_timesteps)
        if ts == self._last_save:
            return
        self._write(ts)
        self._last_save = ts

    def _write(self, ts: int) -> None:
        stage_step = ts - self._stage_start
        path = os.path.join(self._ckpt_dir, f"checkpoint_{ts:08d}.ckpt")
        save_atomic(
            self.model,
            path,
            self._cfg,
            training_step=ts,
            stage=self._stage,
            stage_step=stage_step,
        )


def _make_env(stage: int, seed: int = 0, max_steps: int | None = None):
    if max_steps is None:
        max_seconds = 20.0
    else:
        max_seconds = max_steps / DECISION_HZ

    def _thunk():
        return Monitor(
            GateRacingEnv(stage=stage, max_seconds=max_seconds, seed=seed),
            info_keywords=("gates_cleared",),
        )

    return _thunk


def _vec_env(stage, n_envs=8, seed=0, *, max_steps: int | None = None):
    if max_steps is None:
        max_steps = int(20.0 * DECISION_HZ)
    max_seconds = max_steps / DECISION_HZ
    return DummyVecEnv(
        [
            lambda i=i, s=stage, ms=max_seconds, sd=seed: Monitor(
                GateRacingEnv(stage=s, max_seconds=ms, seed=sd + i),
                info_keywords=("gates_cleared",),
            )
            for i in range(n_envs)
        ]
    )


class StandalonePolicy(nn.Module):
    def __init__(self, obs_dim=spec.OBS_DIM, act_dim=spec.ACTION_DIM, arch=NET_ARCH):
        super().__init__()
        layers, last = [], obs_dim
        for h in arch:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(last, act_dim)

    def forward(self, x):
        return self.head(self.body(x))


def export_policy(model: PPO, out: str = POLICY_PT):
    sb3 = model.policy
    arch = _infer_arch_from_sb3(model)
    std = StandalonePolicy(arch=arch)
    src_body = sb3.mlp_extractor.policy_net.state_dict()
    std.body.load_state_dict(src_body)
    std.head.load_state_dict(sb3.action_net.state_dict())
    std.eval()
    torch.save(
        {
            "state_dict": std.state_dict(),
            "arch": arch,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
            "train_hover_thrust": spec.HOVER_THRUST,
            "action_scale": [
                spec.MAX_ROLL_RATE,
                spec.MAX_PITCH_RATE,
                spec.MAX_YAW_RATE,
            ],
        },
        out,
    )
    print(f"[ppo] exported standalone policy -> {out}", flush=True)
    return std


def _verify_export(model, std, n=64):
    obs = np.random.uniform(-1, 1, (n, spec.OBS_DIM)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mine = std(torch.from_numpy(obs)).numpy()
    mine = np.clip(mine, -1, 1)
    sb3_act = np.clip(sb3_act, -1, 1)
    err = np.abs(sb3_act - mine).max()
    print(f"[ppo] export parity max_abs_action_diff={err:.5f}")
    return err


def load_bc_init(model: PPO, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt_arch = ckpt.get("arch")
    sb3_arch = _infer_arch_from_sb3(model)
    if ckpt_arch != sb3_arch:
        raise ValueError(
            f"BC checkpoint arch {ckpt_arch} does not match SB3 actor arch {sb3_arch}"
        )
    std = StandalonePolicy(arch=ckpt_arch)
    std.load_state_dict(ckpt["state_dict"])
    model.policy.mlp_extractor.policy_net.load_state_dict(std.body.state_dict())
    model.policy.action_net.load_state_dict(std.head.state_dict())
    print(f"[ppo] warm-started actor from BC weights {path}", flush=True)


def train(total_per_stage=300_000, n_envs=8, quick=False, seed=0, bc_init=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    if quick:
        total_per_stage, n_envs = 4000, 4

    policy_kwargs = dict(net_arch=dict(pi=NET_ARCH, vf=NET_ARCH), activation_fn=nn.Tanh)
    env0 = _vec_env(0, n_envs, seed)
    model = PPO(
        "MlpPolicy",
        env0,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=1024 if not quick else 256,
        batch_size=256 if not quick else 64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.005,
        learning_rate=3e-4,
        clip_range=0.2,
        n_epochs=10,
        seed=seed,
        device="cpu",
    )

    if bc_init is None and os.path.exists(POLICY_BC_PT):
        bc_init = POLICY_BC_PT
    if bc_init:
        load_bc_init(model, bc_init)

    for stage in range(len(CURRICULUM)):
        model.set_env(_vec_env(stage, n_envs, seed + 1000 * stage))
        print(
            f"[ppo] === stage {stage} ({CURRICULUM[stage]['num_gates']} gates) "
            f"x {total_per_stage} steps ===",
            flush=True,
        )
        model.learn(
            total_timesteps=total_per_stage,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        rew = evaluate(model, stage=stage, episodes=5)
        print(f"[ppo] stage {stage} mean eval reward={rew:.1f}", flush=True)

    os.makedirs(LEGACY_RUN_DIR, exist_ok=True)
    legacy_zip_path = os.path.join(LEGACY_RUN_DIR, "policy_ppo.zip")
    legacy_policy_path = os.path.join(LEGACY_RUN_DIR, "policy.pt")
    save_sb3_atomic(model, legacy_zip_path)
    print(f"[ppo] saved SB3 model -> {legacy_zip_path}", flush=True)
    std = export_policy(model, out=legacy_policy_path)
    err = _verify_export(model, std)
    assert err < 1e-4, (
        f"export parity failed (max_abs_action_diff={err:.5f}); policy.pt would "
        f"not match the trained SB3 actor"
    )
    return model


def evaluate(model, stage=2, episodes=10, seed=999):
    env = GateRacingEnv(stage=stage, seed=seed)
    rews = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = trunc = False
        tot = 0.0
        while not (done or trunc):
            a, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a)
            tot += r
        rews.append(tot)
    return float(np.mean(rews))


def _largest_safe_divisor(rollout_size: int, configured_batch: int) -> int:
    if rollout_size < 2:
        return max(1, rollout_size)
    for d in range(min(configured_batch, rollout_size), 1, -1):
        if rollout_size % d == 0:
            return d
    return 2


def _select_rollout_size(total_timesteps: int, max_n_steps: int, n_envs: int) -> int:
    """Select a rollout size (n_steps*n_envs) that divides *total_timesteps*.

    Prefers the largest value <= *max_n_steps* that divides evenly.
    """
    target = total_timesteps // n_envs if n_envs > 1 else total_timesteps
    for candidate in range(min(target, max_n_steps), 1, -1):
        if total_timesteps % (candidate * n_envs) == 0 and candidate <= max_n_steps:
            return candidate * n_envs
    best = 2 * n_envs
    if best > total_timesteps:
        best = total_timesteps
    return best


def run(
    cfg: RLConfig,
    *,
    run_name: str = "ppo",
    smoke_steps: int | None = None,
    tb_root: str | None = None,
    bc_init_arg: str | None = None,
) -> PPO:
    validate_run_name(run_name)

    from rl.environment.env import CURRICULUM as _CUR

    if cfg.env.curriculum_stage < 0 or cfg.env.curriculum_stage >= len(_CUR):
        raise ValueError(
            f"env.curriculum_stage {cfg.env.curriculum_stage} out of range "
            f"[0, {len(_CUR) - 1}]"
        )

    device = resolve_device(cfg.device)
    ckpt_dir = str(Path(cfg.checkpoint.dir) / run_name)
    if tb_root is None:
        tb_root = str(Path(DATA_DIR) / "tb")
    tb_log_dir = os.path.join(tb_root, run_name)

    n_steps_cfg = cfg.ppo.n_steps
    batch_cfg = cfg.ppo.batch_size
    n_epochs_cfg = cfg.ppo.n_epochs
    gamma = cfg.ppo.gamma
    lr = cfg.ppo.lr
    net_arch = cfg.ppo.net_arch

    is_smoke = smoke_steps is not None
    if is_smoke:
        if smoke_steps < 2:
            raise ValueError(f"smoke_steps must be >= 2, got {smoke_steps}")
        device = torch.device("cpu")
        n_envs = 1
        n_epochs = min(n_epochs_cfg, 2)
        total_timesteps = smoke_steps
        rollout_size = _select_rollout_size(total_timesteps, n_steps_cfg, n_envs)
        batch_size = _largest_safe_divisor(rollout_size, batch_cfg)
        n_steps = rollout_size // n_envs
    else:
        total_timesteps = cfg.ppo.total_timesteps_per_stage
        n_envs = cfg.ppo.n_envs
        batch_size = batch_cfg
        n_epochs = n_epochs_cfg
        n_steps = n_steps_cfg
        rollout_size = n_steps_cfg * n_envs

    os.makedirs(ckpt_dir, exist_ok=True)
    policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=nn.Tanh)
    max_s = cfg.env.max_steps

    if is_smoke:
        end_stage = 1
    else:
        end_stage = cfg.env.curriculum_stage + 1

    model: PPO | None = None
    start_stage = 0
    stage_start_ts = 0
    resumed_ts = 0
    did_train = False

    if cfg.checkpoint.resume:
        try:
            model, meta = resume_latest(ckpt_dir, cfg, env=None, device=str(device))
            resumed_ts = int(meta["training_step"])
            rstage = int(meta["stage"])
            rstage_step = int(meta["stage_step"])
            stage_start_ts = resumed_ts - rstage_step
            if rstage_step >= total_timesteps:
                start_stage = rstage + 1
                stage_start_ts = resumed_ts
            else:
                start_stage = rstage
            model.tensorboard_log = tb_log_dir
            print(
                f"[ppo] resumed from checkpoint: training_step={resumed_ts}, "
                f"stage={rstage}, stage_step={rstage_step} -> start_stage={start_stage}",
                flush=True,
            )
        except CheckpointNotFoundError:
            model = None

    if model is None:
        env0 = DummyVecEnv(
            [_make_env(0, cfg.seed, max_steps=max_s) for _ in range(n_envs)]
        )
        model = PPO(
            "MlpPolicy",
            env0,
            policy_kwargs=policy_kwargs,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            gae_lambda=0.95,
            gamma=gamma,
            ent_coef=0.005,
            learning_rate=lr,
            clip_range=0.2,
            n_epochs=n_epochs,
            seed=cfg.seed,
            device=device,
            tensorboard_log=tb_log_dir,
        )

        bc_init_path = bc_init_arg
        if bc_init_path is None and not is_smoke and os.path.exists(POLICY_BC_PT):
            bc_init_path = POLICY_BC_PT
        if bc_init_path:
            load_bc_init(model, bc_init_path)

    csv_path = os.path.join(ckpt_dir, "progress.csv")
    callbacks: list[BaseCallback] = [_ProgressCallback(csv_path)]
    ckpt_cb = _CheckpointCallback(cfg, ckpt_dir, initial_step=resumed_ts)
    ckpt_cb.model = model
    callbacks.append(ckpt_cb)

    seeds = list(cfg.seeds)
    for stage in range(start_stage, end_stage):
        if stage != start_stage:
            stage_start_ts = int(model.num_timesteps)

        stage_seed = seeds[stage % len(seeds)]
        model.set_env(
            DummyVecEnv(
                [
                    _make_env(
                        stage,
                        seed=stage_seed + i,
                        max_steps=max_s,
                    )
                    for i in range(n_envs)
                ]
            )
        )
        ckpt_cb.set_stage(stage, stage_start_ts)

        current_stage_step = int(model.num_timesteps) - stage_start_ts
        remaining = total_timesteps - current_stage_step
        print(
            f"[ppo] === stage {stage} ({CURRICULUM[stage]['num_gates']} gates) "
            f"| remaining={remaining}/{total_timesteps} steps ===",
            flush=True,
        )

        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                reset_num_timesteps=False,
                progress_bar=False,
                tb_log_name="ppo",
                callback=callbacks,
            )
            did_train = True

        ckpt_cb.model = model
        ckpt_cb.save_now()

    zip_path = os.path.join(ckpt_dir, "policy_ppo.zip")
    save_sb3_atomic(model, zip_path)
    print(f"[ppo] saved SB3 model -> {zip_path}", flush=True)

    std = export_policy(model, out=os.path.join(ckpt_dir, "policy.pt"))
    err = _verify_export(model, std)
    assert err < 1e-4, f"export parity failed (max_abs_action_diff={err:.5f})"

    if did_train:
        ckpt_cb.model = model
        ckpt_cb.save_now()

    return model


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Config-driven PPO training runner.",
    )
    ap.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML config path (default: configs/default.yaml)",
    )
    ap.add_argument(
        "--run-name",
        default="ppo",
        help="Run identifier for logs/checkpoints (default: ppo)",
    )
    ap.add_argument(
        "--smoke",
        type=int,
        default=None,
        metavar="N",
        help="Bounded smoke mode: N total timesteps on CPU, stage 0",
    )
    ap.add_argument("--quick", action="store_true", help="Deprecated: use --smoke 4000")
    ap.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override ppo.total_timesteps_per_stage from config",
    )
    ap.add_argument(
        "--envs",
        type=int,
        default=None,
        help="Override ppo.n_envs from config",
    )
    ap.add_argument(
        "--bc-init",
        default=None,
        help="BC checkpoint to warm-start (default: policy_bc.pt if exists)",
    )
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()

    if args.quick and args.smoke is None:
        args.smoke = 4000

    cfg = load_config(args.config)

    if args.steps is not None:
        cfg.ppo.total_timesteps_per_stage = args.steps
    if args.envs is not None:
        cfg.ppo.n_envs = args.envs

    if args.smoke is not None:
        run(cfg, run_name=args.run_name, smoke_steps=args.smoke)
    else:
        run(cfg, run_name=args.run_name, bc_init_arg=args.bc_init)
