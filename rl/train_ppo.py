"""Module 8 (training) — PPO with a 3x64 MLP policy + curriculum.

Trains an SB3 PPO agent over the 24-D observation across the 3 curriculum
stages (single gate -> two gates -> full 6-gate course), reusing weights
between stages. Exports a dependency-light ``policy.pt`` (pure-torch
deterministic actor) for deployment, plus the full SB3 zip for resuming.

    uv run -m rl.train_ppo                 # full curriculum
    uv run -m rl.train_ppo --quick         # tiny smoke run (verifies pipeline)
    uv run -m rl.train_ppo --bc-only       # BC base quality check, no PPO
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import spec
from rl.control import geometric_action
from rl.env import CURRICULUM, DECISION_DT, DECISION_HZ, GateRacingEnv, V_MAX

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "policy_ppo.zip")
POLICY_PT = os.path.join(DATA_DIR, "policy.pt")

NET_ARCH = [64, 64, 64]  # 3x64 MLP (shared depth for pi and vf)


def _vec_env(stage, n_envs=8, seed=0):
    return DummyVecEnv(
        [lambda i=i: GateRacingEnv(stage=stage, seed=seed + i) for i in range(n_envs)]
    )


class StandalonePolicy(nn.Module):
    """Pure-torch deterministic actor: obs(24) -> 3x64 tanh -> action(4)."""

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
    """Copy SB3 PPO actor weights into a StandalonePolicy and save."""
    sb3 = model.policy
    std = StandalonePolicy()
    # SB3 MlpExtractor policy_net + action_net -> our body + head.
    src_body = sb3.mlp_extractor.policy_net.state_dict()
    std.body.load_state_dict(src_body)
    std.head.load_state_dict(sb3.action_net.state_dict())
    std.eval()
    torch.save(
        {
            "state_dict": std.state_dict(),
            "arch": NET_ARCH,
            "obs_dim": spec.OBS_DIM,
            "act_dim": spec.ACTION_DIM,
        },
        out,
    )
    print(f"[ppo] exported standalone policy -> {out}", flush=True)
    return std


def _verify_export(model, std, n=64):
    """Deterministic SB3 action == StandalonePolicy output (within tol)."""
    obs = np.random.uniform(-1, 1, (n, spec.OBS_DIM)).astype(np.float32)
    sb3_act, _ = model.predict(obs, deterministic=True)
    with torch.no_grad():
        mine = std(torch.from_numpy(obs)).numpy()
    mine = np.clip(mine, -1, 1)
    sb3_act = np.clip(sb3_act, -1, 1)
    err = np.abs(sb3_act - mine).max()
    print(f"[ppo] export parity max_abs_action_diff={err:.5f}")
    return err


# ----------------------------------------------------------------------------
# Behavior-cloning warm-start (privileged teacher -> student on noisy obs).
# ----------------------------------------------------------------------------
def bc_collect(n_eps=15, seed=0):
    """Collect (noisy_obs, expert_action) pairs across all curriculum stages.

    The geometric expert acts on the TRUE state (privileged teacher); the
    recorded observation is the noisy obs the student/policy will actually see,
    so the clone learns the perception->action mapping it must use at deploy.
    Expert demos use nominal physics (clean), perception noise stays ON.
    """
    obs_buf, act_buf = [], []
    for stage in range(len(CURRICULUM)):
        for ep in range(n_eps):
            env = GateRacingEnv(
                stage=stage,
                seed=seed + 1000 * stage + ep,
                domain_rand=False,
                perception_noise=True,
            )
            o, _ = env.reset()
            done = trunc = False
            while not (done or trunc):
                target = np.asarray(env.gate_map[env.gate_idx]["pos"], float)
                a = geometric_action(env.p, env.v, env.q, target)
                obs_buf.append(o)
                act_buf.append(a)
                o, _, done, trunc, _ = env.step(a)
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32)


def bc_train(X, Y, epochs=20, batch=256, lr=1e-3, seed=0):
    """Supervised MSE fit of a StandalonePolicy to (obs -> expert action)."""
    torch.manual_seed(seed)
    std = StandalonePolicy()
    opt = torch.optim.Adam(std.parameters(), lr=lr)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss = F.mse_loss(std(Xt[idx]), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[bc] epoch {ep:2d} mse={tot / n:.4f}", flush=True)
    std.eval()
    return std


def import_policy(std: "StandalonePolicy", model: PPO):
    """Warm-start the SB3 PPO actor from a BC-trained StandalonePolicy
    (inverse of export_policy: body -> policy_net, head -> action_net)."""
    sb3 = model.policy
    sb3.mlp_extractor.policy_net.load_state_dict(std.body.state_dict())
    sb3.action_net.load_state_dict(std.head.state_dict())


class _DiagCallback(BaseCallback):
    """Accumulate per-stage diagnostics from env-step infos: the max single-step
    (uncapped) progress and the FRACTION of steps the V_MAX cap binds — to see
    not just that the cap engages but how often it's actively clipping — plus
    mean ||action|| and mean ||omega||, to watch the rate/effort penalties don't
    over-damp the policy. Read-only; never alters training."""

    def __init__(self, prog_cap):
        super().__init__()
        self.prog_cap = prog_cap
        self.max_prog = float("-inf")
        self.capped = 0
        self.act_sum = 0.0
        self.omega_sum = 0.0
        self.n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "raw_prog" not in info:
                continue
            self.max_prog = max(self.max_prog, info["raw_prog"])
            if info["raw_prog"] > self.prog_cap:
                self.capped += 1
            self.act_sum += info["act_norm"]
            self.omega_sum += info["omega_norm"]
            self.n += 1
        return True


def train(
    total_per_stage=300_000,
    n_envs=8,
    quick=False,
    seed=0,
    use_bc=True,
    bc_only=False,
    gate_threshold=None,
    max_retries=2,
):
    os.makedirs(DATA_DIR, exist_ok=True)
    if quick:
        total_per_stage, n_envs = 4000, 4
    if bc_only:
        use_bc = True  # the diagnostic IS the warm-start; PPO is skipped

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

    eval_eps = 5 if quick else 20

    if use_bc:
        print("[ppo] BC warm-start: collecting expert demos ...", flush=True)
        X, Y = bc_collect(n_eps=2 if quick else 15, seed=seed)
        std = bc_train(X, Y, epochs=5 if quick else 20, seed=seed)
        import_policy(std, model)
        print(f"[ppo] BC warm-start applied ({len(X)} samples)", flush=True)
        # BC-only baseline BEFORE any PPO: MSE on actions doesn't guarantee
        # closed-loop behavior (per-step error compounds over a trajectory), so
        # roll out the clone on the early stages to confirm PPO starts from a
        # competent base, not a base PPO has to first undo.
        for s in (0, 1):
            m = evaluate(model, stage=s, episodes=eval_eps)
            print(
                f"[bc] baseline stage {s} success={m['success_rate']:.0%} "
                f"gates={m['gate_clear_rate']:.0%} reward={m['mean_reward']:.1f}",
                flush=True,
            )
        if bc_only:
            print(
                "[bc] --bc-only diagnostic: skipping PPO; policy.pt left untouched",
                flush=True,
            )
            return model

    prog_cap = V_MAX * DECISION_DT  # the per-step progress cap, for comparison
    for stage in range(len(CURRICULUM)):
        diag = _DiagCallback(prog_cap)  # accumulates across this stage's attempts
        for attempt in range(max_retries + 1):
            model.set_env(_vec_env(stage, n_envs, seed + 1000 * stage + 100 * attempt))
            print(
                f"[ppo] === stage {stage} ({CURRICULUM[stage]['num_gates']} gates) "
                f"x {total_per_stage} steps (attempt {attempt}) ===",
                flush=True,
            )
            model.learn(
                total_timesteps=total_per_stage,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=diag,
            )
            # Report success rate BEFORE advancing the curriculum.
            m = evaluate(model, stage=stage, episodes=eval_eps)
            lap = m["mean_lap_s"]
            lap_str = f"{lap:.1f}s" if lap == lap else "n/a"  # nan != nan
            print(
                f"[ppo] stage {stage} success={m['success_rate']:.0%} "
                f"gates={m['gate_clear_rate']:.0%} reward={m['mean_reward']:.1f} "
                f"lap={lap_str}",
                flush=True,
            )
            if gate_threshold is None or m["success_rate"] >= gate_threshold:
                break
            if attempt < max_retries:
                print(
                    f"[ppo] stage {stage} < threshold {gate_threshold:.0%}; repeating",
                    flush=True,
                )
            else:
                print(
                    f"[ppo] WARNING: stage {stage} stuck below {gate_threshold:.0%} "
                    f"after {max_retries} retries; advancing anyway",
                    flush=True,
                )
        if diag.n:
            frac = diag.capped / diag.n
            print(
                f"[diag] stage {stage} max_step_progress={diag.max_prog:.3f}m "
                f"(cap={prog_cap:.3f}m, binds {frac:.1%} of steps) "
                f"mean|action|={diag.act_sum / diag.n:.3f} "
                f"mean|omega|={diag.omega_sum / diag.n:.3f}rad/s",
                flush=True,
            )

    model.save(ZIP_PATH)
    print(f"[ppo] saved SB3 model -> {ZIP_PATH}", flush=True)
    std = export_policy(model)
    err = _verify_export(model, std)
    assert err < 1e-4, (
        f"export parity failed (max_abs_action_diff={err:.5f}); policy.pt would "
        f"not match the trained SB3 actor"
    )
    return model


def evaluate(model, stage=2, episodes=20, seed=999):
    """Roll out the deterministic policy; return success metrics for the stage.

    Returns a dict: course-completion rate, mean gate-clear fraction, mean
    reward, and mean lap time (seconds) over completed runs.
    """
    env = GateRacingEnv(stage=stage, seed=seed)
    rews, gate_fracs, laps, completes = [], [], [], 0
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        done = trunc = False
        tot, passed = 0.0, 0
        while not (done or trunc):
            a, _ = model.predict(o, deterministic=True)
            o, r, done, trunc, info = env.step(a)
            tot += r
            if info.get("gate_passed"):
                passed += 1
            if info.get("course_complete"):
                completes += 1
                laps.append(env.steps / DECISION_HZ)
        rews.append(tot)
        gate_fracs.append(passed / max(len(env.gate_map), 1))
    return {
        "success_rate": completes / episodes,
        "gate_clear_rate": float(np.mean(gate_fracs)),
        "mean_reward": float(np.mean(rews)),
        "mean_lap_s": float(np.mean(laps)) if laps else float("nan"),
        "episodes": episodes,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="steps per stage")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--no-bc", dest="use_bc", action="store_false", help="skip BC warm-start"
    )
    ap.add_argument(
        "--bc-only",
        action="store_true",
        help="BC warm-start + baseline eval on stages 0-1, then stop (no PPO)",
    )
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="repeat a stage until success rate >= this (0-1)",
    )
    ap.add_argument(
        "--retries", type=int, default=2, help="max stage repeats below threshold"
    )
    args = ap.parse_args()
    train(
        total_per_stage=args.steps,
        n_envs=args.envs,
        quick=args.quick,
        use_bc=args.use_bc,
        bc_only=args.bc_only,
        gate_threshold=args.gate_threshold,
        max_retries=args.retries,
    )
