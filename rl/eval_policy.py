"""Evaluate a trained policy in the internal env — gates chained + completion.

Headless (no live sim). Loads rl/data/policy.pt (or --policy), runs N episodes
per curriculum stage on the NOMINAL plant (randomize=False, the honest metric),
and reports mean/best gates passed and course-completion rate.

    uv run -m rl.eval_policy                 # all stages, 20 eps each
    uv run -m rl.eval_policy --episodes 50
    uv run -m rl.eval_policy --policy rl/data/best/s3/best_model.zip
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from rl.env import CURRICULUM, GateRacingEnv
from rl.train_ppo import POLICY_PT, StandalonePolicy


def _load_actor(path: str):
    """Return an act(obs)->action fn for a .pt StandalonePolicy or an SB3 .zip."""
    if path.endswith(".zip"):
        from stable_baselines3 import PPO

        model = PPO.load(path, device="cpu")
        return lambda o: np.clip(
            model.predict(o, deterministic=True)[0], -1.0, 1.0
        )
    ck = torch.load(path, map_location="cpu", weights_only=True)
    pol = StandalonePolicy(obs_dim=ck.get("obs_dim"))
    pol.load_state_dict(ck["state_dict"])
    pol.eval()

    def act(o):
        with torch.no_grad():
            return np.clip(pol(torch.from_numpy(o.astype(np.float32))).numpy(), -1.0, 1.0)

    return act


def evaluate(path: str, episodes: int = 20, seed: int = 9000) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[eval] no policy at {path} — run `make train-ppo` first")
    act = _load_actor(path)
    print(f"[eval] {path} — {episodes} episodes/stage (nominal plant)", flush=True)
    for stage in range(len(CURRICULUM)):
        ng = CURRICULUM[stage]["num_gates"]
        gates, comp = [], 0
        for ep in range(episodes):
            env = GateRacingEnv(stage=stage, seed=seed + ep, randomize=False)
            o, _ = env.reset(seed=seed + ep)
            term = trunc = False
            info: dict = {}
            while not (term or trunc):
                o, _r, term, trunc, info = env.step(act(o))
            gates.append(env.gate_idx)
            comp += int(info.get("course_complete", False))
        print(
            f"  stage {stage}: {ng:2d} gates | mean={np.mean(gates):4.1f} "
            f"| best={max(gates):2d} | completion={100 * comp / episodes:3.0f}%",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=POLICY_PT, help="policy .pt or SB3 .zip")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    evaluate(args.policy, episodes=args.episodes)
