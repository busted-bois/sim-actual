"""Evaluate a trained VQ2 policy in the real simulator.

    uv run -m rl.vq2.eval --model rl/data/vq2/vq2_ppo.zip --episodes 5
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from rl.vq2.gym_env import VQ2RealEnv


def evaluate(model_path: str, episodes: int = 5, seconds: float = 30.0,
             gates: int = 6) -> None:
    env = VQ2RealEnv(max_seconds=seconds, num_gates=gates)
    model = PPO.load(model_path)
    passed_hist = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                total += r
                done = terminated or truncated
            passed_hist.append(info.get("gates_passed", 0))
            print(f"  ep {ep+1}: gates={info.get('gates_passed')} "
                  f"steps={info.get('steps')} reason={info.get('reason')} R={total:.1f}",
                  flush=True)
    finally:
        env.close()
    if passed_hist:
        import numpy as np
        print(f"\n[vq2.eval] gates passed: mean={np.mean(passed_hist):.1f} "
              f"max={max(passed_hist)} over {episodes} episodes", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="rl/data/vq2/vq2_ppo.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--gates", type=int, default=6)
    args = ap.parse_args()
    evaluate(args.model, args.episodes, args.seconds, args.gates)
