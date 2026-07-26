"""Evaluate a trained VQ2 policy in the real simulator.

    uv run -m rl.vq2.eval --model rl/data/vq2/vq2_ppo.zip --episodes 5
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from rl.vq2.gym_env import DEFAULT_NUM_GATES, VQ2RealEnv


def evaluate(model_path: str, episodes: int = 5, seconds: float = 30.0,
             gates: int = DEFAULT_NUM_GATES) -> None:
    env = VQ2RealEnv(max_seconds=seconds, num_gates=gates)
    model = PPO.load(model_path, device="cpu")
    passed_hist = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            info = {}
            # per-episode flight trace so we can SEE what the drone did.
            min_range = float("inf")   # closest it got to any gate
            vis_steps = 0              # steps with a gate visible
            max_fwd = 0.0              # farthest forward from spawn
            alt_lo, alt_hi = float("inf"), float("-inf")
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                total += r
                done = terminated or truncated
                gr = info.get("gate_range", -1.0)
                if info.get("gate_visible"):
                    vis_steps += 1
                    if gr > 0:
                        min_range = min(min_range, gr)
                max_fwd = max(max_fwd, info.get("fwd_m", 0.0))
                a = info.get("alt_m", 0.0)
                alt_lo, alt_hi = min(alt_lo, a), max(alt_hi, a)
            passed_hist.append(info.get("gates_passed", 0))
            mr = f"{min_range:.1f}m" if min_range != float("inf") else "never-seen"
            print(f"  ep {ep+1}: gates={info.get('gates_passed')} "
                  f"steps={info.get('steps')} reason={info.get('reason')} R={total:.1f}"
                  f"  | closest-gate={mr} vis={vis_steps}st "
                  f"fwd={max_fwd:.1f}m alt=[{alt_lo:.1f},{alt_hi:.1f}]m"
                  f"  col:hard={info.get('n_hard_col')} soft={info.get('n_soft_col')} "
                  f"lastthreat={info.get('last_threat')} delta={info.get('last_delta'):.2f}",
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
    ap.add_argument("--gates", type=int, default=DEFAULT_NUM_GATES)
    args = ap.parse_args()
    evaluate(args.model, args.episodes, args.seconds, args.gates)
