"""PPO trainer for VQ2RealEnv -- real-sim, single instance, frame-stack MLP.

SB3 PPO with a [64,64,64] MLP (the frame-stack in the env supplies short-term
memory, per the plan). One real-time env, so training is slow -- run it
unattended; the reset harness recycles episodes. Optional BC warm-start via
--resume (a prior checkpoint) once a real-sim demo set exists.

    uv run -m rl.vq2.train --steps 200000            # train
    uv run -m rl.vq2.train --resume rl/data/vq2/ckpts/vq2_ppo_100000_steps.zip
"""

from __future__ import annotations

import argparse
import json
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from rl.vq2.gym_env import DEFAULT_NUM_GATES, VQ2RealEnv

DATA = os.path.join("rl", "data", "vq2")


class EpisodeLog(BaseCallback):
    """Per-episode log so you can SEE whether gates registered and which reward
    terms fired. Appends one JSON line per episode to `episodes.jsonl` AND records
    gates_passed / min_range / per-term reward sums to TensorBoard."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self._ep = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            self._ep += 1
            ep = info.get("episode", {})            # Monitor-added {r, l}
            parts = info.get("reward_parts", {})
            rec = {
                "ep": self._ep, "t": int(self.num_timesteps),
                "R": round(float(ep.get("r", 0.0)), 2), "len": int(ep.get("l", 0)),
                "gates": info.get("gates_passed"), "reason": info.get("reason"),
                "min_range": (round(info["ep_min_range"], 2)
                              if info.get("ep_min_range") is not None else None),
                "hard_col": info.get("n_hard_col"), "soft_col": info.get("n_soft_col"),
                "parts": {k: round(float(v), 2) for k, v in parts.items()},
            }
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            # TensorBoard scalars (watch these trend over training).
            self.logger.record("episode/gates_passed", float(info.get("gates_passed", 0)))
            if info.get("ep_min_range") is not None:
                self.logger.record("episode/min_gate_range", float(info["ep_min_range"]))
            self.logger.record("episode/hard_collisions", float(info.get("n_hard_col", 0)))
            for k, v in parts.items():
                self.logger.record(f"reward/{k}", float(v))
        return True


def _bc_dataset():
    """Accumulated successful segments (preference-weighted) + expert demos.npz."""
    import numpy as np

    from rl.vq2 import success

    obs, act = success.load_for_bc()
    parts_o = [obs] if obs is not None else []
    parts_a = [act] if act is not None else []
    n_succ = len(obs) if obs is not None else 0
    if os.path.exists(os.path.join(DATA, "demos.npz")):
        d = np.load(os.path.join(DATA, "demos.npz"))
        parts_o.append(d["obs"].astype(np.float32))
        parts_a.append(d["act"].astype(np.float32))
    if not parts_o:
        return None, None, 0, 0
    obs = np.concatenate(parts_o)
    act = np.concatenate(parts_a)
    return obs, act, n_succ, len(obs) - n_succ


def train(total_steps: int = 200_000, n_steps: int = 1024, seconds: float = 30.0,
          gates: int = DEFAULT_NUM_GATES, resume: str | None = None,
          bc_epochs: int = 40) -> None:
    os.makedirs(os.path.join(DATA, "ckpts"), exist_ok=True)
    # Closed-loop, online: the policy flies the real sim and saves every segment
    # that reaches a new gate as it happens.
    env = VQ2RealEnv(max_seconds=seconds, num_gates=gates, record_success=True)
    if resume and os.path.exists(resume):
        print(f"[vq2.train] resuming from {resume}", flush=True)
        model = PPO.load(resume, env=env, device="cpu",
                         tensorboard_log=os.path.join(DATA, "tb"))
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=n_steps, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[64, 64, 64]), device="cpu",
            tensorboard_log=os.path.join(DATA, "tb"), verbose=1,
        )
        # Initialize the policy from ALL accumulated successful trajectories
        # (+ the expert bootstrap) instead of starting from random exploration.
        from rl.vq2.train_bc import fit_policy
        obs, act, n_succ, n_demo = _bc_dataset()
        if obs is not None:
            print(f"[vq2.train] BC-init on {len(obs)} samples "
                  f"({n_succ} from successes, {n_demo} from demos)...", flush=True)
            fit_policy(model.policy, obs, act, epochs=bc_epochs)
        else:
            print("[vq2.train] no demos/successes yet -> starting from scratch.", flush=True)
    ckpt = CheckpointCallback(save_freq=n_steps * 5,
                              save_path=os.path.join(DATA, "ckpts"),
                              name_prefix="vq2_ppo")
    eplog = EpisodeLog(os.path.join(DATA, "episodes.jsonl"))
    print(f"[vq2.train] episode log -> {os.path.join(DATA, 'episodes.jsonl')}", flush=True)
    try:
        model.learn(total_timesteps=total_steps, callback=[ckpt, eplog])
    finally:
        model.save(os.path.join(DATA, "vq2_ppo"))
        env.close()
    print(f"[vq2.train] saved {os.path.join(DATA, 'vq2_ppo')}.zip", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--gates", type=int, default=DEFAULT_NUM_GATES)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--bc-epochs", type=int, default=40,
                    help="BC-init epochs on accumulated successes+demos (fresh runs only)")
    args = ap.parse_args()
    train(args.steps, args.n_steps, args.seconds, args.gates, args.resume, args.bc_epochs)
