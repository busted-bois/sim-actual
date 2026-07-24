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
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl.vq2.gym_env import VQ2RealEnv

DATA = os.path.join("rl", "data", "vq2")


def train(total_steps: int = 200_000, n_steps: int = 1024, seconds: float = 30.0,
          gates: int = 6, resume: str | None = None) -> None:
    os.makedirs(os.path.join(DATA, "ckpts"), exist_ok=True)
    env = VQ2RealEnv(max_seconds=seconds, num_gates=gates)
    if resume and os.path.exists(resume):
        print(f"[vq2.train] resuming from {resume}", flush=True)
        model = PPO.load(resume, env=env, tensorboard_log=os.path.join(DATA, "tb"))
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=n_steps, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[64, 64, 64]), device="cpu",
            tensorboard_log=os.path.join(DATA, "tb"), verbose=1,
        )
    ckpt = CheckpointCallback(save_freq=n_steps * 5,
                              save_path=os.path.join(DATA, "ckpts"),
                              name_prefix="vq2_ppo")
    try:
        model.learn(total_timesteps=total_steps, callback=ckpt)
    finally:
        model.save(os.path.join(DATA, "vq2_ppo"))
        env.close()
    print(f"[vq2.train] saved {os.path.join(DATA, 'vq2_ppo')}.zip", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    train(args.steps, args.n_steps, args.seconds, args.gates, args.resume)
