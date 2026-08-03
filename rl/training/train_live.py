"""Train the VQ2 racing policy ON THE LIVE SIMULATOR.

Every environment step in this file is a real MAVLink command to FlightSim and
a real camera frame back. There is no surrogate anywhere in this path.

Algorithm choice is forced by the price of data. The live sim yields tens of
steps per second, not thousands, and each episode costs a reset on top. On-policy
PPO throws every transition away after a single update, which is unaffordable
here; SAC keeps a replay buffer and can take many gradient steps per environment
step, so each hard-won live transition is learned from repeatedly. That is the
entire reason this is SAC and not PPO.

Three things make a live run survivable rather than a way to lose a night:

* The replay buffer is SEEDED from the classical GP pilot flying the real
  course. That pilot is vision-only and already clears 4-5 gates, so SAC starts
  from real successful trajectories instead of random flailing into the ground.
* Model AND replay buffer are checkpointed every few episodes. A fly-away that
  needs a FlightSim restart costs you the sim session, not the training run --
  relaunch and pass --resume.
* A reset that fails raises instead of hanging, so a wedged simulator surfaces
  in seconds rather than silently burning hours.

    make train-vq2                       # or: uv run -m rl.training.train_live
    uv run -m rl.training.train_live --resume
    uv run -m rl.training.train_live --demo-episodes 0   # skip GP seeding
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from rl.core.diagnostics import RunnerLog

OUT_DIR = os.path.join("rl", "data", "live")
MODEL_PATH = os.path.join(OUT_DIR, "sac_live")
BUFFER_PATH = os.path.join(OUT_DIR, "sac_live_buffer.pkl")

# Off-policy hyperparameters tuned for expensive data, not for throughput.
BUFFER_SIZE = 200_000
BATCH_SIZE = 256
# 0, NOT the SB3 default. learning_starts gates on num_timesteps, which seeding
# the replay buffer does NOT advance -- so any positive value flies that many
# steps of UNIFORM RANDOM actions (thrust sampled over the full range) on the
# real drone before the policy is ever consulted. With the buffer already
# seeded from the GP pilot there is nothing to wait for.
LEARNING_STARTS = 0
TRAIN_FREQ = 1
# 1, not 4. Measured on this box: 13.27 ms per SAC gradient step against a
# 33.3 ms control period at 30 Hz. Four inline steps is 53 ms -- 1.6x the whole
# period -- which would leave the drone flying open-loop on a latched setpoint
# for most of every tick. Replay ratio is recovered in the reset window instead,
# where the time is already dead (see RESET_GRADIENT_STEPS).
GRADIENT_STEPS = 1
# The ~4.2 s reset is dead wall-clock; ~300 gradient steps fit in it for free.
RESET_GRADIENT_STEPS = 300
LEARNING_RATE = 3e-4
CHECKPOINT_EVERY_EPISODES = 3


def make_live_env(control_hz: float, max_episode_s: float, verbose: bool = True):
    from rl.environment.live_env import LiveVQ2Env

    return LiveVQ2Env(
        control_hz=control_hz, max_episode_s=max_episode_s, verbose=verbose
    )


def collect_gp_demos(env, n_episodes: int, log: RunnerLog | None = None):
    """Fly the classical GP pilot on the live sim and return its transitions.

    GPPilot's compute_guidance is vision-only (YOLO+PnP -> bearing/elevation),
    so it runs under the VQ2 block and already clears 4-5 gates. Its rollouts
    are the closest thing to expert demonstrations that exists live -- the
    internal-model expert cannot be used here because it reads true position.

    Returns a list of (obs, next_obs, action, reward, done) tuples ready for
    ReplayBuffer.add().
    """
    from simulator.gp_pilot import _fresh_hold_state, compute_guidance

    from rl.core import spec

    transitions = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        hold = _fresh_hold_state()
        term = trunc = False
        n = 0
        while not (term or trunc):
            snap_roll, snap_pitch, _ = env.ahrs.euler_deg()
            vision = _gp_vision_packet(env)
            rd, pd, yd, thrust, _dbg = compute_guidance(
                roll_deg=snap_roll,
                pitch_deg=snap_pitch,
                quat=np.asarray(env.ahrs.q, dtype=float),
                vY=0.0,
                vD=0.0,
                vision=vision,
                vision_vel=None,
                state=hold,
            )
            # compute_guidance emits DEGREE attitude commands; the env's action
            # space is normalized body rates. Convert the same way the RL expert
            # path does (deg -> rad/s), then normalize by the action scale.
            rates = np.radians([rd, pd, yd])
            action = np.array(
                [
                    rates[0] / spec.MAX_ROLL_RATE,
                    rates[1] / spec.MAX_PITCH_RATE,
                    rates[2] / spec.MAX_YAW_RATE,
                    2.0 * float(thrust) - 1.0,
                ],
                dtype=np.float32,
            )
            action = np.clip(action, -1.0, 1.0)
            nxt, reward, term, trunc, _info = env.step(action)
            transitions.append((obs, nxt, action, reward, term))
            obs = nxt
            n += 1
        msg = f"demo episode {ep + 1}/{n_episodes}: {n} steps, gates={env.gates_passed}"
        if log is not None:
            log.info(msg, component="demos")
        else:
            print(f"[demos] {msg}", flush=True)
    return transitions


def _gp_vision_packet(env):
    """Shape the live vision estimate the way compute_guidance expects."""
    est = env._vision()
    if not est:
        return None
    gb = np.asarray(est["gate_pos_body"], dtype=float).reshape(3)
    return {
        "body_x_m": float(gb[0]),
        "body_y_m": float(gb[1]),
        "body_z_m": float(gb[2]),
        "frame_id": int(time.monotonic() * 1000) & 0xFFFF,
        "normal_body": est.get("normal_body"),
    }


def seed_replay_buffer(model, transitions, log: RunnerLog | None = None):
    """Pre-fill SAC's replay buffer with the GP pilot's live transitions."""
    if not transitions:
        return 0
    for obs, nxt, action, reward, done in transitions:
        model.replay_buffer.add(
            np.asarray(obs, np.float32).reshape(1, -1),
            np.asarray(nxt, np.float32).reshape(1, -1),
            np.asarray(action, np.float32).reshape(1, -1),
            np.asarray([reward], np.float32),
            np.asarray([done], bool),
            [{}],
        )
    msg = f"seeded replay buffer with {len(transitions)} GP transitions"
    if log is not None:
        log.info(msg, component="seed")
    else:
        print(f"[seed] {msg}", flush=True)
    return len(transitions)


class LiveCheckpoint:
    """Save model + replay buffer periodically so a sim crash is not fatal."""

    def __init__(self, model, model_path, buffer_path, log=None):
        self.model = model
        self.model_path = model_path
        self.buffer_path = buffer_path
        self.log = log

    def save(self, tag=""):
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        self.model.save(self.model_path)
        self.model.save_replay_buffer(self.buffer_path)
        msg = f"checkpoint{tag} -> {self.model_path}(.zip) + buffer"
        if self.log is not None:
            self.log.info(msg, component="checkpoint")
        else:
            print(f"[checkpoint] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200, help="live episodes to train")
    ap.add_argument("--demo-episodes", type=int, default=5, help="GP seeding episodes")
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--max-episode-s", type=float, default=90.0)
    ap.add_argument("--resume", action="store_true", help="load model + buffer")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    from stable_baselines3 import SAC

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, os.path.basename(MODEL_PATH))
    buffer_path = os.path.join(args.out_dir, os.path.basename(BUFFER_PATH))
    log = RunnerLog(tag="vq2-live", log_dir=os.path.join("logs", "vq2-live"))

    env = None
    try:
        log.info(
            f"LIVE training: episodes={args.episodes} demo_episodes={args.demo_episodes} "
            f"control_hz={args.control_hz} device={args.device}",
            component="config",
        )
        print("=== connecting to the LIVE simulator ===", flush=True)
        env = make_live_env(args.control_hz, args.max_episode_s)

        if args.resume and os.path.exists(model_path + ".zip"):
            print("=== resuming ===", flush=True)
            model = SAC.load(model_path, env=env, device=args.device)
            if os.path.exists(buffer_path):
                model.load_replay_buffer(buffer_path)
                log.info(
                    f"resumed with {model.replay_buffer.size()} buffered transitions",
                    component="config",
                )
        else:
            model = SAC(
                "MlpPolicy",
                env,
                buffer_size=BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                learning_starts=LEARNING_STARTS,
                train_freq=TRAIN_FREQ,
                gradient_steps=GRADIENT_STEPS,
                learning_rate=LEARNING_RATE,
                device=args.device,
                verbose=0,
            )
            # model.train() is called directly below, which needs a logger that
            # learn() would normally install.
            from stable_baselines3.common.logger import configure

            model.set_logger(configure(None, ["stdout"]))
            if args.demo_episodes > 0:
                print("=== seeding from the GP pilot (live) ===", flush=True)
                demos = collect_gp_demos(env, args.demo_episodes, log=log)
                seed_replay_buffer(model, demos, log=log)

        ckpt = LiveCheckpoint(model, model_path, buffer_path, log=log)

        print("=== live SAC training ===", flush=True)
        # One learn() call per episode so a reset failure surfaces between
        # episodes and the checkpoint cadence is episode-aligned.
        steps_per_episode = int(args.control_hz * args.max_episode_s)
        for ep in range(args.episodes):
            t0 = time.monotonic()
            model.learn(
                total_timesteps=steps_per_episode,
                reset_num_timesteps=False,
                log_interval=1000,
            )
            # Spend the reset window learning. The next reset costs ~4.2 s of
            # wall clock no matter what; at 13.27 ms/step that is ~300 free
            # gradient steps, lifting the effective replay ratio well above the
            # UTD=1 the control period can afford inline.
            if model.replay_buffer.size() > model.batch_size:
                model.train(
                    gradient_steps=RESET_GRADIENT_STEPS, batch_size=model.batch_size
                )
            dt = time.monotonic() - t0
            log.info(
                f"episode {ep + 1}/{args.episodes}: gates={env.gates_passed} "
                f"steps={model.num_timesteps} wall={dt:.1f}s "
                f"buffer={model.replay_buffer.size()}",
                component="live",
            )
            if (ep + 1) % CHECKPOINT_EVERY_EPISODES == 0:
                ckpt.save(f" ep{ep + 1}")

        ckpt.save(" final")
    except Exception:
        log.fatal("live training failed")
        raise
    finally:
        if env is not None:
            env.close()
        log.close()


if __name__ == "__main__":
    main()
