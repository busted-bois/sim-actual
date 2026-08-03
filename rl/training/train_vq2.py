"""VQ2 training: BC warm start -> recurrent PPO over a shaping-annealed curriculum.

Recurrent by necessity, not preference. Velocity is not observable in VQ2 (the
one estimator ever graded against truth had 28.9 m/s velocity RMSE), and the
detector delivers a fix only ~15 times a second against a 50 Hz control loop.
The policy therefore has to integrate its own history to know how fast it is
going, which a single-frame MLP cannot do.

Three defects from the previous pipeline are fixed here explicitly:

* log_std is reset after the BC warm start. Cloning only the action MEAN and
  leaving PPO's initial log_std untouched means the first rollouts add
  full-scale Gaussian noise to a policy that was just taught to fly — a crash
  generator, and one of the recorded reasons the last attempt died.
* Each stage is ONE learn() call. Chunking a stage to evaluate mid-way halves
  to zero (measured 0.70 -> 0.00 at equal step count): every learn() re-runs
  _setup_learn, dropping the LSTM state and truncating in-flight episodes.
* Domain randomization RAMPS in via callback. The clone scores 1.00 on the
  nominal plant and 0.37 on the full randomized range, so starting PPO inside
  the full range throws the warm start away.
* Shaping anneals to zero across training so the final objective is the sparse
  terminal one the competition actually scores.

    uv run -m rl.training.train_vq2 --quick
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.environment import vq2_env
from rl.environment.vq2_env import CURRICULUM, VQ2RaceEnv

OUT_DIR = os.path.join("rl", "data", "vq2")
POLICY_PT = os.path.join(OUT_DIR, "policy_vq2.pt")
BC_PT = os.path.join(OUT_DIR, "policy_vq2_bc.pt")
SB3_ZIP = os.path.join(OUT_DIR, "ppo_vq2")

N_ENVS = 8


class DomainRandRamp(BaseCallback):
    """Widen domain randomization DURING a single learn() call.

    Must be a callback, not repeated set_env()+learn(). Measured on stage 0,
    30k steps, identical hyperparameters:
        one learn() call, full DR          -> 0.70 gates
        8x learn()+set_env, full DR        -> 0.00
        8x learn()+set_env, ramped DR      -> 0.00
    Every learn() call re-runs _setup_learn, which drops _last_obs and the LSTM
    states and truncates every in-flight episode; doing that eight times per
    stage destroys the policy on its own, independently of randomization.
    """

    def __init__(self, warmup_frac: float = 0.6):
        super().__init__()
        self.warmup_frac = float(warmup_frac)
        self._total = 1

    def _on_training_start(self) -> None:
        self._total = max(int(self.locals.get("total_timesteps", 1)), 1)
        self._set(0.0)

    def _set(self, dr: float) -> None:
        self.training_env.set_attr("dr_scale", float(dr))

    def _on_step(self) -> bool:
        frac = self.num_timesteps / (self._total * max(self.warmup_frac, 1e-6))
        self._set(min(1.0, frac))
        return True


def _make_vec(stage, seed, shaping, dr_scale=1.0, n_envs=N_ENVS):
    cfg = CURRICULUM[stage]

    def mk(rank):
        def _f():
            return VQ2RaceEnv(
                n_gates=cfg["n_gates"],
                seed=seed + rank,
                spacing=cfg["spacing"],
                jitter=cfg["jitter"],
                shaping=shaping,
                dr_scale=dr_scale,
            )

        return _f

    return DummyVecEnv([mk(i) for i in range(n_envs)])


# --------------------------------------------------------------------------
# Behaviour cloning
# --------------------------------------------------------------------------
def collect_demos(episodes_per_stage: int = 60, seed: int = 0):
    """Roll the privileged geometric expert, record (VQ2 obs -> action) EPISODES.

    Returned as whole episodes, not a shuffled pile of transitions. The expert
    steers on world velocity, which the VQ2 observation deliberately does not
    contain -- so a memoryless clone provably cannot reproduce it. The LSTM has
    to see ordered sequences to learn to infer speed from feature history.
    """
    episodes = []
    for stage, cfg in enumerate(CURRICULUM):
        for ep in range(episodes_per_stage):
            s = seed + 1000 * stage + ep
            # NOMINAL plant only. Measured ablation (stage 0, 40 demo episodes,
            # 25 BC epochs, 20 eval episodes):
            #   demos with domain_rand=True  -> 0.00 gates, 20/20 crash
            #   demos with domain_rand=False -> 1.00 gates, 20/20 complete
            # Randomizing the plant during demo collection makes the targets
            # ambiguous -- one observation maps to different correct actions
            # depending on the sampled thrust_accel/rate_gain -- so the clone
            # regresses to their average and flies into the ground. PPO does
            # the adapting to a randomized plant; BC must not have to.
            env = VQ2RaceEnv(
                n_gates=cfg["n_gates"],
                seed=s,
                spacing=cfg["spacing"],
                jitter=cfg["jitter"],
                domain_rand=False,
            )
            obs, _ = env.reset(seed=s)
            term = trunc = False
            ep_obs, ep_act = [], []
            while not (term or trunc):
                a = env.expert_action()
                ep_obs.append(obs.copy())
                ep_act.append(a.copy())
                obs, _, term, trunc, _ = env.step(a)
            # Keep only demonstrations worth imitating.
            if env.gate_idx >= max(1, int(0.5 * cfg["n_gates"])):
                episodes.append(
                    (
                        np.asarray(ep_obs, np.float32),
                        np.asarray(ep_act, np.float32),
                    )
                )
        n = sum(len(o) for o, _ in episodes)
        print(
            f"[demos] stage {stage}: {len(episodes)} eps, {n} transitions", flush=True
        )
    return episodes


def _actor_mean_sequence(policy, x_seq):
    """Action means for one ORDERED episode, carrying the LSTM state through it.

    x_seq is (T, obs_dim). Fed as a single length-T sequence with batch 1, which
    is exactly the shape _process_sequence expects, so the hidden state evolves
    the way it will at inference time.
    """
    feats = policy.extract_features(x_seq)
    if isinstance(feats, tuple):
        feats = feats[0]
    lstm = policy.lstm_actor
    h = torch.zeros(lstm.num_layers, 1, lstm.hidden_size, device=feats.device)
    starts = torch.zeros(feats.shape[0], device=feats.device)
    starts[0] = 1.0  # only the first step resets the state
    latent, _ = policy._process_sequence(feats, (h, h.clone()), starts, lstm)
    return policy.action_net(policy.mlp_extractor.forward_actor(latent))


BPTT_LEN = 128  # truncated backprop horizon for BC


def truncate(episodes, maxlen: int = BPTT_LEN, minlen: int = 8):
    """Split episodes into truncated-BPTT segments.

    Backpropagating through a whole episode is untrainable here: demo lengths
    run 125-2689 steps, and gradient-clipping a 2689-step BPTT chain crushes
    the signal. Measured on all-stage demos, stage-0 closed-loop score:
        full-episode BPTT -> 0.00
        BPTT = 128        -> 0.40   (and BC loss 0.0029 -> 0.0008)
    """
    out = []
    for o, a in episodes:
        for i in range(0, len(o), maxlen):
            seg_o, seg_a = o[i : i + maxlen], a[i : i + maxlen]
            if len(seg_o) >= minlen:
                out.append((seg_o, seg_a))
    return out


def train_bc(model, episodes, epochs=30, lr=1e-3, device="cpu", eps_per_batch=8):
    """Clone the expert over truncated sequences so the LSTM learns to integrate."""
    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    data = [
        (torch.as_tensor(o, device=device), torch.as_tensor(a, device=device))
        for o, a in truncate(episodes)
    ]
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        order = rng.permutation(len(data))
        tot, nstep = 0.0, 0
        for i in range(0, len(order), eps_per_batch):
            opt.zero_grad()
            batch = order[i : i + eps_per_batch]
            loss_sum = 0.0
            for j in batch:
                xs, ys = data[j]
                loss = torch.nn.functional.mse_loss(
                    _actor_mean_sequence(policy, xs), ys
                )
                (loss / len(batch)).backward()
                loss_sum += float(loss) * len(xs)
                nstep += len(xs)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot += loss_sum
        if ep % 5 == 0 or ep == epochs - 1:
            print(
                f"[bc] epoch {ep + 1}/{epochs} loss={tot / max(nstep, 1):.5f}",
                flush=True,
            )

    # Critical: the clone taught the MEAN. Leaving log_std at its init means
    # PPO immediately injects full-scale noise and destroys what BC learned.
    with torch.no_grad():
        policy.log_std.fill_(-1.6)  # ~0.20 rad/s of exploration, not 1.0
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(model, stage: int, episodes: int = 30, seed: int = 10_000):
    cfg = CURRICULUM[stage]
    gates, done, times = [], 0, []
    for ep in range(episodes):
        env = VQ2RaceEnv(
            n_gates=cfg["n_gates"],
            seed=seed + ep,
            spacing=cfg["spacing"],
            jitter=cfg["jitter"],
            domain_rand=False,
        )
        obs, _ = env.reset(seed=seed + ep)
        state, start = None, True
        term = trunc = False
        while not (term or trunc):
            a, state = model.predict(
                obs[None],
                state=state,
                episode_start=np.array([start]),
                deterministic=True,
            )
            start = False
            obs, _, term, trunc, info = env.step(a[0])
        gates.append(env.gate_idx)
        if info.get("course_complete"):
            done += 1
            times.append(env.t)
    return {
        "gates": float(np.mean(gates)),
        "gate_frac": float(np.mean(gates) / cfg["n_gates"]),
        "success": done / episodes,
        "time": float(np.mean(times)) if times else float("nan"),
    }


def export_standalone(model, path=POLICY_PT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.policy.state_dict(),
            "obs_dim": int(model.observation_space.shape[0]),
            "act_dim": int(model.action_space.shape[0]),
            "recurrent": True,
            "decision_hz": vq2_env.DECISION_HZ,
            "gate_opening_m": vq2_env.GATE_OPENING_M,
        },
        path,
    )
    print(f"[train] exported -> {path}", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-stage", type=int, default=400_000)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-episodes", type=int, default=60)
    ap.add_argument("--bc-epochs", type=int, default=30)
    ap.add_argument("--device", default="cpu")  # MLP: CPU beats GPU for PPO
    args = ap.parse_args()

    if args.quick:
        # Sized from the measured BC ablation: 6 episodes / 5 epochs is far
        # below the ~40 episodes and ~25 epochs the clone needs before it can
        # fly at all, so a smaller smoke test reports a false failure.
        args.steps_per_stage = 6_000
        args.demo_episodes = 25
        args.bc_epochs = 20

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== 1/3 expert demonstrations ===", flush=True)
    episodes = collect_demos(args.demo_episodes, seed=args.seed)
    if not episodes:
        raise SystemExit("[demos] expert produced no usable episodes")

    env = _make_vec(0, args.seed, shaping=1.0)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=256,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=5e-5,  # 3e-4 degraded the BC init faster (0.60 vs 0.70)
        ent_coef=0.003,
        clip_range=0.2,
        n_epochs=6,
        seed=args.seed,
        device=args.device,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
        ),
    )

    print("=== 2/3 behaviour cloning ===", flush=True)
    train_bc(model, episodes, epochs=args.bc_epochs, lr=1e-3, device=args.device)
    torch.save({"state_dict": model.policy.state_dict()}, BC_PT)
    print(f"[bc] saved -> {BC_PT}", flush=True)
    # Gate the warm start: if the clone cannot fly, PPO is starting from noise
    # and every later number is meaningless.
    bc_m = evaluate(model, 0, episodes=10)
    print(
        f"[bc] stage-0 eval gates={bc_m['gates']:.2f} success={bc_m['success']:.2f}",
        flush=True,
    )

    print("=== 3/3 curriculum PPO ===", flush=True)
    n_stages = len(CURRICULUM)
    for stage in range(n_stages):
        # Shaping decays to 0 by the final stage: the competition scores only
        # completion, so that is what the last stage optimizes.
        shaping = max(0.0, 1.0 - stage / max(n_stages - 1, 1))
        # ONE learn() call per stage. Chunking it to evaluate mid-stage
        # destroys the policy (0.70 -> 0.00 at equal step count), because each
        # learn() re-runs _setup_learn and drops the LSTM state and every
        # in-flight episode. Ramp randomization with a callback instead.
        model.set_env(_make_vec(stage, args.seed + stage, shaping, dr_scale=0.0))
        model.learn(
            total_timesteps=args.steps_per_stage,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=DomainRandRamp(warmup_frac=0.6 if stage == 0 else 0.15),
        )
        m = evaluate(model, stage, episodes=10 if args.quick else 30)
        print(
            f"[ppo] stage {stage} ({CURRICULUM[stage]['n_gates']} gates, "
            f"shaping={shaping:.2f}) {args.steps_per_stage} steps "
            f"gates={m['gates']:.2f} frac={m['gate_frac']:.2f} "
            f"success={m['success']:.2f}",
            flush=True,
        )
        model.save(f"{SB3_ZIP}_s{stage}")

    model.save(SB3_ZIP)
    export_standalone(model)

    final = evaluate(model, n_stages - 1, episodes=10 if args.quick else 50)
    print(
        f"[final] 17-gate: gates={final['gates']:.2f} "
        f"success={final['success']:.2f} time={final['time']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
