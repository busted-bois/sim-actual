"""VQ2 training: BC warm start -> recurrent PPO over a shaping-annealed curriculum.

Recurrent by necessity, not preference. Velocity is not observable in VQ2 (the
one estimator ever graded against truth had 28.9 m/s velocity RMSE), and the
detector delivers a fix only ~15 times a second against a 50 Hz control loop.
The policy has to integrate its own history to know how fast it is going, which
a single-frame MLP cannot do.

Measured constraints baked into this file (stage 0, closed-loop gates cleared,
1.00 = clears the single gate):

* BC demos come from the NOMINAL plant. domain_rand=True during demo collection
  makes the targets ambiguous -- one observation maps to several correct actions
  depending on the sampled plant -- and the clone regresses to their average:
  0.00 gates, 20/20 crash, versus 1.00 and 20/20 complete on the nominal plant.
* BC backprops over TRUNCATED sequences (see BPTT_LEN). Demo episodes run
  125-2689 steps; backprop through a whole one is untrainable. Full-episode
  BPTT scored 0.00, BPTT=128 scored 0.40, and BC loss fell 0.0029 -> 0.0008.
* Domain randomization RAMPS in rather than starting at full width. The clone
  scores 1.00 on the nominal plant and 0.37 on the full randomized range --
  rate_gain (+-33%) and thrust_accel (+-10%) each take it to 0.00 on their own
  -- so PPO started inside the full range has nothing good to reinforce and
  walks away from the warm start.
* log_std is reset after the clone. Leaving PPO's initial log_std untouched
  adds full-scale exploration noise to a policy that was just taught to fly.

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

from rl.core.diagnostics import RunnerLog
from rl.environment import vq2_env
from rl.environment.vq2_env import CURRICULUM, VQ2RaceEnv

OUT_DIR = os.path.join("rl", "data", "vq2")
POLICY_PT = os.path.join(OUT_DIR, "policy_vq2.pt")
BC_PT = os.path.join(OUT_DIR, "policy_vq2_bc.pt")
SB3_ZIP = os.path.join(OUT_DIR, "ppo_vq2")

N_ENVS = 8
ADVANCE_GATE_FRAC = 0.6
FINAL_SUCCESS_RATE = 0.6
GAMMA = 0.995


BPTT_LEN = 128  # truncated backprop horizon for behaviour cloning


class DomainRandRamp(BaseCallback):
    """Widen a stage's domain randomization from nominal to full DURING training.

    Keyed to absolute model.num_timesteps so it survives _train_stage's chunked
    learn() calls: the ramp spans the whole stage, not each chunk.

    This exists because the BC clone scores 1.00 on the nominal plant and 0.37
    on the full randomized range; rate_gain and thrust_accel each take it to
    0.00 alone. Starting PPO at full width discards the warm start.
    """

    def __init__(self, start_step: int, ramp_end_step: int):
        super().__init__()
        self.start_step = int(start_step)
        self.ramp_end_step = max(int(ramp_end_step), int(start_step) + 1)

    def _set(self, dr: float) -> None:
        self.training_env.set_attr("dr_scale", float(dr))

    def _on_training_start(self) -> None:
        self._on_step()

    def _on_step(self) -> bool:
        span = self.ramp_end_step - self.start_step
        frac = (self.num_timesteps - self.start_step) / span
        self._set(float(np.clip(frac, 0.0, 1.0)))
        return True


def truncate(episodes, maxlen: int = BPTT_LEN, minlen: int = 8):
    """Split demo episodes into truncated-BPTT segments.

    Episodes are (obs, act, rew, ret); only obs/act are needed for cloning.
    Backprop through a whole episode is untrainable here -- demo lengths run
    125-2689 steps and gradient clipping crushes a 2689-step chain. Measured on
    all-stage demos, stage-0 closed-loop score: full-episode BPTT -> 0.00,
    BPTT=128 -> 0.40, with BC loss 0.0029 -> 0.0008.
    """
    out = []
    for ep in episodes:
        o, a = ep[0], ep[1]
        for i in range(0, len(o), maxlen):
            seg_o, seg_a = o[i : i + maxlen], a[i : i + maxlen]
            if len(seg_o) >= minlen:
                out.append((seg_o, seg_a))
    return out


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def discounted_returns(rewards: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Compute discounted MC returns: return_t = reward_t + gamma * return_{t+1}.

    Raises ValueError if rewards contain nonfinite values or gamma outside [0,1].
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("rewards must be finite")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0,1], got {gamma}")
    returns = np.empty_like(rewards)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns.astype(np.float32)


def _make_vec(
    stage: int,
    seed: int,
    shaping: float,
    n_envs: int = N_ENVS,
    domain_rand: bool = False,
    dr_scale: float = 1.0,
):
    cfg = CURRICULUM[stage]

    def mk(rank):
        def _f():
            return VQ2RaceEnv(
                n_gates=cfg["n_gates"],
                seed=seed + rank,
                spacing=cfg["spacing"],
                jitter=cfg["jitter"],
                shaping=shaping,
                domain_rand=domain_rand,
                dr_scale=dr_scale,
            )

        return _f

    return DummyVecEnv([mk(i) for i in range(n_envs)])


def collect_demos(
    episodes_per_stage: int = 60, seed: int = 0, log: RunnerLog | None = None
):
    episodes = []
    for stage, cfg in enumerate(CURRICULUM):
        for ep in range(episodes_per_stage):
            s = seed + 1000 * stage + ep
            env = VQ2RaceEnv(
                n_gates=cfg["n_gates"],
                seed=s,
                spacing=cfg["spacing"],
                jitter=cfg["jitter"],
                domain_rand=False,
            )
            try:
                obs, _ = env.reset(seed=s)
                term = trunc = False
                ep_obs, ep_act, ep_rew = [], [], []
                while not (term or trunc):
                    a = env.expert_action()
                    ep_obs.append(obs.copy())
                    ep_act.append(a.copy())
                    obs, r, term, trunc, _ = env.step(a)
                    ep_rew.append(float(r))
            finally:
                env.close()
            if env.gate_idx >= max(1, int(0.5 * cfg["n_gates"])):
                obs_arr = np.asarray(ep_obs, np.float32)
                act_arr = np.asarray(ep_act, np.float32)
                rew_arr = np.asarray(ep_rew, np.float32)
                ret_arr = discounted_returns(rew_arr, GAMMA)
                episodes.append((obs_arr, act_arr, rew_arr, ret_arr))
        n = sum(len(o) for o, _, _, _ in episodes)
        msg = f"stage {stage}: {len(episodes)} eps, {n} transitions"
        if log is not None:
            log.info(msg, component="demos")
        else:
            print(f"[demos] {msg}", flush=True)
    return episodes


def _actor_mean_sequence(policy, x_seq):
    feats = policy.extract_features(x_seq)
    if isinstance(feats, tuple):
        feats = feats[0]
    lstm = policy.lstm_actor
    h = torch.zeros(lstm.num_layers, 1, lstm.hidden_size, device=feats.device)
    starts = torch.zeros(feats.shape[0], device=feats.device)
    starts[0] = 1.0
    latent, _ = policy._process_sequence(feats, (h, h.clone()), starts, lstm)
    return policy.action_net(policy.mlp_extractor.forward_actor(latent))


def train_bc(
    model,
    episodes,
    epochs=30,
    lr=1e-3,
    device="cpu",
    eps_per_batch=8,
    log: RunnerLog | None = None,
):
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
                loss_sum += float(loss.detach()) * len(xs)
                nstep += len(xs)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            tot += loss_sum
        if ep % 5 == 0 or ep == epochs - 1:
            msg = f"epoch {ep + 1}/{epochs} loss={tot / max(nstep, 1):.5f}"
            if log is not None:
                log.info(msg, component="bc")
            else:
                print(f"[bc] {msg}", flush=True)

    with torch.no_grad():
        policy.log_std.fill_(-1.6)
    return model


def _critic_params(policy):
    """Collect only value-network parameters (critic path), no duplicates."""
    critic_p = []
    seen = set()
    for name, param in policy.named_parameters():
        if name.startswith("lstm_critic.") or name.startswith("value_net."):
            if id(param) not in seen:
                critic_p.append(param)
                seen.add(id(param))
    for name, param in policy.mlp_extractor.named_parameters():
        if "value_net" in name:
            if id(param) not in seen:
                critic_p.append(param)
                seen.add(id(param))
    return critic_p


def fit_critic(
    model,
    episodes,
    epochs=20,
    lr=1e-3,
    device="cpu",
    eps_per_batch=4,
    max_grad_norm=0.5,
    log: RunnerLog | None = None,
):
    policy = model.policy
    critic_p = _critic_params(policy)
    if not critic_p:
        raise RuntimeError("no critic parameters found")
    opt = torch.optim.Adam(critic_p, lr=lr)
    policy.zero_grad(set_to_none=True)

    data = [
        (torch.as_tensor(o, device=device), torch.as_tensor(r, device=device))
        for o, _, _, r in episodes
    ]
    rng = np.random.default_rng(0)

    for ep in range(epochs):
        order = rng.permutation(len(data))
        tot_v_loss, nstep = 0.0, 0
        for i in range(0, len(order), eps_per_batch):
            opt.zero_grad()
            batch = order[i : i + eps_per_batch]
            v_loss_sum = 0.0
            for j in batch:
                xs, rets = data[j]
                T = xs.shape[0]

                h = torch.zeros(
                    policy.lstm_critic.num_layers,
                    1,
                    policy.lstm_critic.hidden_size,
                    device=device,
                )
                c = h.clone()
                starts = torch.zeros(T, device=device)
                starts[0] = 1.0

                values = policy.predict_values(xs, (h, c), starts)
                v_loss = torch.nn.functional.mse_loss(values.squeeze(-1), rets)
                (v_loss / len(batch)).backward()
                v_loss_sum += float(v_loss.detach()) * T
                nstep += T
            torch.nn.utils.clip_grad_norm_(critic_p, max_grad_norm)
            opt.step()
            tot_v_loss += v_loss_sum
        if ep % 5 == 0 or ep == epochs - 1:
            msg = f"epoch {ep + 1}/{epochs} value_loss={tot_v_loss / max(nstep, 1):.5f}"
            if log is not None:
                log.info(msg, component="critic")
            else:
                print(f"[critic] {msg}", flush=True)
    return model


def evaluate(
    model,
    stage: int,
    episodes: int = 30,
    seed: int = 10_000,
    domain_rand: bool = False,
):
    cfg = CURRICULUM[stage]
    gates, done, times = [], 0, []
    for ep in range(episodes):
        env = VQ2RaceEnv(
            n_gates=cfg["n_gates"],
            seed=seed + ep,
            spacing=cfg["spacing"],
            jitter=cfg["jitter"],
            domain_rand=domain_rand,
        )
        try:
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
        finally:
            env.close()
    return {
        "gates": float(np.mean(gates)),
        "gate_frac": float(np.mean(gates) / cfg["n_gates"]),
        "success": done / episodes,
        "time": float(np.mean(times)) if times else float("nan"),
    }


def export_standalone(model, path=POLICY_PT, log: RunnerLog | None = None):
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
    msg = f"exported -> {path}"
    if log is not None:
        log.info(msg, component="export")
    else:
        print(f"[train] {msg}", flush=True)


def _make_ppo(env, seed: int, device: str = "cpu"):
    return RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=256,
        batch_size=256,
        gamma=GAMMA,
        gae_lambda=0.95,
        learning_rate=1e-4,
        ent_coef=0.003,
        clip_range=0.2,
        n_epochs=6,
        target_kl=0.03,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128,
            n_lstm_layers=1,
            share_features_extractor=False,
        ),
    )


def _stage_solved(stage: int, metrics: dict[str, float]) -> bool:
    if stage == len(CURRICULUM) - 1:
        return metrics["success"] >= FINAL_SUCCESS_RATE
    return metrics["gate_frac"] >= ADVANCE_GATE_FRAC


def _train_stage(
    model,
    stage: int,
    budget: int,
    eval_episodes: int,
    checkpoint_path: str,
    domain_rand: bool,
    log: RunnerLog | None = None,
    ramp_frac: float = 0.6,
):
    """Run one curriculum stage: pre-evaluate, PPO learn if unsolved, save checkpoint.

    Returns dict with keys: stage, pre_eval, learned, learn_steps, post_evals.
    """
    pre_m = evaluate(model, stage, episodes=eval_episodes, domain_rand=domain_rand)
    log_msg = (
        f"stage {stage} pre-eval: gates={pre_m['gates']:.2f} "
        f"frac={pre_m['gate_frac']:.2f} success={pre_m['success']:.2f}"
    )
    if log is not None:
        log.info(log_msg, component="ppo")
    else:
        print(f"[ppo] {log_msg}", flush=True)

    result = {
        "stage": stage,
        "pre_eval": pre_m,
        "learned": False,
        "learn_steps": 0,
        "post_evals": [],
    }

    if _stage_solved(stage, pre_m):
        if stage == len(CURRICULUM) - 1:
            criterion = f"success={pre_m['success']:.2f} >= {FINAL_SUCCESS_RATE}"
        else:
            criterion = f"frac={pre_m['gate_frac']:.2f} >= {ADVANCE_GATE_FRAC}"
        skip_msg = f"stage {stage} already solved ({criterion}), skipping PPO"
        if log is not None:
            log.info(skip_msg, component="ppo")
        else:
            print(f"[ppo] {skip_msg}", flush=True)
        model.save(checkpoint_path)
        if log is not None:
            log.info(f"checkpoint -> {checkpoint_path}", component="checkpoint")
        return result

    chunk = max(budget // 4, 2048)
    steps_before = model.num_timesteps
    # Ramp randomization across the WHOLE stage, keyed to absolute timesteps so
    # the chunked learn() calls below do not restart it each chunk.
    ramp = (
        DomainRandRamp(steps_before, steps_before + int(budget * ramp_frac))
        if domain_rand
        else None
    )
    spent = 0
    while spent < budget:
        remaining = budget - spent
        request = min(chunk, remaining)
        model.learn(
            total_timesteps=request,
            reset_num_timesteps=False,
            progress_bar=False,
            callback=ramp,
        )
        actual_delta = model.num_timesteps - steps_before
        spent = actual_delta
        m = evaluate(model, stage, episodes=eval_episodes, domain_rand=domain_rand)
        result["post_evals"].append(m)
        train_logger = getattr(model, "_logger", None)
        train_values = train_logger.name_to_value if train_logger is not None else {}
        approx_kl = float(train_values.get("train/approx_kl", float("nan")))
        value_loss = float(train_values.get("train/value_loss", float("nan")))
        policy_loss = float(
            train_values.get("train/policy_gradient_loss", float("nan"))
        )
        chunk_msg = (
            f"stage {stage} {spent}/{budget} "
            f"gates={m['gates']:.2f} frac={m['gate_frac']:.2f} "
            f"success={m['success']:.2f} approx_kl={approx_kl:.5f} "
            f"value_loss={value_loss:.5f} policy_loss={policy_loss:.5f}"
        )
        if log is not None:
            log.info(chunk_msg, component="ppo")
        else:
            print(f"[ppo] {chunk_msg}", flush=True)
        if _stage_solved(stage, m):
            solved_msg = f"stage {stage} solved, advancing early"
            if log is not None:
                log.info(solved_msg, component="ppo")
            else:
                print(f"[ppo] {solved_msg}", flush=True)
            break

    result["learned"] = True
    result["learn_steps"] = model.num_timesteps - steps_before
    model.save(checkpoint_path)
    if log is not None:
        log.info(f"checkpoint -> {checkpoint_path}", component="checkpoint")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-stage", type=_positive_int, default=400_000)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-episodes", type=_positive_int, default=60)
    ap.add_argument("--bc-epochs", type=_positive_int, default=30)
    ap.add_argument("--critic-epochs", type=_positive_int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    if args.quick:
        print(
            "[warn] --quick is a bounded smoke test, NOT a convergence proof",
            flush=True,
        )
        args.steps_per_stage = 2_048
        args.demo_episodes = 2
        args.bc_epochs = 2
        args.critic_epochs = 2

    out_dir = args.out_dir
    policy_path = os.path.join(out_dir, os.path.basename(POLICY_PT))
    bc_path = os.path.join(out_dir, os.path.basename(BC_PT))
    sb3_base = os.path.join(out_dir, os.path.basename(SB3_ZIP))
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log = RunnerLog(tag="vq2-train", log_dir=os.path.join("logs", "vq2-training"))

    vec_env = None
    try:
        log.info(
            f"config: seed={args.seed} device={args.device} "
            f"steps_per_stage={args.steps_per_stage} "
            f"demo_episodes={args.demo_episodes} bc_epochs={args.bc_epochs} "
            f"critic_epochs={args.critic_epochs} gamma={GAMMA} "
            f"ppo_lr=1e-4 target_kl=0.03 advance_gate={ADVANCE_GATE_FRAC} "
            f"out_dir={out_dir}",
            component="config",
        )

        print("=== 1/4 expert demonstrations ===", flush=True)
        episodes = collect_demos(args.demo_episodes, seed=args.seed, log=log)
        if not episodes:
            raise RuntimeError("[demos] expert produced no usable episodes")
        n_trans = sum(len(o) for o, _, _, _ in episodes)
        log.info(
            f"collected {len(episodes)} episodes, {n_trans} transitions",
            component="demos",
        )

        print("=== 2/4 behaviour cloning ===", flush=True)
        log.info("starting behaviour cloning", component="bc")
        vec_env = _make_vec(0, args.seed, shaping=1.0, domain_rand=False)
        model = _make_ppo(vec_env, seed=args.seed, device=args.device)
        train_bc(
            model, episodes, epochs=args.bc_epochs, lr=1e-3, device=args.device, log=log
        )

        bc_m = evaluate(model, 0, episodes=10)
        log.info(
            f"BC eval stage-0: gates={bc_m['gates']:.2f} success={bc_m['success']:.2f}",
            component="bc",
        )

        print("=== 3/4 recurrent critic fitting ===", flush=True)
        log.info("fitting recurrent critic on expert returns", component="critic")
        model = fit_critic(
            model,
            episodes,
            epochs=args.critic_epochs,
            lr=1e-3,
            device=args.device,
            log=log,
        )
        torch.save({"state_dict": model.policy.state_dict()}, bc_path)
        log.info(f"BC+critic checkpoint -> {bc_path}", component="bc")

        print("=== 4/4 curriculum PPO ===", flush=True)
        log.info("starting curriculum PPO", component="ppo")
        n_stages = len(CURRICULUM)
        eval_eps = 10 if args.quick else 20
        for stage in range(n_stages):
            shaping = max(0.0, 1.0 - stage / max(n_stages - 1, 1))
            dr = False if stage == 0 else True
            if vec_env is not None:
                vec_env.close()
            vec_env = _make_vec(
                stage, args.seed + stage, shaping, domain_rand=dr, dr_scale=0.0
            )
            model.set_env(vec_env)

            log.info(
                f"stage {stage}/{n_stages - 1}: n_gates={CURRICULUM[stage]['n_gates']} "
                f"shaping={shaping:.2f} domain_rand={dr}",
                component="ppo",
            )
            _train_stage(
                model,
                stage,
                budget=args.steps_per_stage,
                eval_episodes=eval_eps,
                checkpoint_path=f"{sb3_base}_s{stage}",
                domain_rand=dr,
                log=log,
            )

        model.save(sb3_base)
        log.info(f"final PPO checkpoint -> {sb3_base}", component="checkpoint")
        export_standalone(model, path=policy_path, log=log)

        final = evaluate(model, n_stages - 1, episodes=10 if args.quick else 50)
        robust = evaluate(
            model,
            n_stages - 1,
            episodes=10 if args.quick else 50,
            seed=20_000,
            domain_rand=True,
        )
        log.info(
            f"final 17-gate: gates={final['gates']:.2f} "
            f"success={final['success']:.2f} time={final['time']:.1f}s",
            component="final",
        )
        log.info(
            f"final randomized: gates={robust['gates']:.2f} "
            f"success={robust['success']:.2f} time={robust['time']:.1f}s",
            component="final",
        )

    except Exception:
        log.fatal("training failed")
        raise
    finally:
        if vec_env is not None:
            vec_env.close()
        log.close()


if __name__ == "__main__":
    main()
