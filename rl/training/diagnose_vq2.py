"""Why does the VQ2 policy fail where the expert succeeds? Instrumented answer.

The expert clears 25/25 episodes at every curriculum stage, including all 17
gates in ~20.6 s. The cloned policy clears a fraction of one. Regression loss
looks fine (0.0008) while closed-loop behaviour does not, which is the classic
signature of covariate shift -- but "classic signature" is not evidence, so
this module measures it directly.

The decisive comparison is action error on EXPERT-visited states versus on
POLICY-visited states:

  * both low   -> the policy is fine and something downstream is wrong
  * both high  -> the clone never learned the mapping; more/better BC
  * expert low, policy high -> covariate shift. The policy is accurate on the
    demo distribution and wrong on the states its own mistakes lead it into,
    which more BC on the same data cannot fix. That is the DAgger case.

Everything is logged per-step so a failure can be located, not guessed at:
first divergence step, the observation at that step, per-channel action error,
where in the course it died and why.

    uv run -m rl.training.diagnose_vq2 --policy rl/data/vq2/ppo_vq2
    uv run -m rl.training.diagnose_vq2 --stage 0 --episodes 20 --dump-csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import numpy as np

from rl.core import vq2_observation as vo
from rl.environment.vq2_env import CURRICULUM, VQ2RaceEnv

ACTION_NAMES = ("roll", "pitch", "yaw", "thrust")
DIVERGE_THRESH = 0.25  # normalized action units


def _mk(stage: int, seed: int, dr_scale: float = 0.0):
    cfg = CURRICULUM[stage]
    return VQ2RaceEnv(
        n_gates=cfg["n_gates"],
        seed=seed,
        spacing=cfg["spacing"],
        jitter=cfg["jitter"],
        domain_rand=dr_scale > 0.0,
        dr_scale=dr_scale,
    )


def _policy_action(model, obs, state, first):
    a, state = model.predict(
        obs[None], state=state, episode_start=np.array([first]), deterministic=True
    )
    return np.asarray(a[0], dtype=float), state


def roll_expert(stage, seed, model=None, dr_scale=0.0):
    """Fly the EXPERT. If a model is given, also record what it WOULD have done.

    This is the in-distribution probe: the policy never steers, so every state
    is one the demonstrations actually cover.
    """
    env = _mk(stage, seed, dr_scale)
    try:
        obs, _ = env.reset(seed=seed)
        state, first = None, True
        rows = []
        term = trunc = False
        while not (term or trunc):
            exp_a = env.expert_action().astype(float)
            pol_a = None
            if model is not None:
                pol_a, state = _policy_action(model, obs, state, first)
                first = False
            rows.append((obs.copy(), exp_a, pol_a, env.gate_idx))
            obs, _, term, trunc, info = env.step(exp_a)
        return rows, env.gate_idx, info
    finally:
        env.close()


def roll_policy(stage, seed, model, dr_scale=0.0):
    """Fly the POLICY, recording what the expert would have done at each state.

    This is the on-policy probe: states are reached by the policy's own
    choices, so it is exactly the distribution BC does not cover.
    """
    env = _mk(stage, seed, dr_scale)
    try:
        obs, _ = env.reset(seed=seed)
        state, first = None, True
        rows = []
        term = trunc = False
        while not (term or trunc):
            pol_a, state = _policy_action(model, obs, state, first)
            first = False
            exp_a = env.expert_action().astype(float)
            rows.append((obs.copy(), exp_a, pol_a, env.gate_idx))
            obs, _, term, trunc, info = env.step(pol_a)
        return rows, env.gate_idx, info
    finally:
        env.close()


def _err(rows):
    """Per-channel |policy - expert| over rows that carry both."""
    d = np.array([np.abs(p - e) for _, e, p, _ in rows if p is not None])
    return d if len(d) else np.zeros((0, 4))


def _first_divergence(rows, thresh=DIVERGE_THRESH):
    for i, (obs, e, p, gi) in enumerate(rows):
        if p is None:
            continue
        if float(np.abs(p - e).max()) > thresh:
            return i, obs, e, p, gi
    return None, None, None, None, None


def diagnose(model, stage=0, episodes=20, dr_scale=0.0, dump_csv=None, verbose=True):
    cfg = CURRICULUM[stage]
    exp_gates, pol_gates = [], []
    exp_reasons, pol_reasons = Counter(), Counter()
    exp_err, pol_err = [], []
    div_steps, div_fracs = [], []
    dump_rows = []

    for ep in range(episodes):
        seed = 10_000 + ep

        e_rows, e_g, e_info = roll_expert(stage, seed, model, dr_scale)
        exp_gates.append(e_g)
        exp_reasons[
            e_info.get("crash")
            or ("complete" if e_info.get("course_complete") else "timeout")
        ] += 1
        exp_err.append(_err(e_rows))

        p_rows, p_g, p_info = roll_policy(stage, seed, model, dr_scale)
        pol_gates.append(p_g)
        pol_reasons[
            p_info.get("crash")
            or ("complete" if p_info.get("course_complete") else "timeout")
        ] += 1
        pol_err.append(_err(p_rows))

        i, obs, e_a, p_a, gi = _first_divergence(p_rows)
        if i is not None:
            div_steps.append(i)
            div_fracs.append(i / max(len(p_rows), 1))
            if verbose and ep < 3:
                L = vo.OBS_LAYOUT
                print(
                    f"  ep{ep} first divergence @step {i}/{len(p_rows)} "
                    f"(gate {gi}): expert={np.round(e_a, 3)} policy={np.round(p_a, 3)}"
                )
                print(
                    f"        obs: detected={obs[L['detected']][0]:.0f} "
                    f"stale={obs[L['staleness']][0]:.2f} "
                    f"conf={obs[L['conf']][0]:.2f} "
                    f"logrange={obs[L['log_range']][0]:+.2f} "
                    f"dir={np.round(obs[L['gate_dir_body']], 2)} "
                    f"grav_z={obs[L['gravity_body']][2]:+.2f}"
                )
        if dump_csv:
            for i2, (o, e_a2, p_a2, gi2) in enumerate(p_rows):
                if p_a2 is None:
                    continue
                dump_rows.append(
                    [
                        ep,
                        i2,
                        gi2,
                        *np.round(e_a2, 4),
                        *np.round(p_a2, 4),
                        *np.round(o, 4),
                    ]
                )

    E = np.concatenate(exp_err) if exp_err else np.zeros((0, 4))
    P = np.concatenate(pol_err) if pol_err else np.zeros((0, 4))

    report = {
        "stage": stage,
        "n_gates": cfg["n_gates"],
        "episodes": episodes,
        "dr_scale": dr_scale,
        "expert_gates": float(np.mean(exp_gates)),
        "policy_gates": float(np.mean(pol_gates)),
        "expert_reasons": dict(exp_reasons),
        "policy_reasons": dict(pol_reasons),
        "err_on_expert_states": E.mean(axis=0).tolist() if len(E) else [],
        "err_on_policy_states": P.mean(axis=0).tolist() if len(P) else [],
        "shift_ratio": (
            float(P.mean() / E.mean())
            if len(E) and len(P) and E.mean() > 1e-9
            else float("nan")
        ),
        "mean_divergence_step": float(np.mean(div_steps))
        if div_steps
        else float("nan"),
        "mean_divergence_frac": float(np.mean(div_fracs))
        if div_fracs
        else float("nan"),
    }

    if dump_csv:
        os.makedirs(os.path.dirname(dump_csv) or ".", exist_ok=True)
        with open(dump_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["ep", "step", "gate"]
                + [f"exp_{n}" for n in ACTION_NAMES]
                + [f"pol_{n}" for n in ACTION_NAMES]
                + [
                    f"{k}{i}"
                    for k, sl in vo.OBS_LAYOUT.items()
                    for i in range(sl.stop - sl.start)
                ]
            )
            w.writerows(dump_rows)
        report["dump_csv"] = dump_csv
    return report


def print_report(r):
    print(f"\n{'=' * 66}")
    print(
        f"stage {r['stage']} ({r['n_gates']} gates), {r['episodes']} eps, dr={r['dr_scale']}"
    )
    print("=" * 66)
    print(f"  expert gates : {r['expert_gates']:.2f}   {r['expert_reasons']}")
    print(f"  policy gates : {r['policy_gates']:.2f}   {r['policy_reasons']}")
    if r["err_on_expert_states"]:
        print("\n  mean |policy - expert| action error, per channel:")
        print(
            f"    {'channel':>8} {'on EXPERT states':>18} {'on POLICY states':>18}  {'ratio':>7}"
        )
        for i, n in enumerate(ACTION_NAMES):
            e = r["err_on_expert_states"][i]
            p = r["err_on_policy_states"][i]
            print(f"    {n:>8} {e:>18.4f} {p:>18.4f}  {p / max(e, 1e-9):>7.2f}x")
        print(
            f"\n  overall shift ratio (policy-states / expert-states): {r['shift_ratio']:.2f}x"
        )
        if r["shift_ratio"] > 2.0:
            print("  -> COVARIATE SHIFT: accurate on demo states, wrong on its own.")
            print("     More BC on the same data cannot fix this; DAgger can.")
        elif max(r["err_on_expert_states"]) > 0.15:
            print("  -> UNDERFIT: wrong even on demo states. Needs more/better BC.")
        else:
            print("  -> Actions track the expert closely on both distributions.")
    print(
        f"\n  first divergence at step {r['mean_divergence_step']:.0f} "
        f"({100 * r['mean_divergence_frac']:.0f}% into the episode)"
    )
    if r.get("dump_csv"):
        print(f"  per-step dump -> {r['dump_csv']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join("rl", "data", "vq2", "ppo_vq2"))
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--all-stages", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--dr-scale", type=float, default=0.0)
    ap.add_argument("--dump-csv", action="store_true")
    args = ap.parse_args()

    from sb3_contrib import RecurrentPPO

    path = args.policy
    if not (os.path.exists(path) or os.path.exists(path + ".zip")):
        raise SystemExit(
            f"[diagnose] no policy at {path} -- run `make train-vq2` first"
        )
    model = RecurrentPPO.load(path, device="cpu")
    print(f"[diagnose] loaded {path}")

    stages = range(len(CURRICULUM)) if args.all_stages else [args.stage]
    for st in stages:
        dump = (
            os.path.join("rl", "data", "vq2", f"diagnose_s{st}.csv")
            if args.dump_csv
            else None
        )
        print_report(
            diagnose(
                model,
                stage=st,
                episodes=args.episodes,
                dr_scale=args.dr_scale,
                dump_csv=dump,
            )
        )


if __name__ == "__main__":
    main()
