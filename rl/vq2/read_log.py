"""Read the VQ2 training episode log and answer two questions:
  1) Is the drone passing gates? (gates>=1 rate, best gate count, closest approach)
  2) Is the reward function working? (mean per-term reward contribution, why
     episodes end)

    uv run -m rl.vq2.read_log                 # summary + last 15 episodes
    uv run -m rl.vq2.read_log --tail 40       # more rows
    uv run -m rl.vq2.read_log --all           # every episode
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

DEFAULT = os.path.join("rl", "data", "vq2", "episodes.jsonl")


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if not n:
        print("no episodes logged yet (train first: make rl2-train ...)")
        return
    gates = [r.get("gates") or 0 for r in rows]
    passed1 = sum(g >= 1 for g in gates)
    reasons = Counter(r.get("reason") for r in rows)
    ranges = [r["min_range"] for r in rows if r.get("min_range") is not None]

    print(f"episodes: {n}")
    print(f"gate 1+ cleared: {passed1}/{n} ({100*passed1/n:.0f}%)   "
          f"best gates in one ep: {max(gates)}   mean gates: {sum(gates)/n:.2f}")
    if ranges:
        print(f"closest a gate ever got: {min(ranges):.1f}m   "
              f"mean closest-per-ep: {sum(ranges)/len(ranges):.1f}m")
    print("end reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # mean per-term reward contribution over the episodes -> is shaping working?
    keys = ["progress", "fwd", "smooth", "osc", "rate", "time",
            "soft_col", "gate_bonus", "terminal"]
    sums = {k: 0.0 for k in keys}
    for r in rows:
        for k, v in (r.get("parts") or {}).items():
            sums[k] = sums.get(k, 0.0) + v
    print("\nmean reward contribution per episode (does each term fire?):")
    for k in keys:
        print(f"  {k:>10}: {sums[k]/n:+7.2f}")
    print(f"  {'TOTAL':>10}: {sum(r.get('R', 0) for r in rows)/n:+7.2f}")


def show_rows(rows: list[dict], tail: int) -> None:
    rows = rows if tail <= 0 else rows[-tail:]
    print(f"\n{'ep':>5} {'tstep':>8} {'gates':>5} {'reason':>14} {'R':>8} "
          f"{'len':>5} {'closest':>8} {'hard':>4} {'soft':>4}")
    for r in rows:
        mr = f"{r['min_range']:.1f}" if r.get("min_range") is not None else "  --"
        print(f"{r.get('ep', 0):>5} {r.get('t', 0):>8} {r.get('gates', 0):>5} "
              f"{str(r.get('reason', '')):>14} {r.get('R', 0):>8.1f} "
              f"{r.get('len', 0):>5} {mr:>8} "
              f"{r.get('hard_col', 0):>4} {r.get('soft_col', 0):>4}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT)
    ap.add_argument("--tail", type=int, default=15)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rows = load(args.path)
    summarize(rows)
    show_rows(rows, 0 if args.all else args.tail)
