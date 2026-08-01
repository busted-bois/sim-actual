"""Score flown runs of the GP pilot, aggregated across attempts.

`bl_replay.py` does this for the corridor detector and it works: two
plausible-looking detector improvements were falsified offline, without a
flight. The control side had no equivalent, so a pilot change was judged by
flying it and reading an integer gate count -- which is how 6a10771 (corner
clearance + collision float) shipped on sound reasoning and had to be reverted
after a flight proved it worse.

    uv run scripts/gp_score.py
    uv run scripts/gp_score.py --json before.json
    uv run scripts/gp_score.py --baseline before.json
    uv run scripts/gp_score.py --run 20260731_181602

THE POINT IS THE SPLIT INTO TWO SECTIONS.

OUTCOME is gates reached, and it prints NO verdict. Gate counts on this course
run 0-5 with high variance off a handful of attempts per run; better/WORSE on
n=4 is noise wearing a verdict's clothes. It prints `1/4`, never `25 %`, so the
sample size cannot be forgotten.

PER-TICK is where an A/B is actually decided, at n≈10³-10⁴ samples per run.
"ran out of roll authority for 12 % of the flight" or "51 ticks inverted" is
both actionable and statistically real at that count, while "reached gate 2"
is neither. Saturation and attitude metrics need no extra instrumentation --
they fall out of columns the pilot already writes.

Runs are grouped by (sha, diff_sha, env, track) and NEVER pooled across groups
without --force: every recorded run so far flew `dirty: true` with no record of
which env knobs were set, so runs that looked identical were not comparable at
all. That is what run_meta's config capture fixed, and this refuses to pretend
otherwise for runs that predate it.

This is a leaf tool: it imports the pilot's schema, nothing imports it.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.gp_pilot import (  # noqa: E402
    LOG_COLUMNS,
    MAX_DESCENT_RATE_MPS,
    PITCH_WIRE_MAX_DEG,
    ROLL_WIRE_MAX_DEG,
)

_SIDECARS = os.path.join("runs", "videos", "vision_*.json")
_LOGDIR = os.path.join("rl", "data")

# A command within this of its clamp is saturated for scoring purposes. The
# clamp is applied with np.clip, so a saturated tick sits exactly on it;
# the epsilon only absorbs float formatting in the CSV.
_SAT_EPS = 1e-3
# Below this the aircraft is closer to sideways than upright -- see the upset
# discussion in gp_pilot's tilt-compensation block. cos(25)*cos(18) = 0.862 at
# both wire clamps at once, so nothing the guidance can command comes near it.
_UPSET_COS = 0.5


def _f(row, key, default=float("nan")):
    """Column as float. Absent/blank/unparseable all read as `default` --
    logs are written by a flight loop that must never stop to be correct."""
    try:
        v = row.get(key, "")
        return float(v) if v != "" else default
    except (TypeError, ValueError):
        return default


def _pct(xs, q):
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return float("nan")
    i = min(int(q * (len(xs) - 1) + 0.5), len(xs) - 1)
    return xs[i]


def _frac(xs, pred):
    xs = list(xs)
    return (sum(1 for x in xs if pred(x)) / len(xs)) if xs else float("nan")


def config_key(meta):
    """What must match for two runs to be poolable."""
    git = meta.get("git") or {}
    cfg = meta.get("config") or {}
    track = meta.get("track") or {}
    env = cfg.get("env")
    # A schema-1 sidecar has no diff_sha KEY at all, which is not the same
    # claim as a clean tree -- and every one of those runs flew dirty. Keeping
    # the two distinct stops a legacy run being pooled with a genuinely clean
    # one, and stops the report asserting "clean" about a tree it never saw.
    if "diff_sha" not in git:
        diff = "?"
    else:
        diff = git["diff_sha"] or "clean"
    return (
        git.get("sha"),
        diff,
        json.dumps(env, sort_keys=True) if env is not None else None,
        track.get("n_gates"),
    )


def describe_key(key):
    sha, diff, env, n_gates = key
    env_txt = "?" if env is None else (env if env != "{}" else "{}")
    if env_txt != "?" and len(env_txt) > 60:
        env_txt = env_txt[:57] + "..."
    return (
        f"sha={sha or '?'}  diff={diff}  "
        f"track={n_gates if n_gates is not None else '?'}gate  env={env_txt}"
    )


def load_rows(path):
    """Rows of one attempt CSV, or None if its schema is not the current one.

    The header IS the version. Comparing it beats a version integer because it
    also catches a REORDER, and the schema has already drifted nine times
    unversioned (16/20/22/23/27/28/32/33 columns across 63 files) -- so no
    reader may assume a layout."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if header != list(LOG_COLUMNS):
                return None
            return [dict(zip(header, r)) for r in rd if len(r) == len(header)]
    except OSError:
        return None


def collect(sidecar_paths, force=False):
    """Group attempts by config. Returns {key: {"meta", "attempts", "rows"}}."""
    groups, skipped = {}, 0
    for sp in sidecar_paths:
        try:
            with open(sp, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        # Cheap reject: the sidecar records the header, so a legacy schema is
        # identifiable without opening a single CSV.
        cols = meta.get("log_columns")
        if cols is not None and cols != list(LOG_COLUMNS):
            skipped += sum(1 for a in meta.get("attempts") or [] if a.get("telemetry"))
            continue
        key = ("ALL",) if force else config_key(meta)
        g = groups.setdefault(key, {"meta": meta, "attempts": [], "rows": []})
        for att in meta.get("attempts") or []:
            g["attempts"].append(att)
            name = att.get("telemetry")
            if not name:
                continue
            rows = load_rows(os.path.join(_LOGDIR, name))
            if rows is None:
                skipped += 1
                continue
            g["rows"].append(rows)
    return groups, skipped


def outcome_metrics(attempts):
    gates = [a["gates_passed"] for a in attempts if a.get("gates_passed") is not None]
    laps = [a["lap_s"] for a in attempts if a.get("lap_s") is not None]
    return {
        "n_attempts": len(attempts),
        "gates": gates,
        "best_gates": max(gates) if gates else None,
        "reached_1": sum(1 for g in gates if g >= 1),
        "n_scored": len(gates),
        "best_lap_s": min(laps) if laps else None,
        "outcomes": Counter(a.get("outcome") or a.get("end_reason") or "?" for a in attempts),
    }


def tick_metrics(row_sets):
    """Per-tick aggregate over every attempt in a group."""
    rows = [r for rs in row_sets for r in rs]
    if not rows:
        return {"n_ticks": 0}

    # Loop rate per attempt, so a gap between attempts is never a sample.
    dts = []
    for rs in row_sets:
        ts = [_f(r, "t") for r in rs]
        dts += [b - a for a, b in zip(ts, ts[1:]) if 0.0 < (b - a) < 1.0]

    roll = [abs(_f(r, "cmd_roll_deg")) for r in rows]
    pitch = [abs(_f(r, "cmd_pitch_deg")) for r in rows]
    yaw = [abs(_f(r, "cmd_yaw_deg")) for r in rows]
    thrust = [_f(r, "thrust") for r in rows]
    up = [_f(r, "up") for r in rows]
    vD = [_f(r, "vD") for r in rows]
    agl = [_f(r, "agl") for r in rows]
    infer = [_f(r, "infer_ms") for r in rows]
    bl_gap = [_f(r, "bl_gap") for r in rows]

    modes = Counter(r.get("mode", "?") for r in rows)
    dwell = {m: c / len(rows) for m, c in modes.items()}

    # Longest unbroken stretch with no usable vision, in seconds -- a long one
    # is a lost gate, which the mode dwell fraction alone cannot show.
    longest_blind, best = 0.0, 0.0
    for rs in row_sets:
        run_start = None
        for r in rs:
            blind = _f(r, "vision_valid", 0.0) < 0.5
            t = _f(r, "t")
            if blind and run_start is None:
                run_start = t
            elif not blind and run_start is not None:
                best = max(best, t - run_start)
                run_start = None
        if run_start is not None:
            best = max(best, _f(rs[-1], "t") - run_start)
    longest_blind = best

    return {
        "n_ticks": len(rows),
        "n_attempts_with_ticks": len(row_sets),
        "hz_median": (1.0 / statistics.median(dts)) if dts else float("nan"),
        "roll_sat": _frac(roll, lambda v: v >= ROLL_WIRE_MAX_DEG - _SAT_EPS),
        "pitch_sat": _frac(pitch, lambda v: v >= PITCH_WIRE_MAX_DEG - _SAT_EPS),
        "yaw_p99": _pct(yaw, 0.99),
        "yaw_max": max((v for v in yaw if not math.isnan(v)), default=float("nan")),
        "thrust_max_frac": _frac(thrust, lambda v: v >= 1.0 - _SAT_EPS),
        "thrust_zero_frac": _frac(thrust, lambda v: v <= _SAT_EPS),
        "up_min": min((v for v in up if not math.isnan(v)), default=float("nan")),
        "upset_ticks": sum(1 for v in up if not math.isnan(v) and v < _UPSET_COS),
        # What the guard DID, as distinct from how often the aircraft was
        # upset: these two diverge if GP_UPSET_GUARD is off or the threshold
        # is retuned, and that divergence is the thing to look at.
        "guard_fired": sum(1 for r in rows if _f(r, "upset", 0.0) >= 0.5),
        "over_descent": _frac(vD, lambda v: v > MAX_DESCENT_RATE_MPS),
        "vD_p95": _pct(vD, 0.95),
        "agl_min": min((v for v in agl if not math.isnan(v)), default=float("nan")),
        "reliable_frac": _frac(
            [_f(r, "reliable", 0.0) for r in rows], lambda v: v >= 0.5
        ),
        "blind_longest_s": longest_blind,
        "infer_p50": _pct(infer, 0.5),
        "infer_p90": _pct(infer, 0.9),
        "bl_gap_p90": _pct([g for g in bl_gap if g >= 0], 0.9),
        "dwell": dwell,
    }


# (label, key, format, direction). direction says which way is better and is
# what turns --baseline into a verdict instead of two bare numbers; 0 means
# informational, printed but never judged.
_TICK_ROWS = [
    ("loop rate median Hz", "hz_median", "{:.1f}", 0),
    ("|cmd_roll| at clamp", "roll_sat", "{:.1%}", -1),
    ("|cmd_pitch| at clamp", "pitch_sat", "{:.1%}", -1),
    ("cmd_yaw p99 deg", "yaw_p99", "{:.1f}", -1),
    ("cmd_yaw max deg", "yaw_max", "{:.1f}", -1),
    ("thrust at 1.000", "thrust_max_frac", "{:.1%}", -1),
    ("thrust at 0.000", "thrust_zero_frac", "{:.1%}", 0),
    ("min up (cos*cos)", "up_min", "{:.3f}", +1),
    ("ticks up < 0.50", "upset_ticks", "{:.0f}", -1),
    ("upset guard fired", "guard_fired", "{:.0f}", -1),
    ("ticks over descent cap", "over_descent", "{:.1%}", -1),
    ("vD p95 m/s", "vD_p95", "{:.2f}", -1),
    ("min agl m", "agl_min", "{:.2f}", +1),
    ("vision reliable", "reliable_frac", "{:.1%}", +1),
    ("longest blind run s", "blind_longest_s", "{:.2f}", -1),
    ("YOLO infer p50 ms", "infer_p50", "{:.0f}", -1),
    ("YOLO infer p90 ms", "infer_p90", "{:.0f}", -1),
    ("blue-line gap p90 fr", "bl_gap_p90", "{:.0f}", -1),
]


def _verdict(was, now, direction):
    if was is None or (isinstance(was, float) and math.isnan(was)):
        return ""
    if isinstance(now, float) and math.isnan(now):
        return ""
    if direction == 0 or abs(now - was) < 1e-9:
        return "--"
    return "better" if (now - was) * direction > 0 else "WORSE"


def report(key, out, tk, baseline=None):
    print(f"\nCONFIG   {describe_key(key)}")
    print(f"         {out['n_attempts']} attempts, {tk.get('n_ticks', 0)} ticks\n")

    # No verdicts here, deliberately -- see the module docstring.
    print(f"OUTCOME  (n={out['n_attempts']} attempts -- indicative only, no verdict)")
    gates = ",".join(str(g) for g in out["gates"]) or "-"
    print(f"  {'gates reached':22s} {gates}    best {out['best_gates']}")
    print(f"  {'reached gate >= 1':22s} {out['reached_1']}/{out['n_scored']}")
    if out["best_lap_s"] is not None:
        print(f"  {'best lap s':22s} {out['best_lap_s']:.2f}")
    ends = ", ".join(f"{k} x{v}" for k, v in out["outcomes"].most_common())
    print(f"  {'outcomes':22s} {ends}")

    if not tk.get("n_ticks"):
        print("\nPER-TICK  no readable telemetry for this config\n")
        return

    print(f"\nPER-TICK (n={tk['n_ticks']} ticks)", end="")
    if baseline is None:
        print()
        for label, k, fmt, _d in _TICK_ROWS:
            print(f"  {label:22s} {fmt.format(tk[k]):>10s}")
    else:
        print(f"{'':11s}{'baseline':>10s} {'now':>10s}   verdict")
        for label, k, fmt, direction in _TICK_ROWS:
            was, now = baseline.get(k), tk[k]
            wtxt = "" if was is None else fmt.format(was)
            print(
                f"  {label:22s} {wtxt:>10s} {fmt.format(now):>10s}   "
                f"{_verdict(was, now, direction)}"
            )
    dwell = ", ".join(
        f"{m} {f:.0%}" for m, f in sorted(tk["dwell"].items(), key=lambda kv: -kv[1])
    )
    print(f"  {'mode dwell':22s} {dwell}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", metavar="RUN_ID", help="score only this run id")
    ap.add_argument("--json", metavar="OUT", help="write metrics to this file")
    ap.add_argument("--baseline", metavar="IN", help="compare against a saved --json")
    ap.add_argument(
        "--force",
        action="store_true",
        help="pool every run into one group even if the config differs",
    )
    args = ap.parse_args()

    pattern = (
        os.path.join("runs", "videos", f"vision_{args.run}.json")
        if args.run
        else _SIDECARS
    )
    sidecars = sorted(glob.glob(pattern))
    if not sidecars:
        print(f"no sidecars matched {pattern}")
        return 1

    groups, skipped = collect(sidecars, force=args.force)
    baseline = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)

    dump = {}
    for key, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        out = outcome_metrics(g["attempts"])
        tk = tick_metrics(g["rows"])
        report(key, out, tk, (baseline or {}).get("per_tick") if baseline else None)
        dump = {"config": describe_key(key), "outcome": {**out, "outcomes": dict(out["outcomes"])}, "per_tick": tk}

    if skipped:
        print(f"  skipped {skipped} legacy logs (schema predates the current header)\n")
    if len(groups) > 1:
        print(
            "  NOTE: configs above are NOT comparable to each other. Re-run with\n"
            "  --run <RUN_ID> per config, or --force to pool them anyway.\n"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print(f"  wrote {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
