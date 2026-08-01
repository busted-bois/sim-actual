"""Replay a run recording through the blue-line detector and score its output.

Every corridor-detector change is measurable off a recorded mp4, so it does not
need a flight to evaluate. This runs `BlueLineTracker` over the video exactly as
`vision_rx` does and prints the signal-quality metrics the detector is actually
judged on -- run it before and after a change and diff the two tables.

    uv run scripts/bl_replay.py runs/videos/vision.mp4
    uv run scripts/bl_replay.py runs/videos/vision.mp4 --start 900 --count 600
    uv run scripts/bl_replay.py runs/videos/vision.mp4 --json before.json
    uv run scripts/bl_replay.py runs/videos/vision.mp4 --baseline before.json

Baseline vs the B-workstream fixes, default invocation above (frames 900-1500,
2026-07-30). Reproduce the baseline column with all four flags off:

    GP_BL_HDG_GATE=0 GP_BL_CENTRE_FIT=0 GP_BL_CX_FILTER=0 uv run scripts/bl_replay.py

                        before     after
    heading valid        25.2 %    34.4 %
    |d hdg| p90           6.29      2.67   deg
    |d hdg| max          25.00      6.25   deg   <- off the rate limiter
    hdg sign flips        2.25      0.91   /s
    |d cx| p90            0.133     0.090
    |d cx| max            0.709     0.090        <- cx had no filter at all
    cx sign flips         2.33      1.70   /s
    conf < 0.50          39.6 %    35.2 %

`|d hdg| max` sitting exactly ON the tracker's own 25 deg/frame rate limiter was
the tell: the limiter was hiding a broken estimate rather than smoothing a good
one. Nothing reaches it now.

Two p50s move the "wrong" way (|d hdg| 0.11 -> 0.17, |d cx| 0.003 -> 0.013) —
that is the EMAs spreading a correction over several frames instead of applying
it in one, which is the point.

What did NOT move: band span p50 stays 0.24 and detect rate stays 79.5 %. The
20 deg up-tilted camera simply does not show more corridor, and no amount of
resampling invents lever arm — see the GP_BL_ADAPTIVE_BANDS negative result in
blue_line_vision.py.

Why these seven: the span drives whether a heading fit means anything; the two
step/flip pairs are what reaches the roll and yaw commands as noise; crossings
are a hard geometry bug (the centre lands a full corridor-width wrong); and conf
is what every downstream gate in `gp_pilot` keys off.

This is a leaf tool: it imports the detector, nothing imports it.
"""

import argparse
import json
import math
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simulator.blue_line_vision as B  # noqa: E402
from simulator.track_line import track_mask  # noqa: E402


def _pct(vals, p):
    """Percentile of a plain list, without pulling in the numpy machinery."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[int(p * (len(s) - 1))]


def _sign_flips_per_s(vals, fps):
    """Zero crossings per second -- the shape of the noise the pilot feels."""
    if len(vals) < 2:
        return float("nan")
    n = sum(1 for i in range(1, len(vals)) if vals[i] * vals[i - 1] < 0)
    return n * fps / len(vals)


def _fit_hazard(mask, sh, sw, ds):
    """(independent fits would cross, side assignment differs) for one frame.

    Two separate questions, because the centre+half-width fit makes the first
    one unobservable by construction (centre +- |hw| cannot invert):

    * `cross` is the EXPOSURE: how often independently-fitted rails invert at a
      row a band actually occupies. It measures the hazard the frame presents,
      so it stays comparable across the fix rather than collapsing to 0%.
    * `side_flip` is the IMPACT: how often that hazard would actually have
      placed a single-rail band's centre on the wrong side.
    """
    area = ds * ds
    min_px = max(4, B._MIN_BAND_PX_FULLRES // area)
    min_run = max(3, B._MIN_RUN_PX_FULLRES // area)
    gap = max(2, B._RUN_MERGE_GAP_FULLRES // ds)
    sep = max(4, B._MIN_RAIL_SEP_FULLRES // ds)
    edges = B._band_edges(mask, sh)

    bands = []
    for i in range(B._N_BANDS - 1, -1, -1):
        r = B._band_runs(
            mask[edges[i] : edges[i + 1]], edges[i], sw, min_px, min_run, gap, sep
        )
        if r is not None:
            bands.append(r)

    pv, pl, pr = [], [], []
    for v, runs in bands:
        if len(runs) >= 2 and (runs[-1][0] - runs[0][0]) >= sep:
            pv.append(v)
            pl.append(runs[0][0])
            pr.append(runs[-1][0])
    if len(pv) < 2:
        return None  # no two-rail fit to cross

    ind_l, ind_r = B._fit(pv, pl), B._fit(pv, pr)
    fit_hw = B._fit(pv, [0.5 * (b - a) for a, b in zip(pl, pr)])
    fit_c = B._fit(pv, [0.5 * (a + b) for a, b in zip(pl, pr)])

    cross = any(ind_r(v) <= ind_l(v) for v, _ in bands)
    side_flip = False
    for v, runs in bands:
        if len(runs) >= 2 and (runs[-1][0] - runs[0][0]) >= sep:
            continue  # band resolved both rails itself; no side to infer
        u = runs[0][0]
        old = 1 if abs(u - ind_r(v)) < abs(u - ind_l(v)) else -1
        hw = abs(fit_hw(v))
        new = 1 if abs(u - (fit_c(v) + hw)) < abs(u - (fit_c(v) - hw)) else -1
        if old != new:
            side_flip = True
            break
    return cross, side_flip


def measure(path, start=0, count=600):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    tracker = B.BlueLineTracker()
    cxs, hdgs, spans, confs, bands_used = [], [], [], [], []
    dcx, dhdg = [], []
    crossed = crossable = flipped = 0
    seen = found = 0
    hdg_valid = 0
    prev = None

    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        seen += 1
        est, _ = tracker.update(frame, start + i)

        ds = B._DOWNSCALE
        mask = track_mask(frame, ds)
        sh, sw = mask.shape[:2]
        hz = _fit_hazard(mask, sh, sw, ds)
        if hz is not None:
            crossable += 1
            crossed += bool(hz[0])
            flipped += bool(hz[1])

        if not est.found:
            prev = None
            continue
        found += 1
        h = frame.shape[0]
        vs = [v for _u, v in est.points]
        spans.append((max(vs) - min(vs)) / h if vs else 0.0)
        bands_used.append(len(est.points))
        confs.append(est.conf)
        cx = est.cx_norm
        hdg = math.degrees(est.heading_err)
        # heading_valid is added by the B1 fix; absent on the baseline detector.
        valid = bool(getattr(est, "heading_valid", True))
        cxs.append(cx)
        if valid:
            hdgs.append(hdg)
            hdg_valid += 1
        if prev is not None:
            dcx.append(abs(cx - prev[0]))
            # Only across consecutive VALID frames: a gated heading publishes
            # 0.0 meaning "none", and differencing that against a real angle
            # measures the gate firing, not the estimate moving.
            if valid and prev[2]:
                dhdg.append(abs(hdg - prev[1]))
        prev = (cx, hdg, valid)

    cap.release()
    return {
        "video": os.path.basename(path),
        "total_frames": total,
        "fps": fps,
        "range": [start, start + seen],
        "frames": seen,
        "detect_rate": found / max(seen, 1),
        "hdg_valid_rate": hdg_valid / max(found, 1),
        "bands_p50": _pct(bands_used, 0.5),
        "span_p50": _pct(spans, 0.5),
        "span_p90": _pct(spans, 0.9),
        "dhdg_p50": _pct(dhdg, 0.5),
        "dhdg_p90": _pct(dhdg, 0.9),
        "dhdg_max": max(dhdg) if dhdg else float("nan"),
        "hdg_flips_s": _sign_flips_per_s(hdgs, fps),
        "dcx_p50": _pct(dcx, 0.5),
        "dcx_p90": _pct(dcx, 0.9),
        "dcx_max": max(dcx) if dcx else float("nan"),
        "cx_flips_s": _sign_flips_per_s(cxs, fps),
        "cross_rate": crossed / max(crossable, 1),
        "side_flip_rate": flipped / max(crossable, 1),
        "conf_p50": _pct(confs, 0.5),
        "conf_lt_50": sum(1 for c in confs if c < 0.5) / max(len(confs), 1),
    }


# (label, key, format, direction) -- direction says which way is better, and is
# what lets --baseline print a verdict instead of just two numbers.
_ROWS = [
    ("detect rate", "detect_rate", "{:.1%}", +1),
    ("heading valid", "hdg_valid_rate", "{:.1%}", 0),
    ("bands used p50", "bands_p50", "{:.0f}", +1),
    ("band span p50", "span_p50", "{:.2f}", +1),
    ("band span p90", "span_p90", "{:.2f}", +1),
    ("|d hdg| p50 deg", "dhdg_p50", "{:.2f}", -1),
    ("|d hdg| p90 deg", "dhdg_p90", "{:.2f}", -1),
    ("|d hdg| max deg", "dhdg_max", "{:.2f}", -1),
    ("hdg flips /s", "hdg_flips_s", "{:.2f}", -1),
    ("|d cx| p50", "dcx_p50", "{:.3f}", -1),
    ("|d cx| p90", "dcx_p90", "{:.3f}", -1),
    ("|d cx| max", "dcx_max", "{:.3f}", -1),
    ("cx flips /s", "cx_flips_s", "{:.2f}", -1),
    # Informational, not scored: on a frame where the independent fits cross,
    # centre +- |half_width| is the correct answer by construction, so a higher
    # "side corrected" number means the fix is catching more, not failing more.
    ("indep fits cross", "cross_rate", "{:.1%}", 0),
    ("  side corrected", "side_flip_rate", "{:.1%}", 0),
    ("conf p50", "conf_p50", "{:.2f}", 0),
    ("conf < 0.50", "conf_lt_50", "{:.1%}", -1),
]


def report(m, baseline=None):
    print(
        f"\n{m['video']}  frames {m['range'][0]}-{m['range'][1]} "
        f"of {m['total_frames']} @ {m['fps']:.1f} fps\n"
    )
    if baseline is None:
        for label, key, fmt, _dir in _ROWS:
            print(f"  {label:18s} {fmt.format(m[key]):>10s}")
        print()
        return

    print(f"  {'':18s} {'baseline':>10s} {'now':>10s}   verdict")
    for label, key, fmt, direction in _ROWS:
        was, now = baseline.get(key), m[key]
        if was is None or (isinstance(was, float) and math.isnan(was)):
            verdict = ""
        elif direction == 0 or abs(now - was) < 1e-9:
            verdict = "--"
        else:
            verdict = "better" if (now - was) * direction > 0 else "WORSE"
        wtxt = "" if was is None else fmt.format(was)
        print(f"  {label:18s} {wtxt:>10s} {fmt.format(now):>10s}   {verdict}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", nargs="?", default="runs/videos/vision.mp4")
    ap.add_argument("--start", type=int, default=900, help="first frame index")
    ap.add_argument("--count", type=int, default=600, help="frames to replay")
    ap.add_argument("--json", metavar="OUT", help="write metrics to this file")
    ap.add_argument("--baseline", metavar="IN", help="compare against a saved --json")
    args = ap.parse_args()

    m = measure(args.video, args.start, args.count)

    baseline = None
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)
    report(m, baseline)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)
        print(f"  wrote {args.json}\n")


if __name__ == "__main__":
    main()
