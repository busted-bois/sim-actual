"""Offline replay + shadow accuracy report for estimator flight logs.

A flight log (rl/data/est_log_*.jsonl, written by simulator/est_recorder.py)
holds the EXACT event stream the live estimator consumed. Replaying it
through the pure EstimatorCore reproduces the live estimate bit-for-bit
(asserted against the 1 Hz est_ckpt events recorded in flight), and the
odometry `truth` events (Training mode) grade it:

    uv run -m simulator.est_replay rl/data/est_log_X.jsonl
    uv run -m simulator.est_replay rl/data/est_log_X.jsonl --plot
    uv run -m simulator.est_replay rl/data/est_log_X.jsonl --set c_thrust=34
    uv run -m simulator.est_replay rl/data/est_log_X.jsonl --rederive-landmarks

    --set k=v            override a filter param (tuning sweeps, repeatable)
    --rederive-landmarks recompute p_meas = map_p - R(q_replayed) @ gate_body
                         from recorded provenance, i.e. judge landmarks with
                         the REPLAYED attitude instead of the live one
    --plot               save PNG (pos/z/attitude/landmarks) next to the log

Same module powers the end-of-flight shadow report (report_from_log, called
by SimInterface.finish_flight) and `--selftest` (make rl-test).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from simulator.est_recorder import read_log
from simulator.state_estimator import (
    EstimatorCore,
    quat_mult,
    quat_to_R,
)
from simulator.transforms import quat_to_yaw

# PASS/FAIL thresholds for the shadow report (Training-mode flights).
THRESHOLDS = {
    "pos_rms_m": 1.0,
    "pos_max_m": 3.0,
    "z_rms_m": 0.7,
    "yaw_max_deg": 10.0,
    "att_rms_deg": 5.0,
    "lm_accept_ratio": 0.5,
    "ckpt_dev": 1e-6,  # record/replay fidelity — live pose must reproduce
}


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_angle_deg(qa, qb):
    """Angle between two attitudes, degrees."""
    dq = quat_mult(np.array([qa[0], -qa[1], -qa[2], -qa[3]]), np.asarray(qb, float))
    return math.degrees(2 * math.acos(min(1.0, abs(float(dq[0])))))


class FrameAlign:
    """Fixed yaw-rotation + translation from the boot-anchored estimator
    frame into the odometry frame, captured from one paired sample (the two
    frames differ only by a constant SE(2)xZ offset)."""

    def __init__(self, p_est, q_est, p_odo, q_odo):
        dyaw = quat_to_yaw(*q_odo) - quat_to_yaw(*q_est)
        c, s = math.cos(dyaw), math.sin(dyaw)
        self.Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.off = np.asarray(p_odo, float) - self.Rz @ np.asarray(p_est, float)
        self.q_align = np.array([math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)])

    def pos(self, p_est):
        return self.Rz @ np.asarray(p_est, float) + self.off

    def vel(self, v_est):
        return self.Rz @ np.asarray(v_est, float)

    def quat(self, q_est):
        return quat_mult(self.q_align, np.asarray(q_est, float))


def replay(events, overrides=None, rederive_landmarks=False):
    """Run EstimatorCore over a recorded event stream.

    Returns a metrics dict; deterministic — same events, same numbers.
    """
    events = list(events)
    params = {}
    for ev in events:
        if ev["k"] == "meta":
            params = dict(ev["d"].get("params", {}))
            break
    if overrides:
        for k, v in overrides.items():
            if k not in EstimatorCore().p:
                raise SystemExit(f"unknown param {k!r} (have: {list(params)})")
            params[k] = v
    core = EstimatorCore(**params) if params else EstimatorCore()

    t0 = next((ev.get("tw") for ev in events if ev.get("tw")), 0.0) or 0.0
    align = None
    ckpt_dev = 0.0
    n_ckpt = 0
    lm_live_accepted = 0
    lm_events = []  # (t, accepted_in_replay)
    # Error series sampled at truth events: (t, pos_err, z_err, vel_err,
    # att_err_deg, yaw_err_deg, p_est_aligned, p_truth)
    series = []
    # Dead-reckoned trajectory sampled ~10 Hz at imu events: (t, p, v, yaw_deg).
    # This is what the plot shows for VQ2 logs, where no truth exists.
    est_series = []
    n_imu = 0

    for ev in events:
        k = ev["k"]
        t = (ev.get("tw") or t0) - t0
        if k == "imu":
            core.apply(ev)
            n_imu += 1
            if n_imu % 12 == 0:
                pose = core.pose()
                if pose is not None:
                    p_e, v_e, q_e = pose
                    est_series.append((t, p_e, v_e, math.degrees(quat_to_yaw(*q_e))))
            continue
        if k == "landmark":
            d = ev["d"]
            if d.get("accepted"):
                lm_live_accepted += 1
            p_meas = d["p_meas"]
            if rederive_landmarks and "map_p" in d and "gate_body" in d:
                pose = core.pose()
                if pose is None:
                    continue
                _, _, q_r = pose
                p_meas = np.asarray(d["map_p"], float) - quat_to_R(q_r) @ np.asarray(
                    d["gate_body"], float
                )
            ok = core.update_landmark(p_meas)
            lm_events.append((t, bool(ok)))
        elif k in ("thrust", "collision", "reset"):
            core.apply(ev)
        elif k == "est_ckpt":
            pose = core.pose()
            if pose is None:
                continue
            p_r, v_r, q_r = pose
            d = ev["d"]
            dev = max(
                float(np.max(np.abs(p_r - np.asarray(d["p"], float)))),
                float(np.max(np.abs(v_r - np.asarray(d["v"], float)))),
                float(np.max(np.abs(q_r - np.asarray(d["q"], float)))),
            )
            ckpt_dev = max(ckpt_dev, dev)
            n_ckpt += 1
        elif k == "truth":
            pose = core.pose()
            if pose is None:
                continue
            p_e, v_e, q_e = pose
            d = ev["d"]
            p_o = np.array([d["x"], d["y"], d["z"]])
            v_o = np.array([d["vx"], d["vy"], d["vz"]])
            q_o = np.array([d["qw"], d["qx"], d["qy"], d["qz"]])
            if align is None:
                align = FrameAlign(p_e, q_e, p_o, q_o)
            p_a, v_a, q_a = align.pos(p_e), align.vel(v_e), align.quat(q_e)
            yaw_err = math.degrees(abs(_wrap(quat_to_yaw(*q_o) - quat_to_yaw(*q_a))))
            series.append(
                (
                    t,
                    float(np.linalg.norm(p_a - p_o)),
                    abs(float(p_a[2] - p_o[2])),
                    float(np.linalg.norm(v_a - v_o)),
                    _quat_angle_deg(q_o, q_a),
                    yaw_err,
                    p_a,
                    p_o,
                )
            )

    m = {
        "n_events": len(events),
        "params": dict(core.p),
        "ckpt_dev": ckpt_dev if n_ckpt else None,
        "n_ckpt": n_ckpt,
        "lm_accepted": core.n_landmarks,
        "lm_rejected": core.n_landmarks_rejected,
        "lm_live_accepted": lm_live_accepted,
        "lm_innov_avg": (core.lm_innov_sum / core.n_landmarks)
        if core.n_landmarks
        else None,
        "lm_innov_max": core.lm_innov_max if core.n_landmarks else None,
        "nan_skipped": core.n_nan_skipped,
        "spikes_skipped": core.n_spikes_skipped,
        "series": series,
        "est_series": est_series,
        "lm_events": lm_events,
        "final_pose": core.pose(),
    }
    if series:
        arr = np.array([s[1:6] for s in series], float)  # pos,z,vel,att,yaw
        ts = np.array([s[0] for s in series])

        def rms(col):
            return float(np.sqrt(np.mean(arr[:, col] ** 2)))

        m.update(
            pos_rms_m=rms(0),
            pos_max_m=float(arr[:, 0].max()),
            z_rms_m=rms(1),
            z_max_m=float(arr[:, 1].max()),
            vel_rms=rms(2),
            att_rms_deg=rms(3),
            att_max_deg=float(arr[:, 3].max()),
            yaw_rms_deg=rms(4),
            yaw_max_deg=float(arr[:, 4].max()),
        )
        # Dead-reckoning drift: error growth over the longest span without an
        # accepted landmark (needs truth samples inside the span).
        lm_ts = [t for t, ok in lm_events if ok]
        edges = [ts[0], *lm_ts, ts[-1]]
        best = None
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < 3.0:
                continue
            sel = (ts >= a) & (ts <= b)
            if sel.sum() < 2:
                continue
            e = arr[sel, 0]
            rate = (e[-1] - e[0]) / (b - a)
            if best is None or (b - a) > best[0]:
                best = (b - a, rate)
        if best:
            m["drift_span_s"], m["drift_rate_mps"] = best
    return m


def _fmt(v, unit=""):
    return "n/a" if v is None else f"{v:.3f}{unit}"


def print_report(m, extra=None):
    total_lm = m["lm_accepted"] + m["lm_rejected"]
    ratio = m["lm_accepted"] / total_lm if total_lm else None
    have_truth = bool(m["series"])

    print("\n===== ESTIMATOR SHADOW REPORT =====")
    print(f"events={m['n_events']}  truth_samples={len(m['series'])}")
    checks = []  # (name, value_str, ok_or_None)

    def add(name, value, thr=None, less=True, unit=""):
        ok = None
        if value is not None and thr is not None:
            ok = (value < thr) if less else (value > thr)
        checks.append((name, _fmt(value, unit), ok, thr, unit))

    if have_truth:
        add("pos RMS", m.get("pos_rms_m"), THRESHOLDS["pos_rms_m"], unit=" m")
        add("pos max", m.get("pos_max_m"), THRESHOLDS["pos_max_m"], unit=" m")
        add("z RMS", m.get("z_rms_m"), THRESHOLDS["z_rms_m"], unit=" m")
        add("att RMS", m.get("att_rms_deg"), THRESHOLDS["att_rms_deg"], unit=" deg")
        add("yaw max", m.get("yaw_max_deg"), THRESHOLDS["yaw_max_deg"], unit=" deg")
        add("vel RMS", m.get("vel_rms"), unit=" m/s")
    else:
        print(
            "(no odometry truth in log -- accuracy metrics unavailable; "
            "fly in Training mode for a graded report)"
        )
    add("landmark accept ratio", ratio, THRESHOLDS["lm_accept_ratio"], less=False)
    add("replay fidelity (ckpt dev)", m["ckpt_dev"], THRESHOLDS["ckpt_dev"])
    if "drift_rate_mps" in m:
        print(
            f"  dead-reckoning drift: {m['drift_rate_mps']:+.3f} m/s over the "
            f"longest landmark-free span ({m['drift_span_s']:.1f} s)"
        )
    print(
        f"  landmarks: {m['lm_accepted']} accepted / {m['lm_rejected']} rejected"
        f" (live run accepted {m['lm_live_accepted']})"
        f"   imu skipped: {m['nan_skipped']} NaN, {m['spikes_skipped']} spikes"
    )
    if m.get("lm_innov_avg") is not None:
        # How far the estimate sat from each accepted vision fix -- the
        # truth-free estimation-quality number to drive down across flights.
        print(
            f"  landmark innovation: avg {m['lm_innov_avg']:.2f} m, "
            f"max {m['lm_innov_max']:.2f} m"
        )
    for name, val, ok, thr, unit in checks:
        mark = "  --  " if ok is None else ("  PASS" if ok else "  FAIL")
        lim = "" if thr is None else f"   (limit {thr}{unit})"
        print(f"  {name:<28}{val:>12}{mark}{lim}")
    graded = [ok for _, _, ok, _, _ in checks if ok is not None]
    verdict = "PASS" if graded and all(graded) else ("FAIL" if graded else "UNGRADED")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    print(f"===== VERDICT: {verdict} =====\n")
    return verdict


def report_from_log(path, extra=None, overrides=None, rederive_landmarks=False):
    m = replay(read_log(path), overrides, rederive_landmarks)
    m["verdict"] = print_report(m, extra=extra)
    return m


def save_plot(m, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = m["series"]
    es = m.get("est_series") or []
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    if s:
        # Graded (Training-mode) log: estimate vs odometry truth.
        ts = [x[0] for x in s]
        pe = np.array([x[6] for x in s])
        po = np.array([x[7] for x in s])
        for i, lbl in enumerate("xyz"):
            axes[0].plot(ts, po[:, i], lw=1.8, label=f"truth {lbl}")
            axes[0].plot(ts, pe[:, i], lw=1.0, ls="--", label=f"est {lbl}")
        axes[0].set_ylabel("pos NED (m)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].plot(ts, po[:, 2], lw=1.8, label="truth z")
        axes[1].plot(ts, pe[:, 2], lw=1.0, ls="--", label="est z")
        axes[1].set_ylabel("z (m)")
        axes[1].legend(fontsize=8)
        axes[2].plot(ts, [x[4] for x in s], label="attitude err (deg)")
        axes[2].plot(ts, [x[5] for x in s], label="yaw err (deg)")
        axes[2].set_ylabel("deg")
        axes[2].legend(fontsize=8)
        fig.suptitle("estimator replay vs odometry truth")
    elif es:
        # Ungraded (VQ2) log: dead-reckoned trajectory only.
        ts = [x[0] for x in es]
        pe = np.array([x[1] for x in es])
        ve = np.array([x[2] for x in es])
        for i, lbl in enumerate("xyz"):
            axes[0].plot(ts, pe[:, i], lw=1.2, label=f"est {lbl}")
        axes[0].set_ylabel("est pos NED (m)")
        axes[0].legend(ncol=3, fontsize=8)
        for i, lbl in enumerate("xyz"):
            axes[1].plot(ts, ve[:, i], lw=1.0, label=f"est v{lbl}")
        axes[1].set_ylabel("est vel (m/s)")
        axes[1].legend(ncol=3, fontsize=8)
        axes[2].plot(ts, [x[3] for x in es], label="est yaw (deg)")
        axes[2].set_ylabel("deg")
        axes[2].legend(fontsize=8)
        fig.suptitle("estimator replay (no truth in log — dead-reckoned estimate)")
    for t, ok in m["lm_events"]:
        axes[3].axvline(t, color="g" if ok else "r", alpha=0.4, lw=0.8)
    axes[3].set_ylabel("landmarks\n(green=accepted)")
    axes[3].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"[replay] plot -> {out_path}")


# ---------------------------------------------------------------------------
def _selftest():
    """Round-trip determinism: live adapter + recorder -> JSON -> replay
    reproduces the live estimate exactly. No live sim, no files."""
    from simulator.est_recorder import ListRecorder, _jsonable
    from simulator.state_estimator import GRAVITY, G_WORLD, StateEstimator

    rng = np.random.default_rng(11)
    rec = ListRecorder()
    est = StateEstimator(recorder=rec, init_samples=50)
    nan = float("nan")

    def imu(k, f, g):
        return dict(
            ax=f[0],
            ay=f[1],
            az=f[2],
            gx=g[0],
            gy=g[1],
            gz=g[2],
            mx=nan,
            my=nan,
            mz=nan,
            abs_pressure=nan,
            pressure_alt=nan,
            temperature=nan,
            time_us=k,
        )

    for k in range(50):  # ground init
        est.on_imu(imu(k, -G_WORLD + rng.normal(0, 0.02, 3), np.zeros(3)))
    assert est.ready

    dt = 1 / 150.0
    est.thrust_cmd = GRAVITY / est.p["c_thrust"]
    kd = est.p["drag_kd"]
    v0 = np.array([2.0, 1.0, -0.5])
    n = int(12.0 / dt)
    for k in range(n):
        t = k * dt
        p_rel = v0 / kd * (1.0 - math.exp(-kd * t))
        est.on_imu(
            imu(
                50 + int((t + dt) * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.002, 3),
            )
        )
        if t > 6.0 and k % 30 == 0:
            est.update_landmark(
                p_rel + rng.normal(0, 0.15, 3),
                raw={
                    "map_p": p_rel,
                    "gate_body": np.zeros(3),
                    "quat_used": est.pose()[2],
                },
            )
            # Truth stream (as mavlink_rx would record it, ~20 Hz is plenty).
            rec.log(
                "truth",
                {
                    "time_us": int(t * 1e6),
                    "x": p_rel[0],
                    "y": p_rel[1],
                    "z": p_rel[2],
                    "vx": 0.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "qw": 1.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                },
            )
    est.notify_collision()
    for k in range(300):  # post-collision samples
        est.on_imu(
            imu(
                10**9 + int(k * dt * 1e6),
                rng.uniform(-400, 400, 3),
                rng.normal(0, 0.002, 3),
            )
        )
    p_live, v_live, q_live = est.pose()

    # JSON round-trip (exactly what the file does), then replay twice.
    events = [json.loads(json.dumps(_jsonable(e))) for e in rec.events]
    m1 = replay(events)
    m2 = replay(events)
    p_r, v_r, q_r = m1["final_pose"]
    dev = max(
        float(np.max(np.abs(p_r - p_live))),
        float(np.max(np.abs(v_r - v_live))),
        float(np.max(np.abs(q_r - q_live))),
    )
    print(
        f"[selftest] replay-vs-live final deviation = {dev:.2e} "
        f"(ckpt_dev={m1['ckpt_dev']:.2e}, {m1['n_ckpt']} ckpts)"
    )
    assert dev < 1e-9, "replay must reproduce the live estimate"
    assert m1["ckpt_dev"] is not None and m1["ckpt_dev"] < 1e-9
    assert m1["lm_accepted"] == m1["lm_live_accepted"], "verdicts must match live"

    def strip(m):
        return {
            k: v
            for k, v in m.items()
            if k not in ("series", "est_series", "lm_events", "final_pose")
        }

    assert json.dumps(strip(m1), sort_keys=True) == json.dumps(
        strip(m2), sort_keys=True
    ), "replay must be deterministic"
    assert m1["series"], "truth events must produce graded metrics"
    assert m1["pos_rms_m"] < 1.0

    # Param override changes the outcome (tuning sweeps work).
    m3 = replay(events, overrides={"c_thrust": 30.0})
    assert m3["pos_rms_m"] != m1["pos_rms_m"]

    # Rederived landmarks run end-to-end.
    m4 = replay(events, rederive_landmarks=True)
    assert (
        m4["lm_accepted"] + m4["lm_rejected"] == m1["lm_accepted"] + m1["lm_rejected"]
    )

    print_report(m1, extra={"scenario": "selftest synthetic flight"})
    print("[selftest] OK — record -> JSON -> replay is exact and deterministic")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", help="rl/data/est_log_*.jsonl")
    ap.add_argument("--plot", action="store_true", help="save PNG next to the log")
    ap.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="K=V",
        help="override a filter param (repeatable)",
    )
    ap.add_argument("--rederive-landmarks", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.log:
        ap.error("log path required (or --selftest)")
    overrides = {}
    for s in args.sets:
        k, _, v = s.partition("=")
        overrides[k.strip()] = float(v)
    m = report_from_log(
        args.log, overrides=overrides, rederive_landmarks=args.rederive_landmarks
    )
    if args.plot:
        stem = os.path.splitext(os.path.basename(args.log))[0]
        out = os.path.join(os.path.dirname(args.log) or ".", f"est_replay_{stem}.png")
        save_plot(m, out)


if __name__ == "__main__":
    main()
