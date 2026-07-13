"""Entry point: vertical test suite, report, exit code.

Usage:
    uv run python -m flightlab.run_vertical
    uv run python -m flightlab.run_vertical --method pid --only V1
    uv run python -m flightlab.run_vertical --list
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# MAVLINK20 before any pymavlink (via bus import chain).
os.environ["MAVLINK20"] = "1"

from flightlab.bus import CONTROL_HZ, HOVER_THRUST, Bus
from flightlab.controllers import list_methods, make_controller
from flightlab import metrics as M
from flightlab.maneuvers import Schedule, Segment, altitude_steps, lean_hold
from flightlab.protocol import Controller, Target
from flightlab.safety import SafetyMonitor
from flightlab.state import State

DT = 1.0 / CONTROL_HZ
GROUND_Z_THRESH = 0.15  # NED z near spawn/ground
SETTLE_VZ = 0.15  # m/s
SETTLE_HOLD_S = 0.4


@dataclass
class TickLog:
    t: float
    n: float
    e: float
    z: float
    vn: float
    ve: float
    vz: float
    roll: float
    pitch: float
    yaw: float
    thrust: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    z_tgt: float | None
    vz_tgt: float | None
    label: str
    test: str


@dataclass
class TestResult:
    name: str
    passed: bool
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    blowup: bool = False


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _alt(z: float) -> float:
    return -z


def _run_loop(
    bus: Bus,
    ctrl: Controller,
    safety: SafetyMonitor,
    schedule: Schedule,
    test_name: str,
    log: list[TickLog],
    max_s: float | None = None,
    stop_fn=None,
) -> tuple[bool, str]:
    """Closed-loop until schedule done, safety trip, or stop_fn(state)->True.

    Returns (ok, reason). ok=False on blowup.
    """
    t0 = time.monotonic()
    last = t0
    s = bus.drain()
    ctrl.reset(s)

    while True:
        loop_start = time.monotonic()
        elapsed = loop_start - t0
        if max_s is not None and elapsed >= max_s:
            return True, "timeout_ok"
        if schedule.done(elapsed) and stop_fn is None:
            return True, "schedule_done"

        s = bus.drain()
        sr = safety.check(s)
        if sr.tripped:
            safety.handle_trip(bus, sr.reason)
            return False, f"blowup:{sr.reason}"

        tgt = schedule.target_at(elapsed)
        if tgt is None and stop_fn is None:
            return True, "schedule_done"
        if tgt is None:
            tgt = Target(z=s.pos[2], vz=0.0)

        dt = loop_start - last
        last = loop_start
        cmd = ctrl.update(s, tgt, dt)
        bus.send(cmd.roll_rate, cmd.pitch_rate, cmd.yaw_rate, cmd.thrust)

        log.append(
            TickLog(
                t=elapsed,
                n=s.pos[0],
                e=s.pos[1],
                z=s.pos[2],
                vn=s.vel[0],
                ve=s.vel[1],
                vz=s.vel[2],
                roll=s.roll,
                pitch=s.pitch,
                yaw=s.yaw,
                thrust=cmd.thrust,
                roll_rate=cmd.roll_rate,
                pitch_rate=cmd.pitch_rate,
                yaw_rate=cmd.yaw_rate,
                z_tgt=tgt.z,
                vz_tgt=tgt.vz,
                label=schedule.label_at(elapsed),
                test=test_name,
            )
        )

        if stop_fn is not None and stop_fn(s, elapsed):
            return True, "stop_fn"

        # Pace to CONTROL_HZ
        spent = time.monotonic() - loop_start
        sleep = DT - spent
        if sleep > 0:
            time.sleep(sleep)


def _takeoff_to(
    bus: Bus,
    ctrl: Controller,
    safety: SafetyMonitor,
    z_tgt: float,
    log: list[TickLog],
    test: str,
) -> tuple[bool, str]:
    """Climb/descend to z_tgt and settle briefly."""
    sched = Schedule(
        [
            Segment(duration=20.0, z=z_tgt, vz=0.0, label="takeoff"),
        ]
    )

    def settled(s: State, elapsed: float) -> bool:
        if elapsed < 1.0:
            return False
        return abs(s.pos[2] - z_tgt) < 0.3 and abs(s.vel[2]) < 0.3 and elapsed > 2.0

    return _run_loop(bus, ctrl, safety, sched, test, log, max_s=25.0, stop_fn=settled)


def _prep(bus: Bus, ctrl: Controller, safety: SafetyMonitor) -> bool:
    safety.reset()
    # Snapshot race_start so we can detect a fresh countdown after reset.
    bus.drain()
    race = bus.tracker.data.get("race_status") or {}
    if race.get("race_start_boot_time_ms", -1) >= 0:
        bus.tracker.data["_preflight_race_start_baseline"] = race[
            "race_start_boot_time_ms"
        ]
    bus.reset()
    time.sleep(2.0)
    # Re-boot ESKF after teleport (VQ2 has no odometry).
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15.0:
        s = bus.drain()
        if s.has_pose:
            break
        time.sleep(0.02)
    else:
        print("[prep] no pose after reset", flush=True)
        return False
    # Do NOT arm/climb during 3-2-1 — wait until on-screen GO.
    if not bus.wait_for_fresh_race_start(timeout_s=30.0):
        print("[prep] no fresh race_start after reset", flush=True)
        return False
    if not bus.wait_for_race_go(timeout_s=45.0, is_restart=True):
        print("[prep] race GO timeout — stay on countdown until 0", flush=True)
        return False
    # Hold level + hover briefly so estimator sees thrust_cmd before climb.
    for _ in range(int(0.5 * CONTROL_HZ)):
        bus.drain()
        bus.send(0.0, 0.0, 0.0, HOVER_THRUST)
        time.sleep(1.0 / CONTROL_HZ)
    if not bus.arm(timeout_s=15.0):
        return False
    time.sleep(0.3)
    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_v1(
    bus: Bus, ctrl: Controller, safety: SafetyMonitor, log: list[TickLog]
) -> TestResult:
    """Hover calibration + jitter at 3 m AGL for 30 s."""
    name = "V1"
    if not _prep(bus, ctrl, safety):
        return TestResult(name, False, "arm_failed")

    # Spawn z≈0; 3 m AGL → z = -3
    z_hold = -3.0
    ok, reason = _takeoff_to(bus, ctrl, safety, z_hold, log, name)
    if not ok:
        return TestResult(name, False, reason, blowup=reason.startswith("blowup"))

    # Clear takeoff samples from metric window — mark hold start
    hold_start_idx = len(log)
    sched = Schedule([Segment(duration=30.0, z=z_hold, vz=0.0, label="hover")])
    ok, reason = _run_loop(bus, ctrl, safety, sched, name, log, max_s=32.0)
    if not ok:
        return TestResult(name, False, reason, blowup=True)

    hold = log[hold_start_idx:]
    if len(hold) < 10:
        return TestResult(name, False, "too_few_samples")

    alts = [_alt(r.z) for r in hold]
    vzs = [r.vz for r in hold]
    thrusts = [r.thrust for r in hold]
    rolls = [math.degrees(r.roll) for r in hold]
    pitches = [math.degrees(r.pitch) for r in hold]
    ns = [r.n for r in hold]
    es = [r.e for r in hold]
    drift = math.hypot(ns[-1] - ns[0], es[-1] - es[0])

    m = {
        "alt_std": M.std(alts),
        "drift_m": drift,
        "vz_std": M.std(vzs),
        "thrust_std": M.std(thrusts),
        "thrust_mean": M.mean(thrusts),
        "roll_p2p_deg": M.peak_to_peak(rolls),
        "pitch_p2p_deg": M.peak_to_peak(pitches),
        "true_hover_thrust": M.mean(thrusts),
    }
    checks = [
        (m["alt_std"] < 0.15, f"alt_std={m['alt_std']:.3f}"),
        (m["drift_m"] < 0.3, f"drift={m['drift_m']:.3f}"),
        (m["vz_std"] < 0.2, f"vz_std={m['vz_std']:.3f}"),
        (m["thrust_std"] < 0.02, f"thrust_std={m['thrust_std']:.4f}"),
        (m["roll_p2p_deg"] < 3.0, f"roll_p2p={m['roll_p2p_deg']:.2f}"),
        (m["pitch_p2p_deg"] < 3.0, f"pitch_p2p={m['pitch_p2p_deg']:.2f}"),
    ]
    failed = [c[1] for c in checks if not c[0]]
    passed = len(failed) == 0
    return TestResult(
        name,
        passed,
        "PASS" if passed else "FAIL:" + ";".join(failed),
        metrics=m,
    )


def test_v2(
    bus: Bus, ctrl: Controller, safety: SafetyMonitor, log: list[TickLog]
) -> TestResult:
    """Altitude steps +5 m / −5 m."""
    name = "V2"
    if not _prep(bus, ctrl, safety):
        return TestResult(name, False, "arm_failed")

    z_base = -3.0
    ok, reason = _takeoff_to(bus, ctrl, safety, z_base, log, name)
    if not ok:
        return TestResult(name, False, reason, blowup=reason.startswith("blowup"))

    step_idx = len(log)
    sched = altitude_steps(z_base, delta=5.0, hold_s=5.0)
    ok, reason = _run_loop(bus, ctrl, safety, sched, name, log, max_s=12.0)
    if not ok:
        return TestResult(name, False, reason, blowup=True)

    rows = log[step_idx:]
    t = [r.t - rows[0].t for r in rows]
    alts = [_alt(r.z) for r in rows]
    # Climb step: +5 m AGL → alt from 3 → 8; z from -3 → -8
    z_up = z_base - 5.0
    alt_base = _alt(z_base)
    alt_up = _alt(z_up)

    # Split at midpoint of schedule (5 s)
    climb = [(ti, a, r) for ti, a, r in zip(t, alts, rows) if ti < 5.0]
    descend = [(ti, a, r) for ti, a, r in zip(t, alts, rows) if ti >= 5.0]

    def _step_metrics(seg, start_alt, tgt_alt, t_offset):
        if len(seg) < 5:
            return None
        ts = [x[0] - t_offset for x in seg]
        ys = [x[1] for x in seg]
        ov = M.overshoot_frac(ys, start_alt, tgt_alt)
        st = M.settle_time_enter(ts, ys, tgt_alt, 0.25, 0.0, hold_s=0.3)
        errs = [y - tgt_alt for y in ys]
        peaks = M.extract_error_peaks(ts, errs, 0.0)
        bounce = M.secondary_bounce_ok(peaks, 0.5)
        return ov, st, bounce, peaks

    c = _step_metrics(climb, alt_base, alt_up, 0.0)
    d = _step_metrics(descend, alt_up, alt_base, 5.0)
    if c is None or d is None:
        return TestResult(name, False, "too_few_samples")

    m = {
        "climb_overshoot": c[0],
        "climb_settle_s": c[1],
        "climb_bounce_ok": c[2],
        "descend_overshoot": d[0],
        "descend_settle_s": d[1],
        "descend_bounce_ok": d[2],
    }
    checks = [
        (c[0] < 0.15, f"climb_os={c[0]:.2%}"),
        (c[1] is not None and c[1] < 2.5, f"climb_settle={c[1]}"),
        (c[2], "climb_bounce"),
        (d[0] < 0.15, f"desc_os={d[0]:.2%}"),
        (d[1] is not None and d[1] < 2.5, f"desc_settle={d[1]}"),
        (d[2], "desc_bounce"),
    ]
    failed = [x[1] for x in checks if not x[0]]
    passed = len(failed) == 0
    return TestResult(
        name,
        passed,
        "PASS" if passed else "FAIL:" + ";".join(failed),
        metrics=m,
    )


def test_v3(
    bus: Bus, ctrl: Controller, safety: SafetyMonitor, log: list[TickLog]
) -> TestResult:
    """Rate tracking: vz = -2,-4,+2,+4 m/s, 3 s each."""
    name = "V3"
    if not _prep(bus, ctrl, safety):
        return TestResult(name, False, "arm_failed")

    z0 = -8.0  # room to climb/descend
    ok, reason = _takeoff_to(bus, ctrl, safety, z0, log, name)
    if not ok:
        return TestResult(name, False, reason, blowup=reason.startswith("blowup"))

    rates = [-2.0, -4.0, 2.0, 4.0]
    segs = [Segment(duration=3.0, z=None, vz=vz, label=f"vz={vz:+.1f}") for vz in rates]
    # Between rate segments keep mild z authority off — pure vz tracking
    sched = Schedule(segs)
    start_idx = len(log)
    ok, reason = _run_loop(bus, ctrl, safety, sched, name, log, max_s=14.0)
    if not ok:
        return TestResult(name, False, reason, blowup=True)

    rows = log[start_idx:]
    m: dict = {}
    failed = []
    for i, vz_tgt in enumerate(rates):
        t0 = i * 3.0
        t1 = t0 + 3.0
        # Use middle 2 s to avoid transients
        window = [
            r for r in rows if (r.t - rows[0].t) >= t0 + 0.5 and (r.t - rows[0].t) < t1
        ]
        if len(window) < 5:
            failed.append(f"vz={vz_tgt}:few")
            continue
        mean_vz = M.mean([r.vz for r in window])
        mean_thrust = M.mean([r.thrust for r in window])
        key = f"vz_{vz_tgt:+.0f}"
        err_frac = abs(mean_vz - vz_tgt) / abs(vz_tgt) if vz_tgt != 0 else 0.0
        sat = mean_thrust <= 0.13 or mean_thrust >= 0.58
        m[key] = {
            "mean_vz": mean_vz,
            "target": vz_tgt,
            "err_frac": err_frac,
            "mean_thrust": mean_thrust,
            "saturated": sat,
        }
        if err_frac > 0.15 and not sat:
            failed.append(f"{key}:err={err_frac:.0%}")
        elif sat:
            m[key]["note"] = "saturated_at_thrust_clamp"

    # Max climb / descend achieved
    climb = [
        v["mean_vz"] for k, v in m.items() if isinstance(v, dict) and v["target"] < 0
    ]
    desc = [
        v["mean_vz"] for k, v in m.items() if isinstance(v, dict) and v["target"] > 0
    ]
    m["max_climb_mps"] = abs(min(climb)) if climb else None  # NED neg = climb
    m["max_descend_mps"] = max(desc) if desc else None

    passed = len(failed) == 0
    return TestResult(
        name,
        passed,
        "PASS" if passed else "FAIL:" + ";".join(failed),
        metrics=m,
    )


def test_v4(
    bus: Bus, ctrl: Controller, safety: SafetyMonitor, log: list[TickLog]
) -> TestResult:
    """Soft landing from 5 m and 10 m — disarm on measured settle only."""
    name = "V4"
    results_m: dict = {}
    all_pass = True
    fail_reasons = []

    for alt_m in (5.0, 10.0):
        if not _prep(bus, ctrl, safety):
            return TestResult(name, False, "arm_failed")

        z_start = -alt_m
        ok, reason = _takeoff_to(bus, ctrl, safety, z_start, log, name)
        if not ok:
            return TestResult(name, False, reason, blowup=reason.startswith("blowup"))

        safety.expect_near_ground = True
        land_idx = len(log)
        # Descend ~1.5 m/s (NED +vz)
        sched = Schedule(
            [Segment(duration=60.0, z=None, vz=1.5, label=f"land_from_{alt_m}")]
        )

        touchdown_vz = None
        settle_t = None
        armed_at_settle = True

        def stop_on_settle(s: State, elapsed: float) -> bool:
            nonlocal touchdown_vz, settle_t, armed_at_settle
            # Near ground: alt < 0.35 m (z > -0.35)
            if _alt(s.pos[2]) > 0.4:
                return False
            if abs(s.vel[2]) < SETTLE_VZ:
                if settle_t is None:
                    settle_t = elapsed
                elif elapsed - settle_t >= SETTLE_HOLD_S:
                    touchdown_vz = s.vel[2]
                    armed_at_settle = s.armed
                    return True
            else:
                settle_t = None
            return False

        ok, reason = _run_loop(
            bus, ctrl, safety, sched, name, log, max_s=45.0, stop_fn=stop_on_settle
        )
        safety.expect_near_ground = False

        if not ok:
            return TestResult(name, False, reason, blowup=True)

        if touchdown_vz is None:
            all_pass = False
            fail_reasons.append(f"from_{alt_m}m:no_settle")
            results_m[f"from_{alt_m}m"] = {"settled": False}
            bus.disarm()
            continue

        # Disarm only on measured settle
        bus.disarm()
        time.sleep(0.2)
        s = bus.drain()
        td_ok = abs(touchdown_vz) < 2.0
        disarmed = not s.armed
        results_m[f"from_{alt_m}m"] = {
            "touchdown_vz": touchdown_vz,
            "td_ok": td_ok,
            "disarmed": disarmed,
            "settled": True,
        }
        if not td_ok:
            all_pass = False
            fail_reasons.append(f"from_{alt_m}m:vz={touchdown_vz:.2f}")
        if not disarmed:
            all_pass = False
            fail_reasons.append(f"from_{alt_m}m:still_armed")

        # Discard unused land_idx lint — kept for future windowed metrics
        _ = land_idx

    return TestResult(
        name,
        all_pass,
        "PASS" if all_pass else "FAIL:" + ";".join(fail_reasons),
        metrics=results_m,
    )


def test_tilt_check(
    bus: Bus, ctrl: Controller, safety: SafetyMonitor, log: list[TickLog]
) -> TestResult:
    """Scripted 8° lean hold — alt sag < 0.3 m (for pid_tilt methods)."""
    name = "TILT"
    if not _prep(bus, ctrl, safety):
        return TestResult(name, False, "arm_failed")

    z_hold = -3.0
    ok, reason = _takeoff_to(bus, ctrl, safety, z_hold, log, name)
    if not ok:
        return TestResult(name, False, reason, blowup=reason.startswith("blowup"))

    # Level hold 3 s to establish baseline alt
    base_idx = len(log)
    sched = Schedule([Segment(duration=3.0, z=z_hold, vz=0.0, label="pre_lean")])
    ok, reason = _run_loop(bus, ctrl, safety, sched, name, log)
    if not ok:
        return TestResult(name, False, reason, blowup=True)
    base_alt = M.mean([_alt(r.z) for r in log[base_idx:]])

    lean_idx = len(log)
    lean = lean_hold(z_hold, lean_rad=math.radians(8.0), duration=5.0, axis="roll")
    ok, reason = _run_loop(bus, ctrl, safety, lean, name, log)
    if not ok:
        return TestResult(name, False, reason, blowup=True)

    lean_alts = [_alt(r.z) for r in log[lean_idx:]]
    sag = base_alt - min(lean_alts) if lean_alts else 999.0
    m = {"alt_sag_m": sag, "base_alt": base_alt}
    passed = sag < 0.3
    return TestResult(
        name,
        passed,
        "PASS" if passed else f"FAIL:sag={sag:.3f}",
        metrics=m,
    )


TESTS = {
    "V1": test_v1,
    "V2": test_v2,
    "V3": test_v3,
    "V4": test_v4,
    "TILT": test_tilt_check,
}


def _write_log(path: Path, log: list[TickLog]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row.__dict__) + "\n")


def _write_report(
    path: Path,
    method: str,
    results: list[TestResult],
    run_dir: Path,
) -> str:
    lines = [
        "# Vertical harness report",
        "",
        f"- method: `{method}`",
        f"- run: `{run_dir}`",
        f"- utc: {_utc_stamp()}",
        f"- hover_default: {HOVER_THRUST}",
        "",
        "| Test | Result | Notes |",
        "|------|--------|-------|",
    ]
    for r in results:
        status = "PASS" if r.passed else ("BLOWUP" if r.blowup else "FAIL")
        notes = r.reason.replace("|", "/")
        lines.append(f"| {r.name} | {status} | {notes} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    for r in results:
        lines.append(f"### {r.name}")
        lines.append("```json")
        lines.append(json.dumps(r.metrics, indent=2, default=str))
        lines.append("```")
        lines.append("")
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def _print_table(results: list[TestResult]) -> None:
    print()
    print(f"{'TEST':<6} {'RESULT':<8} REASON")
    print("-" * 60)
    for r in results:
        status = "PASS" if r.passed else ("BLOWUP" if r.blowup else "FAIL")
        print(f"{r.name:<6} {status:<8} {r.reason}")
    print("-" * 60)
    # Highlight true hover if present
    for r in results:
        if r.name == "V1" and "true_hover_thrust" in r.metrics:
            print(f"TRUE HOVER THRUST = {r.metrics['true_hover_thrust']:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Vertical-axis flight harness")
    ap.add_argument(
        "--method",
        default="pd",
        choices=list_methods(),
        help="controller method",
    )
    ap.add_argument("--only", default=None, help="run a single test (e.g. V1)")
    ap.add_argument("--list", action="store_true", help="list tests and methods")
    ap.add_argument(
        "--skip-tilt",
        action="store_true",
        help="skip TILT check even for tilt methods",
    )
    args = ap.parse_args(argv)

    if args.list:
        print("methods:", ", ".join(list_methods()))
        print("tests:", ", ".join(TESTS))
        return 0

    run_dir = Path("runs") / f"vertical_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log: list[TickLog] = []

    print(f"[vertical] method={args.method} run_dir={run_dir}", flush=True)
    try:
        bus = Bus()
    except TimeoutError as e:
        print(str(e), flush=True)
        return 1
    if not bus.wait_for_pose(timeout_s=30.0):
        print(
            "no IMU/pose: stay in the TRAINING race world (not menu), "
            "then rerun. If imu_seen=0 above, click Race / GO first.",
            flush=True,
        )
        bus.close()
        return 1

    ctrl = make_controller(args.method)
    safety = SafetyMonitor()

    to_run = []
    if args.only:
        key = args.only.upper()
        if key not in TESTS:
            print(f"unknown test {args.only!r}; choose from {list(TESTS)}", flush=True)
            bus.close()
            return 1
        to_run = [key]
    else:
        to_run = ["V1", "V2", "V3", "V4"]
        if args.method.startswith("pid_tilt") and not args.skip_tilt:
            to_run.append("TILT")

    results: list[TestResult] = []
    try:
        for key in to_run:
            print(f"[vertical] === {key} ===", flush=True)
            # Fresh controller state each test
            ctrl = make_controller(args.method)
            res = TESTS[key](bus, ctrl, safety, log)
            results.append(res)
            print(f"[vertical] {key}: {res.reason}", flush=True)
            bus.disarm()
            time.sleep(0.5)
            if not res.blowup:
                bus.reset()
                time.sleep(2.0)
    finally:
        try:
            bus.disarm()
        except Exception:
            pass
        _write_log(run_dir / "log.jsonl", log)
        _write_report(run_dir / "report.md", args.method, results, run_dir)
        _print_table(results)
        print(f"\nreport → {run_dir / 'report.md'}", flush=True)
        bus.close()

    all_pass = all(r.passed for r in results) and len(results) > 0
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
