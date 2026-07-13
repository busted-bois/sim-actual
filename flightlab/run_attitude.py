"""Automated attitude-loop harness — run with sim in TRAINING session.

uv run python -m flightlab.run_attitude
uv run python -m flightlab.run_attitude --method PD --only B3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from flightlab.bus import CONTROL_DT, Bus
from flightlab.controllers import CONTROLLERS, make_controller
from flightlab.maneuvers import (
    HOVER_Z_NED,
    Phase,
    angle_step_phases,
    disturbance_phase,
    hover_jitter_phase,
    hover_target,
    rate_tracking_phase,
    sign_id_phases,
)
from flightlab.metrics import (
    SIGNS_PATH,
    analyze_jitter,
    analyze_step_response,
    identify_sign_from_pulse,
    imu_tilt_sign_check,
    load_signs,
    measure_latency_ms,
    rate_tracking_error,
    save_signs,
)
from flightlab.safety import HOVER_THRUST, SafetyMonitor
from flightlab.state import Cmd, Controller, State
from simulator.preflight import wait_for_race_go

ALL_TESTS = ("B0", "B1", "B2", "B3", "B4", "B5")


@dataclass
class TickLog:
    t: float
    test: str
    phase: str
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    cmd_rr: float
    cmd_pr: float
    cmd_yr: float
    thrust: float
    z: float


@dataclass
class TestResult:
    test_id: str
    passed: bool
    details: dict = field(default_factory=dict)
    fail_reason: str = ""


@dataclass
class RunContext:
    bus: Bus
    controller: Controller | None
    safety: SafetyMonitor
    run_dir: str
    log_f: object
    method_name: str = "PD"
    vq2_mode: bool = False
    hover_z: float = HOVER_Z_NED
    ticks: list[TickLog] = field(default_factory=list)
    t0: float = 0.0
    gain_scale: float = 1.0
    sign_accum: dict[str, float] = field(default_factory=dict)

    def log_tick(self, test: str, phase: str, s: State, cmd: Cmd) -> None:
        row = TickLog(
            t=s.t_mono - self.t0,
            test=test,
            phase=phase,
            roll_deg=math.degrees(s.roll),
            pitch_deg=math.degrees(s.pitch),
            yaw_deg=math.degrees(s.yaw),
            roll_rate=s.roll_rate,
            pitch_rate=s.pitch_rate,
            yaw_rate=s.yaw_rate,
            cmd_rr=cmd.roll_rate,
            cmd_pr=cmd.pitch_rate,
            cmd_yr=cmd.yaw_rate,
            thrust=cmd.thrust,
            z=s.pos_ned[2],
        )
        self.ticks.append(row)
        self.log_f.write(json.dumps(row.__dict__) + "\n")
        self.log_f.flush()


def _hover_thrust(s: State, z_tgt: float) -> float:
    z, vz = s.pos_ned[2], s.vel_ned[2]
    return float(max(0.18, min(0.5, HOVER_THRUST + 0.025 * (z - z_tgt) + 0.03 * vz)))


def _run_phases(
    ctx: RunContext,
    test_id: str,
    phases: list[Phase],
) -> tuple[bool, str]:
    """Execute phases; return (ok, fail_reason)."""
    if ctx.controller is not None:
        s0 = ctx.bus.snapshot()
        if s0 is not None:
            ctx.controller.reset(s0)

    for phase in phases:
        phase_end = time.monotonic() + phase.duration_s
        angle_before: dict[str, float] = {}
        if phase.name.endswith("_pulse"):
            s = ctx.bus.snapshot()
            if s:
                angle_before = {"roll": s.roll, "pitch": s.pitch, "yaw": s.yaw}

        while time.monotonic() < phase_end:
            s = ctx.bus.snapshot()
            if s is None:
                time.sleep(CONTROL_DT)
                continue
            if not ctx.bus.pose_usable(s):
                time.sleep(CONTROL_DT)
                continue

            saf = ctx.safety.check(s)
            if saf.tripped:
                _recover_blowup(ctx, saf.reason)
                return False, f"blowup:{saf.reason}"

            if phase.open_loop_rates is not None:
                rr, pr, yr = phase.open_loop_rates
                thrust = (
                    phase.open_loop_thrust
                    if phase.open_loop_thrust is not None
                    else _hover_thrust(s, phase.target.z)
                )
                cmd = Cmd(rr, pr, yr, thrust)
            elif ctx.controller is not None:
                cmd = ctx.controller.update(s, phase.target, CONTROL_DT)
            else:
                cmd = Cmd(0.0, 0.0, 0.0, _hover_thrust(s, phase.target.z))

            ctx.bus.send_cmd(cmd)
            ctx.log_tick(test_id, phase.name, s, cmd)
            time.sleep(CONTROL_DT)

        if phase.name.endswith("_pulse") and angle_before:
            s_after = ctx.bus.snapshot()
            if s_after:
                axis = phase.name.split("_")[1]
                before = angle_before[axis]
                after = getattr(s_after, axis)
                rate = (phase.open_loop_rates or (0, 0, 0))[
                    ["roll", "pitch", "yaw"].index(axis)
                ]
                sign = identify_sign_from_pulse(before, after, rate)
                ctx.sign_accum[axis] = sign

    return True, ""


def _recover_blowup(ctx: RunContext, reason: str) -> None:
    print(f"[harness] BLOW-UP: {reason} — level+hover 2s, disarm, reset", flush=True)
    t_end = time.monotonic() + 2.0
    cmd = Cmd(0.0, 0.0, 0.0, HOVER_THRUST)
    while time.monotonic() < t_end:
        s = ctx.bus.snapshot()
        if s:
            ctx.bus.send_cmd(cmd)
        time.sleep(CONTROL_DT)
    ctx.bus.disarm()
    time.sleep(0.5)
    ctx.bus.reset_sim()
    time.sleep(2.0)
    ctx.safety.reset()


def _prep_hover(
    ctx: RunContext, settle_s: float = 5.0, open_loop: bool = False
) -> bool:
    """Reset, wait for race GO, arm, climb to hover altitude using ODOMETRY.

    open_loop=True flies the prep with zero rate commands + alt-hold thrust
    only — no attitude feedback. B0 must use this: before B0 has measured the
    command signs, a closed-loop prep can be positive feedback on every axis
    (the old {-1,-1,-1} defaults nosed the drone into the gate base).
    """
    print("[harness] sim reset + wait for race GO ...", flush=True)
    ctx.bus.reset_sim()
    time.sleep(1.5)
    ctx.bus.arm()

    if not wait_for_race_go(ctx.bus.data, timeout_s=45.0):
        print("[harness] race GO timeout after reset", flush=True)
        return False

    if not ctx.bus.ensure_armed():
        print("[harness] arm failed", flush=True)
        return False

    s = ctx.bus.snapshot()
    if s is None or not ctx.bus.pose_usable(s):
        src = s.pose_source if s else "none"
        print(f"[harness] need usable pose, got {src}", flush=True)
        return False

    if not s.alt_trusted and not ctx.vq2_mode:
        # Without a trusted altitude the thrust law silently freezes at hover
        # and the drone sags into the ground — fail loudly instead.
        print(
            "[harness] no trusted altitude (odometry/LOCAL_POSITION_NED absent) "
            "— is the sim in a TRAINING session?",
            flush=True,
        )
        return False

    # NED: z more negative = higher. Climb toward 3 m unless already above.
    ctx.hover_z = min(s.pos_ned[2], HOVER_Z_NED)
    print(f"[harness] hover z_target={ctx.hover_z:.2f} m (NED)", flush=True)

    tgt = hover_target(ctx.hover_z)
    phases = [
        Phase(
            "prep_hover",
            settle_s,
            tgt,
            open_loop_rates=(0.0, 0.0, 0.0) if open_loop else None,
        )
    ]
    if ctx.controller is not None:
        ctx.controller.reset(s)
    # Re-anchor the safety grace NOW: flight starts here, and the race-GO wait
    # above may have consumed many seconds (the old reset-then-wait order let
    # the grace expire before the first tick, tripping alt<0.2m at spawn).
    ctx.safety.reset()
    ok, reason = _run_phases(ctx, "prep", phases)
    return ok and not reason


def test_b0(ctx: RunContext) -> TestResult:
    print("[B0] sign auto-ID ...", flush=True)
    # Open-loop prep: signs are unmeasured until B0's own pulses run, so any
    # closed-loop attitude feedback here could be positive feedback.
    if not _prep_hover(ctx, open_loop=True):
        return TestResult("B0", False, fail_reason="prep_failed")

    # The sign pulses themselves are open-loop (open_loop_rates on each pulse
    # phase); the settle phases between them use the controller, so detach it
    # for the whole B0 sequence and hover on thrust alone.
    saved_controller = ctx.controller
    ctx.controller = None
    ctx.sign_accum = {}
    try:
        ok, reason = _run_phases(ctx, "B0", sign_id_phases(ctx.hover_z))
    finally:
        ctx.controller = saved_controller
    if not ok:
        return TestResult("B0", False, fail_reason=reason)
    signs = ctx.sign_accum
    if len(signs) < 3:
        return TestResult("B0", False, fail_reason="incomplete_signs")
    out = {k: float(signs[k]) for k in ("roll", "pitch", "yaw")}
    save_signs(out)

    s = ctx.bus.snapshot()
    imu_check = {}
    if s:
        imu_check = imu_tilt_sign_check(s.roll, s.pitch, s.gravity_body)

    passed = all(k in out for k in ("roll", "pitch", "yaw"))
    return TestResult(
        "B0",
        passed,
        {"signs": out, "imu_crosscheck": imu_check},
        "" if passed else "sign_id_failed",
    )


def test_b1(ctx: RunContext) -> TestResult:
    print("[B1] rate tracking + latency ...", flush=True)
    if not _prep_hover(ctx):
        return TestResult("B1", False, fail_reason="prep_failed")

    t_start = len(ctx.ticks)
    ok, reason = _run_phases(ctx, "B1", rate_tracking_phase(ctx.hover_z))
    if not ok:
        return TestResult("B1", False, fail_reason=reason)

    times, cmds, rates = [], [], []
    for row in ctx.ticks[t_start:]:
        if row.phase == "rate_roll_cmd":
            times.append(row.t)
            cmds.append(row.cmd_rr)
            rates.append(row.roll_rate)

    latency = measure_latency_ms(times, cmds, rates)
    mean_err = 0.0
    n = 0
    for row in ctx.ticks[t_start:]:
        if row.phase == "rate_roll_cmd" and abs(row.cmd_rr) > 0.1:
            mean_err += rate_tracking_error(row.cmd_rr, row.roll_rate)
            n += 1
    mean_err = mean_err / n if n else 1.0

    passed = mean_err <= 0.20 and (latency is None or latency < 100.0)
    return TestResult(
        "B1",
        passed,
        {
            "latency_ms": latency,
            "mean_rate_error_frac": mean_err,
            "cmd_rad_s": 0.3,
        },
        "" if passed else f"rate_err={mean_err:.2f} latency={latency}",
    )


def test_b2(ctx: RunContext, axis: str = "roll") -> TestResult:
    print(f"[B2] angle step ({axis}) ...", flush=True)
    if not _prep_hover(ctx):
        return TestResult("B2", False, fail_reason="prep_failed")

    t_start = len(ctx.ticks)
    ok, reason = _run_phases(ctx, "B2", angle_step_phases(axis, z=ctx.hover_z))
    if not ok:
        return TestResult("B2", False, fail_reason=reason)

    times, angles = [], []
    step_seen = False
    for row in ctx.ticks[t_start:]:
        if row.phase.startswith("step_"):
            step_seen = True
        if step_seen:
            times.append(row.t)
            angles.append(row.roll_deg if axis == "roll" else row.pitch_deg)

    y0 = angles[0] if angles else 0.0
    target = y0 + (8.0 if axis == "roll" else 8.0)
    m = analyze_step_response(times, angles, target)

    passed = (
        m.rise_s < 0.4
        and m.overshoot_pct < 20.0
        and m.settle_s < 1.2
        and (m.damped or len(m.peaks) <= 1)
    )
    return TestResult(
        f"B2_{axis}",
        passed,
        {
            "rise_s": m.rise_s,
            "overshoot_pct": m.overshoot_pct,
            "settle_s": m.settle_s,
            "damped": m.damped,
            "peaks": m.peaks,
        },
        "" if passed else "step_criteria_failed",
    )


def test_b3(ctx: RunContext) -> TestResult:
    print("[B3] hover jitter ...", flush=True)
    if not _prep_hover(ctx):
        return TestResult("B3", False, fail_reason="prep_failed")

    t_start = len(ctx.ticks)
    ok, reason = _run_phases(ctx, "B3", hover_jitter_phase(20.0, z=ctx.hover_z))
    if not ok:
        return TestResult("B3", False, fail_reason=reason)

    rolls, pitches, rate_cmds = [], [], []
    for row in ctx.ticks[t_start:]:
        if row.phase == "hover_jitter":
            rolls.append(row.roll_deg)
            pitches.append(row.pitch_deg)
            rate_cmds.append(math.hypot(row.cmd_rr, row.cmd_pr))

    m = analyze_jitter(rolls, pitches, rate_cmds, 1.0 / CONTROL_DT)
    passed = (
        m.roll_std_deg < 1.0
        and m.pitch_std_deg < 1.0
        and m.roll_ptp_deg < 3.0
        and m.pitch_ptp_deg < 3.0
        and m.rate_cmd_std < 0.08
        and m.psd_pass
    )
    return TestResult(
        "B3",
        passed,
        {
            "roll_std_deg": m.roll_std_deg,
            "pitch_std_deg": m.pitch_std_deg,
            "roll_ptp_deg": m.roll_ptp_deg,
            "pitch_ptp_deg": m.pitch_ptp_deg,
            "rate_cmd_std": m.rate_cmd_std,
            "psd_peak_db": m.psd_peak_db,
            "psd_pass": m.psd_pass,
        },
        "" if passed else "jitter_criteria_failed",
    )


def test_b4(ctx: RunContext) -> TestResult:
    print("[B4] stability margin ...", flush=True)
    scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    last_stable = 1.0
    onset = None
    details: dict = {"runs": []}

    for sc in scales:
        ctx.controller = make_controller(
            ctx.method_name, gain_scale=sc, vq2=ctx.vq2_mode
        )
        r2 = test_b2(ctx, "roll")
        r3 = test_b3(ctx)
        oscillating = not r2.passed and r2.details.get("damped") is False
        entry = {
            "gain_scale": sc,
            "B2": r2.passed,
            "B3": r3.passed,
            "oscillating": oscillating,
        }
        details["runs"].append(entry)
        if r2.passed and r3.passed:
            last_stable = sc
        elif onset is None and (oscillating or not r2.passed or not r3.passed):
            onset = sc
            break

    if onset is None:
        onset = scales[-1]

    passed = last_stable >= 1.0
    details["last_stable_gain"] = last_stable
    details["onset_gain"] = onset
    return TestResult("B4", passed, details, "" if passed else "no_stable_margin")


def test_b5(ctx: RunContext) -> TestResult:
    print("[B5] disturbance recovery ...", flush=True)
    if not _prep_hover(ctx):
        return TestResult("B5", False, fail_reason="prep_failed")

    t_start = len(ctx.ticks)
    ok, reason = _run_phases(ctx, "B5", disturbance_phase(ctx.hover_z))
    if not ok:
        return TestResult("B5", False, fail_reason=reason)

    recover_start = None
    settled = None
    for row in ctx.ticks[t_start:]:
        if row.phase == "recover":
            if recover_start is None:
                recover_start = row.t
            if abs(row.roll_deg) < 1.0 and abs(row.pitch_deg) < 1.0:
                if settled is None:
                    settled = row.t
    settle_s = (
        (settled - recover_start) if (settled and recover_start) else float("inf")
    )
    passed = settle_s < 1.5
    return TestResult(
        "B5",
        passed,
        {"settle_s": settle_s},
        "" if passed else f"settle_s={settle_s:.2f}",
    )


def _print_table(results: list[TestResult]) -> None:
    print("\n=== ATTITUDE HARNESS RESULTS ===", flush=True)
    print(f"{'TEST':<8} {'PASS':<6} DETAILS", flush=True)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = r.fail_reason or json.dumps(r.details, default=str)[:60]
        print(f"{r.test_id:<8} {status:<6} {detail}", flush=True)


def _write_report(
    path: str, method: str, results: list[TestResult], signs: dict
) -> None:
    lines = [
        "# Attitude Harness Report",
        "",
        f"Method: **{method}**",
        f"Signs: `{json.dumps(signs)}`",
        "",
        "## Results",
        "",
        "| Test | Pass | Details |",
        "|------|------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r.test_id} | {'PASS' if r.passed else 'FAIL'} | "
            f"{r.fail_reason or json.dumps(r.details, default=str)[:80]} |"
        )

    b1 = next((r for r in results if r.test_id == "B1"), None)
    if b1 and b1.details.get("latency_ms"):
        lines += [
            "",
            f"**Latency (RL action-delay):** {b1.details['latency_ms']:.1f} ms",
        ]

    b4 = next((r for r in results if r.test_id == "B4"), None)
    if b4:
        lines += [
            "",
            f"**Stable gain:** {b4.details.get('last_stable_gain')}",
            f"**Onset gain:** {b4.details.get('onset_gain')}",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Attitude-loop automated harness")
    ap.add_argument("--method", default="PD", choices=list(CONTROLLERS))
    ap.add_argument("--only", nargs="+", choices=ALL_TESTS, help="run subset")
    ap.add_argument("--list", action="store_true", help="list methods and tests")
    args = ap.parse_args()

    if args.list:
        print("Methods:", ", ".join(CONTROLLERS))
        print("Tests:", ", ".join(ALL_TESTS))
        return

    utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"attitude_{utc}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.jsonl")
    report_path = os.path.join(run_dir, "report.md")

    bus = Bus()
    pose_mode = bus.wait_for_flight_ready(timeout_s=90.0)
    if not pose_mode:
        print(
            "ERROR: no flight pose within 90 s.\n"
            "1. `make free-port`  2. FlightSim → TRAINING/SUBMISSION session\n"
            "3. Run harness  4. Click Race while it waits\n"
            f"Received:\n{bus.pose_diagnostic()}",
            flush=True,
        )
        bus.close()
        sys.exit(1)

    vq2 = pose_mode == "vq2"
    method = args.method
    if vq2:
        print("[harness] VQ2 mode — EKF pose fallback", flush=True)

    controller = make_controller(method, vq2=vq2)
    safety = SafetyMonitor()
    with open(log_path, "w", encoding="utf-8") as log_f:
        ctx = RunContext(
            bus=bus,
            controller=controller,
            safety=safety,
            run_dir=run_dir,
            log_f=log_f,
            method_name=method,
            vq2_mode=vq2,
            t0=time.monotonic(),
        )

        tests_to_run = list(args.only) if args.only else list(ALL_TESTS)
        results: list[TestResult] = []
        b0_passed = "B0" not in tests_to_run and os.path.isfile(SIGNS_PATH)

        for tid in tests_to_run:
            if tid != "B0" and not b0_passed:
                results.append(TestResult(tid, False, fail_reason="B0_gate"))
                continue
            if tid == "B0":
                r = test_b0(ctx)
                results.append(r)
                b0_passed = r.passed
                if r.passed:
                    ctx.controller = make_controller(method, vq2=ctx.vq2_mode)
            elif tid == "B2":
                r_roll = test_b2(ctx, "roll")
                r_pitch = test_b2(ctx, "pitch")
                results.append(
                    TestResult(
                        "B2",
                        r_roll.passed and r_pitch.passed,
                        {
                            "roll": r_roll.details,
                            "pitch": r_pitch.details,
                        },
                        "" if (r_roll.passed and r_pitch.passed) else "step_failed",
                    )
                )
            elif tid == "B1":
                results.append(test_b1(ctx))
            elif tid == "B3":
                results.append(test_b3(ctx))
            elif tid == "B4":
                results.append(test_b4(ctx))
            elif tid == "B5":
                results.append(test_b5(ctx))

        signs = load_signs(vq2=vq2)
        _print_table(results)
        _write_report(report_path, method, results, signs)
        print(f"\n[log] {log_path}", flush=True)
        print(f"[report] {report_path}", flush=True)

    bus.close()
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
