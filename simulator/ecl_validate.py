"""VQ1 ground-truth validation harness for the ecl/EKF2 state estimator.

VQ1 re-exposes telemetry (LOCAL_POSITION_NED velocity + ATTITUDE), so we finally
have a TRUTH signal to check the estimator against. This flies the existing GP
pilot on the VQ1 sim, runs `EclEkf` as a live SHADOW fed the same IMU +
de-rotated PnP vision velocity the deploy path would use, and every tick logs
three velocity series to JSONL:

    * EKF-fused        (simulator.ecl_ekf.EclEkf)      -- what we're validating
    * IMU dead-reckon  (the pilot's GPEstimation)      -- the baseline it must beat
    * TRUTH            (LOCAL_POSITION_NED / ATTITUDE)  -- ground truth

On exit it prints an RMSE report: does the EKF track truth and stay BOUNDED
where dead-reckon diverges? Whatever it shows on VQ1 transfers to VQ2 (same
physics, telemetry just hidden).

Nothing here is built from scratch -- it reuses setup_components (sim/RX/vision),
GPEstimation (baseline), best_pose_gate + VisionVelocityTracker (vision), EclEkf
(estimator), and setup._request_data_streams (to make VQ1 stream truth).

Run against the VQ1 sim (telemetry ON):
    make est-validate
    # or re-report an existing log without flying:
    uv run -m simulator.ecl_validate --report logs/ecl_validate_<boot>.jsonl
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import traceback

import numpy as np

from simulator import setup as setup_mod
from simulator.ecl_ekf import EclEkf
from simulator.gp_vision import VisionVelocityTracker, _yolo_pose_estimate
from simulator.setup import setup_components

# Sensor conventions, matching the validated gp_estimation.py strapdown:
#   accel = specific force, used raw (+1); gyro is inverted vs NED (-1).
# Exposed as env in case VQ1 truth shows a flipped attitude -> flip and re-fly.
ACC_SIGN = float(os.environ.get("ECL_ACC_SIGN", "1.0"))
GYRO_SIGN = float(os.environ.get("ECL_GYRO_SIGN", "-1.0"))
# Which body axes of PnP vision velocity to fuse as EV. Default all three so the
# truth comparison reveals whether forward PnP velocity is usable (the pilot
# currently trusts only lateral+vertical). Set ECL_EV_AXES=yz to match deploy.
EV_AXES = os.environ.get("ECL_EV_AXES", "xyz").lower()
EV_STD = float(os.environ.get("ECL_EV_NOISE", "0.3"))  # m/s; var = std^2
EV_VAR = EV_STD * EV_STD

SHADOW_POLL_HZ = 400
LOG_HZ = 30
G = 9.81


def _quat_to_euler(w, x, y, z):
    """(w,x,y,z) body->NED quaternion -> (roll, pitch, yaw) radians."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class EclShadow:
    """Live EKF shadow: consumes shared_data['imu'] + PnP vision velocity on its
    own thread (mirrors GPEstimation's IMU loop) and exposes a fused snapshot.

    All EclEkf calls happen on this one thread, so no locking of the filter is
    needed; only the published snapshot is lock-guarded for the logger."""

    def __init__(self, data: dict, gyro_sign: float = GYRO_SIGN, acc_sign=None, record=False):
        self.data = data
        self._gsign = gyro_sign
        # Per-axis accel sign; VQ1 flips forward (ax). Default isotropic ACC_SIGN.
        self._asign = acc_sign if acc_sign is not None else (ACC_SIGN, ACC_SIGN, ACC_SIGN)
        # Full-rate raw-IMU + truth capture for offline sign calibration.
        self._record = record
        self._raw: list = []
        self.ekf = EclEkf()
        self.tracker = VisionVelocityTracker()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_imu_ts: int | None = None
        self._last_vis_fid: int | None = None
        self._last_gz = 0.0  # rad/s, signed as fed to EKF (for de-rotation)
        self._ekf_time_us = 0  # monotonic clock fed to the EKF (reset-proof)
        self._in_air = False   # latches true on first real motion
        self._seen_still = False
        self._t0 = time.time()
        self._prev_status = (False, False, False)
        self._snap = {
            "vel_ned": (0.0, 0.0, 0.0),
            "euler": (0.0, 0.0, 0.0),
            "status": (False, False, False),  # tilt_align, yaw_align, ev_vel
            "valid": False,
            "ev_pushed": False,
        }

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ecl-shadow")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        interval = 1.0 / SHADOW_POLL_HZ
        last_tb = 0.0
        while self._running:
            try:
                self._step()
            except Exception:
                now = time.monotonic()
                if now - last_tb >= 5.0:
                    traceback.print_exc()
                    last_tb = now
            time.sleep(interval)

    def _step(self) -> None:
        imu = self.data.get("imu")
        if imu is None:
            return
        ts_us = int(imu.get("time_us") or imu.get("time_usec") or 0)
        if self._last_imu_ts is None:
            self._last_imu_ts = ts_us
            return
        if ts_us == self._last_imu_ts:
            return
        # Physical dt from the sim clock; abs() + clamp survives the backward jump
        # when the sim clock resets between races.
        dt = max(0.0005, min(0.1, abs(ts_us - self._last_imu_ts) * 1e-6))
        self._last_imu_ts = ts_us
        # Feed the EKF a MONOTONIC clock so a sim clock reset can't corrupt its
        # time-ordered ring buffers.
        self._ekf_time_us += int(dt * 1e6)
        et = self._ekf_time_us

        rax = float(imu.get("ax", imu.get("xacc", 0.0)))
        ray = float(imu.get("ay", imu.get("yacc", 0.0)))
        raz = float(imu.get("az", imu.get("zacc", 0.0)))
        rgx = float(imu.get("gx", imu.get("xgyro", 0.0)))
        rgy = float(imu.get("gy", imu.get("ygyro", 0.0)))
        rgz = float(imu.get("gz", imu.get("zgyro", 0.0)))

        # Keep the EKF at-rest (ZUPT tilt alignment) for the ENTIRE true stationary
        # hold, then latch in_air at launch and never go back. Aligning tilt while
        # the drone is genuinely level+still is what makes tilt -- and thus the
        # velocity -- correct; aligning late (mid-acceleration) gave the 44-deg
        # pitch error. On VQ1 the truest launch signal is ground-truth SPEED; fall
        # back to a gyro spike when truth is absent (VQ2 deploy).
        if not self._in_air:
            # Detect launch from the GYRO (live IMU) so a frozen/stale position
            # feed can't wrongly report "already moving" and skip ZUPT alignment.
            # Truth speed only helps as a backup once we've actually seen it low
            # (guards against a stale constant like 10.5 m/s at rest).
            g_norm = math.sqrt(rgx * rgx + rgy * rgy + rgz * rgz)
            lp = self.data.get("local_position_ned") or self.data.get("odometry")
            sp = None
            if lp is not None:
                sp = math.sqrt(float(lp.get("vx", 0.0)) ** 2
                               + float(lp.get("vy", 0.0)) ** 2
                               + float(lp.get("vz", 0.0)) ** 2)
                if sp < 0.5:
                    self._seen_still = True
            launched = g_norm > 0.8 or (sp is not None and self._seen_still and sp > 2.0)
            if launched:
                self._in_air = True
                self.ekf.set_at_rest(False)
                self.ekf.set_in_air(True)
            else:
                self.ekf.set_at_rest(True)

        # Record RAW imu + truth at full IMU rate so --calibrate can replay
        # faithfully (alignment needs full rate; the 30 Hz JSONL is too coarse).
        if self._record:
            lp = self.data.get("local_position_ned") or self.data.get("odometry")
            od = self.data.get("odometry")
            self._raw.append({
                "et": et, "dt": dt,
                "a": [rax, ray, raz], "g": [rgx, rgy, rgz],
                "tv": [float(lp["vx"]), float(lp["vy"]), float(lp["vz"])] if lp else None,
                "tq": [float(od["qw"]), float(od["qx"]), float(od["qy"]), float(od["qz"])] if od else None,
            })

        ax, ay, az = self._asign[0] * rax, self._asign[1] * ray, self._asign[2] * raz
        gx, gy, gz = self._gsign * rgx, self._gsign * rgy, self._gsign * rgz
        self._last_gz = gz
        self.ekf.push_imu(ax, ay, az, gx, gy, gz, dt, et)

        # A vision hiccup must NEVER stall the filter: push EV if we can, but
        # always run update() so tilt keeps aligning and IMU keeps propagating.
        try:
            ev_pushed = self._maybe_push_vision(et)
        except Exception:
            ev_pushed = False
        self.ekf.update()

        vn = self.ekf.velocity_ned()
        qw, qx, qy, qz = self.ekf.quaternion()
        status = self.ekf.status()
        # Announce alignment milestones so the operator knows when it's safe to
        # start the race (tilt must align first, then EV velocity can engage).
        if status[0] and not self._prev_status[0]:
            print(f"[ecl] >>> TILT ALIGNED at t={time.time()-self._t0:.1f}s "
                  f"-- safe to START THE RACE now <<<", flush=True)
        if status[2] and not self._prev_status[2]:
            print(f"[ecl] EV velocity fusion engaged at t={time.time()-self._t0:.1f}s", flush=True)
        self._prev_status = status
        with self._lock:
            self._snap = {
                "vel_ned": vn,
                "euler": _quat_to_euler(qw, qx, qy, qz),
                "status": status,
                "valid": self.ekf.valid(),
                "ev_pushed": ev_pushed,
            }

    def _maybe_push_vision(self, ts_us: int) -> bool:
        est = _yolo_pose_estimate(self.data)  # dict with body_x_m/frame_id (or None)
        vel = self.tracker.update(est)
        if vel is None or est is None:
            return False
        fid = est.get("frame_id")
        if fid is not None and fid == self._last_vis_fid:
            return False
        self._last_vis_fid = fid
        bx = float(est.get("body_x_m", 0.0))
        vx = float(vel["vx_body_mps"])
        vy = float(vel["vy_body_mps"])
        vz = float(vel["vz_body_mps"])
        # De-rotate lateral: a yaw sweeps the gate across the image and fakes a
        # sideways velocity (v_true = v - omega_z * bx). Sign-safe like the pilot:
        # only take it when it SHRINKS |vy| (removes a real phantom).
        vy_derot = vy - self._last_gz * bx
        if abs(vy_derot) <= abs(vy):
            vy = vy_derot
        # Suppress an axis both ways: huge variance so post-engage FUSION ignores
        # it (IMU carries that axis), AND value 0 so the one-shot velocity RESET
        # at EV-engage (ev_vel.h resetVelocityTo, which uses the raw value
        # regardless of variance) seeds 0 rather than a distrusted vision value.
        # The drone is stationary at engage, so 0 is the right seed.
        big = 1e6
        ox, vx_v = (vx, EV_VAR) if "x" in EV_AXES else (0.0, big)
        oy, vy_v = (vy, EV_VAR) if "y" in EV_AXES else (0.0, big)
        oz, vz_v = (vz, EV_VAR) if "z" in EV_AXES else (0.0, big)
        self.ekf.push_vision_velocity(ox, oy, oz, (vx_v, vy_v, vz_v), ts_us)
        return True

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def dump_raw(self, path: str) -> int:
        with open(path, "w") as fh:
            for row in self._raw:
                fh.write(json.dumps(row) + "\n")
        return len(self._raw)


def _truth(data: dict):
    """(vel_ned tuple or None, (roll,pitch,yaw) or None) from VQ1 telemetry."""
    lp = data.get("local_position_ned") or data.get("odometry")
    vel = None
    if lp is not None:
        vel = (float(lp["vx"]), float(lp["vy"]), float(lp["vz"]))
    att = data.get("attitude")
    euler = None
    if att is not None:
        euler = (float(att["roll"]), float(att["pitch"]), float(att["yaw"]))
    elif data.get("odometry") is not None:
        o = data["odometry"]
        euler = _quat_to_euler(o["qw"], o["qx"], o["qy"], o["qz"])
    return vel, euler


def _fit_convention(rows: list[dict]) -> None:
    """Fit raw gyro against TRUE body rates per axis to recover the VQ1 gyro
    sign/scale. GPEstimation and the EKF feed (-1 * raw_gyro) as the body rate;
    that is correct only if raw_gyro ~= -1 * true_rate (i.e. fitted k ~= -1)."""
    axes = ["x/roll", "y/pitch", "z/yaw"]
    pairs = [[], [], []]
    for r in rows:
        g, a = r.get("gyro_raw"), r.get("att_rate")
        if not g or not a:
            continue
        for i in range(3):
            if g[i] is not None and a[i] is not None:
                pairs[i].append((a[i], g[i]))  # (true_rate, raw_gyro)
    if not any(pairs):
        return
    print("\n  GYRO CONVENTION vs true body rates (raw_gyro = k * true_rate):")
    print(f"    {'axis':<9}{'k (sign*scale)':>16}{'R^2':>8}{'n_moving':>10}   feed multiplier")
    current = GYRO_SIGN  # what GPEstimation(-1) and the shadow feed
    worst_ok = True
    for i, name in enumerate(axes):
        mv = [(a, g) for (a, g) in pairs[i] if abs(a) > 0.1]  # need real motion
        if len(mv) < 5:
            print(f"    {name:<9}{'(no motion)':>16}{'':>8}{len(mv):>10}")
            continue
        sa = sum(a * a for a, _ in mv)
        sag = sum(a * g for a, g in mv)
        k = sag / sa if sa > 0 else float("nan")
        # R^2 of g ~= k*a
        gm = sum(g for _, g in mv) / len(mv)
        ss_tot = sum((g - gm) ** 2 for _, g in mv) or 1e-9
        ss_res = sum((g - k * a) ** 2 for a, g in mv)
        r2 = 1 - ss_res / ss_tot
        want_mult = (1.0 / k) if abs(k) > 1e-6 else float("nan")  # raw->true_rate
        ok = abs(want_mult - current) < 0.25 * max(1.0, abs(current))
        worst_ok = worst_ok and ok
        flag = "OK" if ok else f"<-- feed {want_mult:+.2f}, not {current:+.0f}"
        print(f"    {name:<9}{k:>16.3f}{r2:>8.2f}{len(mv):>10}   {want_mult:+.2f}  {flag}")
    if not worst_ok:
        print("    => VQ1 gyro differs from the current -1 convention. This is WHY")
        print("       GPEstimation (and the pilot) diverged. Set GPEstimation gyro")
        print("       handling + ECL_GYRO_SIGN/scale to the 'feed multiplier' above.")
    else:
        print("    => gyro convention matches -1; divergence is elsewhere.")


def report(rows: list[dict]) -> None:
    """Print the RMSE comparison: EKF vs truth vs dead-reckon."""
    has_truth = [r for r in rows if r.get("truth_vel")]
    valid = [r for r in has_truth if r.get("valid")]
    tilt_row = next((r for r in rows if r.get("status") and r["status"][0]), None)
    ev_any = sum(1 for r in rows if r.get("ev_pushed"))
    print("\n" + "=" * 68)
    print("  ecl/EKF2 vs VQ1 GROUND TRUTH")
    print("=" * 68)
    print(f"  samples:   {len(rows)} logged")
    print(f"  truth:     {len(has_truth)} with LOCAL_POSITION_NED/ATTITUDE  "
          f"{'<-- 0 means VQ1 telemetry never streamed' if not has_truth else ''}")
    print(f"  EKF valid: {len(valid)} (tilt-aligned)  "
          f"{'tilt at t=%.1fs' % tilt_row['t'] if tilt_row else '<-- tilt NEVER aligned (ZUPT should do it in ~1s at rest)'}")
    print(f"  EV pushed: {ev_any} vision-velocity samples")

    _fit_convention(rows)

    # Accel at rest: |a| should be ~9.81 and az carries the gravity sign. This is
    # what tilt alignment locks onto, so a wrong scale/sign here => wrong attitude.
    rest = [r for r in rows if r.get("accel_raw") and not r.get("in_air")]
    if rest:
        anorm = [math.sqrt(sum(c * c for c in r["accel_raw"])) for r in rest]
        mn = sum(anorm) / len(anorm)
        mz = sum(r["accel_raw"][2] for r in rest) / len(rest)
        print(f"\n  ACCEL AT REST ({len(rest)} samples): |a|={mn:.2f} (expect ~9.81)  "
              f"az={mz:+.2f}")
        if abs(mn - 9.81) > 1.5:
            print("    !! |a| far from 9.81 -> accel scale differs on VQ1; tilt align unreliable")

    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2, axis=0))

    # Dead-reckon vs truth needs NO EKF validity -- always show this contrast.
    if has_truth:
        tv_all = np.array([r["truth_vel"] for r in has_truth])
        dv_all = np.array([r["dead_vel_ned"] for r in has_truth])
        d_all_rmse = rmse(dv_all, tv_all)
        d_all_max = np.max(np.abs(dv_all - tv_all), axis=0)
        print("\n  IMU DEAD-RECKON vs truth over ALL truth samples (NED m/s):")
        print(f"    rmse N/E/D: {d_all_rmse[0]:.2f}/{d_all_rmse[1]:.2f}/{d_all_rmse[2]:.2f}"
              f"   max: {d_all_max.max():.2f}   (this is the baseline the EKF must beat)")

    if not valid:
        print("\n  No EKF-valid samples yet -- can't score the fused estimate.")
        if not has_truth:
            print("  FIX: confirm you're on the VQ1 (telemetry) sim, not VQ2.")
        else:
            print("  FIX: tilt should ZUPT-align in ~1s while stationary -- if it")
            print("       didn't, check ACCEL AT REST above (bad |a| blocks alignment).")
        print("=" * 68 + "\n")
        return

    tv = np.array([r["truth_vel"] for r in valid])
    ev = np.array([r["ekf_vel_ned"] for r in valid])
    dv = np.array([r["dead_vel_ned"] for r in valid])

    e_rmse, d_rmse = rmse(ev, tv), rmse(dv, tv)
    e_max = np.max(np.abs(ev - tv), axis=0)
    d_max = np.max(np.abs(dv - tv), axis=0)

    print("\n  VELOCITY error vs truth (NED, m/s):")
    print(f"    {'axis':<6}{'EKF rmse':>10}{'EKF max':>10}{'dead rmse':>12}{'dead max':>10}")
    for i, ax in enumerate("NED"):
        print(f"    {ax:<6}{e_rmse[i]:>10.3f}{e_max[i]:>10.3f}{d_rmse[i]:>12.3f}{d_max[i]:>10.3f}")
    e_speed = np.sqrt(np.sum(e_rmse ** 2))
    d_speed = np.sqrt(np.sum(d_rmse ** 2))
    print(f"    {'|v|':<6}{e_speed:>10.3f}{'':>10}{d_speed:>12.3f}")

    # Attitude (roll/pitch absolute; yaw reference-relative -> report drift).
    at = [r for r in valid if r.get("truth_att")]
    if at:
        ee = np.array([r["ekf_euler"] for r in at])
        tt = np.array([r["truth_att"] for r in at])
        rp_err = np.degrees(np.mean(np.abs(ee[:, :2] - tt[:, :2]), axis=0))
        yaw0 = ee[0, 2] - tt[0, 2]
        yaw_drift = np.degrees(np.abs(((ee[-1, 2] - tt[-1, 2]) - yaw0)))
        print("\n  ATTITUDE error vs truth (deg):")
        print(f"    roll {rp_err[0]:.2f}   pitch {rp_err[1]:.2f}   yaw drift {yaw_drift:.2f} (mag-free, reference-relative)")

    # Alignment timing.
    tilt_row = next((r for r in rows if r.get("status") and r["status"][0]), None)
    ev_frac = np.mean([1.0 if (r.get("status") and r["status"][2]) else 0.0 for r in valid])
    if tilt_row:
        print(f"\n  tilt_align at t={tilt_row['t']:.1f}s   ev_vel engaged {ev_frac*100:.0f}% of valid samples")

    print("\n  VERDICT:")
    pass_bounded = e_max.max() < max(2.0, 0.5 * d_max.max())
    pass_track = e_speed < 0.8
    print(f"    EKF stays bounded (EKF max {e_max.max():.2f} << dead max {d_max.max():.2f}): "
          f"{'PASS' if pass_bounded else 'CHECK'}")
    print(f"    EKF tracks truth (|v| rmse {e_speed:.2f} < 0.8): {'PASS' if pass_track else 'CHECK'}")
    print("    (dead-reckon SHOULD diverge -- that contrast is the whole point.)")
    print("=" * 68 + "\n")


def run_live() -> None:
    os.environ.setdefault("AUTO_PILOT", "gp")
    os.environ.setdefault("GP_DISPLAY", "0")  # no vision window; keep it lean

    # VQ1 uses the opposite gyro sign to VQ2 (measured +1). --vq1 applies it to
    # BOTH the EKF shadow AND the pilot's GPEstimation (via env, set before the
    # pilot is constructed in setup_components) so the pilot flies straight.
    vq1 = "--vq1" in sys.argv
    gyro_sign = GYRO_SIGN
    acc_sign = None
    if vq1:
        # VQ1 gyro matched ATTITUDE rates at +1; accel is left RAW (per-axis
        # flipping the accel reflects the gravity vector and breaks the filter --
        # confirmed catastrophic). The exact FRD-consistent signs are pinned down
        # by --calibrate on a full-rate capture; this is the least-bad config that
        # aligns, good enough to record from.
        gyro_sign = 1.0
        acc_sign = (1.0, 1.0, 1.0)
        os.environ["GP_GYRO_SIGN"] = "1.0"
        os.environ["GP_ACC_SIGN"] = "1,1,1"
        print("[ecl] --vq1: gyro +1, accel raw. Recording full-rate raw+truth for --calibrate.", flush=True)

    boot_ms = int(time.time() * 1000)
    shared = {}
    try:
        comps = setup_components(shared, boot_ms, "127.0.0.1", 14550)
    except TimeoutError as exc:
        print(f"ERROR: {exc}", flush=True)
        sys.exit(1)

    # VQ1: explicitly request the truth streams (auto path only asks for IMU).
    setup_mod._request_data_streams(comps["sim_conn"], rate_hz=50)

    controller = comps["controller"]
    pilot = getattr(controller, "pilot", None)

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(logs_dir, f"ecl_validate_{boot_ms}.jsonl"))
    print(f"Logging to {log_path}", flush=True)

    shadow = EclShadow(shared, gyro_sign=gyro_sign, acc_sign=acc_sign, record=True)
    raw_path = os.path.abspath(os.path.join(logs_dir, f"ecl_raw_{boot_ms}.jsonl"))
    shadow.start()  # start now so tilt aligns during the pre-arm/hold
    print("EKF shadow started (ZUPT aligns tilt in ~1s while stationary)...", flush=True)
    print("  >>> Ideally START THE RACE only after '[ecl] TILT ALIGNED' (usually <2s). <<<", flush=True)

    print("Arming...", flush=True)
    controller.arm()
    print("Flying GP pilot + EKF shadow. Ctrl+C to stop and report.", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    next_log = 0.0
    next_status = 0.0
    last_tb = 0.0
    fh = open(log_path, "w")
    try:
        while True:
            try:
                controller.update()
            except Exception:
                now = time.monotonic()
                if now - last_tb >= 5.0:
                    traceback.print_exc()
                    last_tb = now
                time.sleep(1.0 / 90.0)
            now = time.time()
            if now >= next_log:
                next_log = now + 1.0 / LOG_HZ
                snap = shadow.snapshot()
                tvel, tatt = _truth(shared)
                # Raw gyro vs TRUE body rates -> lets the report fit the VQ1 gyro
                # sign/scale convention (the thing that made GPEstimation and thus
                # the pilot diverge). ATTITUDE.*speed are true body rates (rad/s).
                imu_now = shared.get("imu") or {}
                att_now = shared.get("attitude") or {}
                gyro_raw = None
                accel_raw = None
                if imu_now:
                    gyro_raw = [
                        float(imu_now.get("gx", imu_now.get("xgyro", 0.0))),
                        float(imu_now.get("gy", imu_now.get("ygyro", 0.0))),
                        float(imu_now.get("gz", imu_now.get("zgyro", 0.0))),
                    ]
                    accel_raw = [
                        float(imu_now.get("ax", imu_now.get("xacc", 0.0))),
                        float(imu_now.get("ay", imu_now.get("yacc", 0.0))),
                        float(imu_now.get("az", imu_now.get("zacc", 0.0))),
                    ]
                att_rate = None
                if att_now:
                    att_rate = [
                        float(att_now.get("roll_speed", 0.0)),
                        float(att_now.get("pitch_speed", 0.0)),
                        float(att_now.get("yaw_speed", 0.0)),
                    ]
                dead = (0.0, 0.0, 0.0)
                if pilot is not None:
                    try:
                        dead = tuple(float(v) for v in pilot.est.snapshot()["vel_ned"])
                    except Exception:
                        pass
                row = {
                    "t": round(now - t0, 3),
                    "truth_vel": list(tvel) if tvel else None,
                    "truth_att": list(tatt) if tatt else None,
                    "ekf_vel_ned": list(snap["vel_ned"]),
                    "ekf_euler": list(snap["euler"]),
                    "dead_vel_ned": list(dead),
                    "status": list(snap["status"]),
                    "valid": snap["valid"],
                    "ev_pushed": snap["ev_pushed"],
                    "gyro_raw": gyro_raw,
                    "accel_raw": accel_raw,
                    "att_rate": att_rate,
                    "in_air": shadow._in_air,
                }
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
            if now >= next_status:
                next_status = now + 2.0
                ti, ya, ev = snap["status"]
                tstr = "truth" if tvel else "NO-TRUTH"
                ekf_sp = math.sqrt(sum(v * v for v in snap["vel_ned"]))
                truth_sp = math.sqrt(sum(v * v for v in tvel)) if tvel else float("nan")
                dead_sp = math.sqrt(sum(v * v for v in dead))
                print(f"[ecl] t={now-t0:5.1f}s tilt={int(ti)} ev={int(ev)} valid={int(snap['valid'])} "
                      f"{tstr} | |v| ekf={ekf_sp:4.1f} dead={dead_sp:5.1f} truth={truth_sp:4.1f} m/s",
                      flush=True)
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        # Guard every cleanup step so a shutdown hiccup can NEVER swallow the
        # report (the whole point of the run). The JSONL is already on disk, so
        # `--report <log>` can always recover it too.
        for label, fn in (
            ("close-log", fh.close),
            ("stop-shadow", shadow.stop),
            ("pilot-shutdown", (pilot.shutdown if pilot is not None and hasattr(pilot, "shutdown") else lambda: None)),
            ("join-ts", lambda: comps["ts_loop"].get_thread_for_join().join(timeout=2.0)),
            ("join-rx", lambda: comps["mavlink_rx"].get_thread_for_join().join(timeout=2.0)),
            ("join-vision", lambda: comps["vision_rx"].get_thread_for_join().join(timeout=2.0)),
        ):
            try:
                fn()
            except Exception as exc:
                print(f"[ecl] cleanup '{label}' failed: {exc}", flush=True)
        try:
            report(rows)
        except Exception:
            traceback.print_exc()
            print(f"[ecl] report failed above — re-run: "
                  f"uv run -m simulator.ecl_validate --report {log_path}", flush=True)
        try:
            n = shadow.dump_raw(raw_path)
            print(f"[ecl] full-rate raw+truth capture ({n} samples): {raw_path}")
            print(f"[ecl] find the correct IMU signs: "
                  f"uv run -m simulator.ecl_validate --calibrate {raw_path}")
        except Exception as exc:
            print(f"[ecl] raw dump failed: {exc}", flush=True)
        print(f"Log: {log_path}")


def calibrate(path: str) -> None:
    """Brute-force the IMU sign convention: replay a full-rate raw+truth capture
    through the EKF under each candidate (gyro-sign, accel per-axis sign) and rank
    by NED velocity RMSE vs truth. NED velocity is convention-independent, so the
    lowest RMSE IS the physically-correct convention -- no euler-frame ambiguity."""
    raw = [json.loads(l) for l in open(path) if l.strip()]
    raw = [r for r in raw if r.get("tv")]
    print(f"[calibrate] {len(raw)} full-rate samples with truth from {path}")
    if len(raw) < 500:
        print("[calibrate] too few samples -- fly a bit longer.")
        return

    def trial(gs, asx, asy, asz):
        ekf = EclEkf()
        seen_still = False
        inair = False
        errs = []
        for r in raw:
            tv = r["tv"]
            sp = math.sqrt(sum(v * v for v in tv))
            if sp < 0.5:
                seen_still = True
            if seen_still and not inair and sp > 2.0:
                inair = True
                ekf.set_at_rest(False)
                ekf.set_in_air(True)
            elif not inair:
                ekf.set_at_rest(True)
            a, g = r["a"], r["g"]
            ekf.push_imu(asx * a[0], asy * a[1], asz * a[2],
                         gs * g[0], gs * g[1], gs * g[2], r["dt"], r["et"])
            ekf.update()
            if inair and ekf.valid():
                ev = ekf.velocity_ned()
                errs.append(sum((ev[i] - tv[i]) ** 2 for i in range(3)))
        rmse = math.sqrt(sum(errs) / len(errs)) if errs else float("inf")
        return rmse, len(errs)

    results = []
    # Accel must stay a valid gravity vector (no per-axis reflection), so accel is
    # tried only as all-raw or all-negated; gyro sign is the real unknown.
    for gs in (1.0, -1.0):
        for asgn in (1.0, -1.0):
            rmse, n = trial(gs, asgn, asgn, asgn)
            results.append((rmse, gs, asgn, n))
            print(f"  gyro={gs:+.0f} accel={asgn:+.0f}: velRMSE={rmse:8.3f} m/s  (n={n})")
    results.sort(key=lambda x: x[0])
    best = results[0]
    print(f"\n[calibrate] BEST: gyro_sign={best[1]:+.0f}, accel_sign={best[2]:+.0f}  "
          f"(velRMSE={best[0]:.3f} m/s)")
    print(f"  apply:  GP_GYRO_SIGN={best[1]:+.0f}  GP_ACC_SIGN={best[2]:+.0f},{best[2]:+.0f},{best[2]:+.0f}  "
          f"ECL_GYRO_SIGN / shadow acc_sign to match")


def main() -> None:
    if "--report" in sys.argv:
        path = sys.argv[sys.argv.index("--report") + 1]
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        report(rows)
        return
    if "--calibrate" in sys.argv:
        calibrate(sys.argv[sys.argv.index("--calibrate") + 1])
        return
    run_live()


if __name__ == "__main__":
    main()
