"""Shadow-mode validation of the VQ2 ecl/EKF2 state estimator against a
KNOWN-GOOD flight.

Abhay's telemetry raceline pilot (aigp-round1-qualifier-1/fly.py) completes the
course on the legacy VQ1 simulator. We run it COMPLETELY UNMODIFIED, in-process,
via a transparent MAVLink tap, while our VQ2 estimator observes in parallel as a
PASSIVE OBSERVER. The estimator consumes ONLY the sensors it would have in VQ2:

    * HIGHRES_IMU   (tapped from the same MAVLink stream)
    * camera images (independent VisionRX on UDP 5600 -> vision velocity)

It NEVER consumes the legacy ground-truth position/velocity/attitude -- those are
logged separately (LOCAL_POSITION_NED, ATTITUDE) and only used to SCORE the
estimator afterwards. The pilot's flight logic, navigation, and gains are
untouched; the estimator never commands the drone.

Two estimators run side by side so we can isolate what vision contributes:
    * FULL : IMU + camera vision-velocity   (the real VQ2 estimator)
    * IMU  : IMU only, vision disabled       (dead-reckoning reference)

Ground-truth attitude uses the VQ1 ATTITUDE message, whose PITCH sign is flipped
vs the true (odometry-quaternion) convention -- so a large *constant* pitch
offset in the report is a reporting artifact, not estimator error (see analysis).

Usage (our venv, sim on Windows):
    uv run -m simulator.shadow_validate            # runs Abhay's pilot + shadow
    # then drive the race in the sim exactly as you would for his fly.py
    uv run -m simulator.shadow_validate --report logs/shadow_<ts>.jsonl
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

# Abhay's repo lives beside this workspace; add it so we can run his pilot as-is.
ABHAY_DIR = os.environ.get(
    "ABHAY_DIR", r"C:\Users\kunal\Desktop\aigp-round1-qualifier-1"
)

# VQ1 IMU convention (measured vs truth): gyro +1 all axes, accel raw.
GYRO_SIGN = 1.0
LOG_HZ_CAP = 60.0  # cap synchronized-log rate (truth arrives ~50 Hz)


def _euler_from_quat(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _add_abhay_path():
    for p in (ABHAY_DIR, os.path.join(ABHAY_DIR, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


class ShadowValidator:
    """Owns the MAVLink tap, the two estimators, the camera receiver, and the
    synchronized truth-vs-estimate log."""

    def __init__(self):
        # What the ESTIMATOR sees (IMU from the tap + pose from the camera). No
        # ground truth ever goes in here.
        self.est_data: dict = {"_quiet_vision": True}  # silence per-frame [vision] spam
        # Ground truth, logged only (never fed to the estimator).
        self.truth = {"pos": None, "vel": None, "att": None, "t_ms": None, "gate": -1}

        self.conn = None
        self._imu_seen = False
        self._first_hb = threading.Event()

        # Import EclShadow lazily (pulls the DLL) after path setup is fine.
        from simulator.ecl_validate import EclShadow

        # Per-shadow IMU queues: the tap appends EVERY HIGHRES_IMU sample so the
        # shadow threads process them all with correct dt, even when GIL-starved
        # by Abhay's control loop + YOLO. (A single overwritten dict dropped ~90%
        # of samples -> dt clamped to 0.1 -> tilt never aligned.)
        from collections import deque

        self._q_full = deque(maxlen=8000)
        self._q_imu = deque(maxlen=8000)
        self.full = EclShadow(self.est_data, gyro_sign=GYRO_SIGN, use_vision=True,
                              verbose=False, imu_queue=self._q_full)
        self.imu = EclShadow(self.est_data, gyro_sign=GYRO_SIGN, use_vision=False,
                             verbose=False, imu_queue=self._q_imu)

        self.vision_rx = None
        self.rows: list[dict] = []
        self._last_log = 0.0
        self._t0 = time.time()

        # Optional race-status parsing (context only), reusing Abhay's parser.
        _add_abhay_path()
        try:
            from aigp_pilot import parsers  # noqa

            self._parsers = parsers
        except Exception:
            self._parsers = None

    # ---- MAVLink tap -----------------------------------------------------
    def install_tap(self):
        """Monkeypatch pymavlink so every message on Abhay's connection is teed
        to us. His code calls mavutil.mavlink_connection(...).recv_match(...);
        we wrap both. His flight logic is not touched."""
        import pymavlink.mavutil as mavutil

        orig_conn = mavutil.mavlink_connection

        def patched(*a, **k):
            conn = orig_conn(*a, **k)
            self.conn = conn
            orig_recv = conn.recv_match

            def teed(*ra, **rk):
                msg = orig_recv(*ra, **rk)
                if msg is not None:
                    try:
                        self._on_msg(msg)
                    except Exception:
                        pass
                return msg

            conn.recv_match = teed
            return conn

        mavutil.mavlink_connection = patched

    def _on_msg(self, msg):
        typ = msg.get_type()
        if typ == "HIGHRES_IMU":
            self._imu_seen = True
            # Estimator input only. Accel = specific force (m/s^2), gyro rad/s.
            s = {
                "ax": msg.xacc, "ay": msg.yacc, "az": msg.zacc,
                "gx": msg.xgyro, "gy": msg.ygyro, "gz": msg.zgyro,
                "time_us": int(msg.time_usec),
            }
            self.est_data["imu"] = s          # latest (for status readout)
            self._q_full.append(s)            # queue EVERY sample to each shadow
            self._q_imu.append(s)
        elif typ == "LOCAL_POSITION_NED":
            self.truth["pos"] = (float(msg.x), float(msg.y), float(msg.z))
            self.truth["vel"] = (float(msg.vx), float(msg.vy), float(msg.vz))
            self.truth["t_ms"] = int(msg.time_boot_ms)
            self._maybe_log()  # truth pos+vel arrived -> synchronized sample
        elif typ == "ATTITUDE":
            self.truth["att"] = (float(msg.roll), float(msg.pitch), float(msg.yaw))
            self.truth["t_ms"] = int(msg.time_boot_ms)
        elif typ == "HEARTBEAT":
            self._first_hb.set()
        elif typ == "ENCAPSULATED_DATA" and self._parsers is not None:
            raw = bytes(msg.data)
            if raw and raw[0] == self._parsers.RACE_STATUS_ID:
                try:
                    self.truth["gate"] = self._parsers.parse_race_status(raw).active_gate_index
                except Exception:
                    pass

    def _maybe_log(self):
        now = time.time()
        if now - self._last_log < 1.0 / LOG_HZ_CAP:
            return
        self._last_log = now
        if self.truth["pos"] is None or self.truth["vel"] is None or self.truth["att"] is None:
            return
        sf = self.full.snapshot()
        si = self.imu.snapshot()
        self.rows.append({
            "t": round(now - self._t0, 4),
            "t_ms": self.truth["t_ms"],
            "gate": self.truth["gate"],
            "truth_pos": list(self.truth["pos"]),
            "truth_vel": list(self.truth["vel"]),
            "truth_att": list(self.truth["att"]),
            "full_pos": list(sf["pos_ned"]), "full_vel": list(sf["vel_ned"]),
            "full_euler": list(sf["euler"]), "full_valid": sf["valid"],
            "full_var_vel": list(sf["var"]["vel_ned"]),
            "full_var_accbias": list(sf["var"]["accel_bias"]),
            "full_status": list(sf["status"]), "ev_pushed": sf["ev_pushed"],
            "full_in_air": sf["in_air"],
            "acc_fed": list(sf["acc_fed"]), "gyro_fed": list(sf["gyro_fed"]),
            "ev_diag": sf.get("ev_diag"),
            "imu_pos": list(si["pos_ned"]), "imu_vel": list(si["vel_ned"]),
            "imu_euler": list(si["euler"]), "imu_valid": si["valid"],
        })

    # ---- HIGHRES_IMU request (Abhay never asks for it) -------------------
    def _request_imu_loop(self):
        import pymavlink.mavutil as mavutil

        self._first_hb.wait(timeout=60)
        for _ in range(40):
            if self._imu_seen:
                return
            conn = self.conn
            if conn is not None and getattr(conn, "target_system", 0):
                try:
                    conn.mav.command_long_send(
                        conn.target_system, conn.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                        mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 10000,
                        0, 0, 0, 0, 0,
                    )
                except Exception:
                    pass
            time.sleep(0.5)

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        self.full.start()
        self.imu.start()
        try:
            from simulator.vision_rx import VisionRX

            self.vision_rx = VisionRX(self.est_data)
            print("[shadow] VisionRX started (camera on UDP 5600).", flush=True)
        except Exception as exc:
            print(f"[shadow] camera/VisionRX unavailable ({exc}); IMU-only vision aiding off.",
                  flush=True)
        threading.Thread(target=self._request_imu_loop, daemon=True).start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    def _status_loop(self):
        """Live proof the estimator is receiving IMU + camera and tracking."""
        last_imu_ts = None
        while True:
            time.sleep(2.0)
            imu = self.est_data.get("imu")
            imu_ts = imu.get("time_us") if imu else None
            adv = (imu_ts is not None and imu_ts != last_imu_ts)
            last_imu_ts = imu_ts
            sf = self.full.snapshot()
            tv = self.truth["vel"]
            ekf_sp = math.sqrt(sum(v * v for v in sf["vel_ned"]))
            tr_sp = math.sqrt(sum(v * v for v in tv)) if tv else float("nan")
            ti, ya, evf = sf["status"]
            anorm = math.sqrt(sum(v * v for v in sf["acc_fed"]))
            gnorm = math.sqrt(sum(v * v for v in sf["gyro_fed"]))
            evd = sf.get("ev_diag")
            evstr = (f"ev:{'ACC' if evd['accept'] else 'REJ'} innov={evd['innov']}" if evd else "ev:--")
            print(f"[shadow] imu={'live' if adv else 'STALE'} in_air={int(sf['in_air'])} "
                  f"tilt={int(ti)} valid={int(sf['valid'])} gate={self.truth['gate']} {evstr} "
                  f"| |a|={anorm:4.1f} |g|={gnorm:4.2f} | |v| ekf={ekf_sp:4.1f} truth={tr_sp:4.1f} m/s",
                  flush=True)

    def run_abhay(self):
        """Import and run Abhay's pilot unmodified. Blocks until his flight ends."""
        _add_abhay_path()
        if not os.path.exists(os.path.join(ABHAY_DIR, "fly.py")):
            raise FileNotFoundError(
                f"Abhay's fly.py not found under {ABHAY_DIR}. Set ABHAY_DIR env var."
            )
        import fly  # his entry point

        print("[shadow] running Abhay's pilot (unmodified) with estimator shadow...", flush=True)
        fly.main()


# ------------------------- metrics / analysis -----------------------------
def _rmse(err):
    a = np.asarray(err, dtype=float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def analyze(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  VQ2 ESTIMATOR — SHADOW-MODE VALIDATION vs KNOWN-GOOD FLIGHT")
    print("=" * 72)
    print(f"  logged samples: {len(rows)}")

    full = [r for r in rows if r.get("full_valid")]
    imu = [r for r in rows if r.get("imu_valid")]
    ev = sum(1 for r in rows if r.get("ev_pushed"))
    gates = [r["gate"] for r in rows if r.get("gate", -1) >= 0]
    print(f"  FULL (imu+vision) valid: {len(full)}   IMU-only valid: {len(imu)}   "
          f"EV vision-vel samples: {ev}")
    if gates:
        print(f"  race gate range during flight: {min(gates)} -> {max(gates)}")
    if not full:
        print("\n  No valid FULL-estimator samples -- tilt never aligned. Diagnosing:")
        with_imu = [r for r in rows if r.get("acc_fed")]
        if with_imu:
            an = [math.sqrt(sum(v * v for v in r["acc_fed"])) for r in with_imu]
            gn = [math.sqrt(sum(v * v for v in r["gyro_fed"])) for r in with_imu]
            inair = [r for r in with_imu if r.get("full_in_air")]
            print(f"    accel |a| fed to EKF: mean={np.mean(an):.2f} (expect ~9.81)  "
                  f"min={min(an):.2f} max={max(an):.2f}")
            print(f"    gyro  |g| fed to EKF: mean={np.mean(gn):.2f}  max={max(gn):.2f} rad/s "
                  f"(at rest expect ~0)")
            first_air = next((r["t"] for r in with_imu if r.get("full_in_air")), None)
            print(f"    in_air latched: {len(inair)}/{len(with_imu)} samples"
                  + (f", first at t={first_air:.1f}s" if first_air is not None else " (never)"))
            if abs(np.mean(an) - 9.81) > 1.5:
                print("    -> |a| far from 9.81: accel scale/units wrong -> tilt can't align.")
            elif inair and first_air is not None and first_air < 3.0:
                print("    -> in_air latched too early (before real launch): ZUPT skipped."
                      " Debounce should now prevent this.")
            else:
                print("    -> check gyro |g| at rest; if >0.6 it blocks the at-rest ZUPT.")
        print("=" * 72 + "\n")
        return

    tv = np.array([r["truth_vel"] for r in full])
    tp = np.array([r["truth_pos"] for r in full])
    ta = np.array([r["truth_att"] for r in full])
    t = np.array([r["t"] for r in full])

    def block(label, rowset):
        rs = [r for r in rowset if r.get("full_valid")] if label == "FULL" else \
             [r for r in rowset if r.get("imu_valid")]
        if not rs:
            return
        pfx = "full" if label == "FULL" else "imu"
        ev_ = np.array([r[f"{pfx}_vel"] for r in rs])
        tv_ = np.array([r["truth_vel"] for r in rs])
        ep_ = np.array([r[f"{pfx}_pos"] for r in rs])
        tp_ = np.array([r["truth_pos"] for r in rs])
        verr = ev_ - tv_
        vspeed_err = np.linalg.norm(verr, axis=1)
        # Position: the EKF has no position aiding and starts at origin, while
        # truth is absolute NED -> compare DISPLACEMENT from the first sample
        # (the honest dead-reckoning drift), not the absolute offset.
        disp_est = ep_ - ep_[0]
        disp_truth = tp_ - tp_[0]
        perr = np.linalg.norm(disp_est - disp_truth, axis=1)
        print(f"\n  [{label}]  ({len(rs)} valid samples)")
        print(f"    VELOCITY RMSE (m/s)   N={_rmse(verr[:,0]):.2f}  E={_rmse(verr[:,1]):.2f}  "
              f"D={_rmse(verr[:,2]):.2f}   |v|={_rmse(vspeed_err):.2f}")
        print(f"    max velocity error    {float(vspeed_err.max()):.2f} m/s")
        print(f"    POSITION drift (m)    mean={float(perr.mean()):.1f}  max={float(perr.max()):.1f}"
              f"  (displacement vs truth; no position aiding)")

    block("FULL", full)
    block("IMU", imu)

    # Vision measurement health: innovation gating + accept/reject (from ev_diag).
    diags = [r["ev_diag"] for r in rows if r.get("ev_diag")]
    if diags:
        acc = [d for d in diags if d["accept"]]
        rej = [d for d in diags if not d["accept"]]
        innov = np.array([d["innov"] for d in diags])
        spd = np.array([d["speed"] for d in diags])
        print(f"\n  VISION MEASUREMENT HEALTH ({len(diags)} vision frames)")
        print(f"    accepted={len(acc)} ({100*len(acc)/len(diags):.0f}%)  rejected={len(rej)} "
              f"({100*len(rej)/len(diags):.0f}%)")
        print(f"    innovation |vision-pred| (m/s): mean={innov.mean():.1f}  p50={np.median(innov):.1f}  "
              f"max={innov.max():.1f}")
        print(f"    raw vision speed (m/s):         mean={spd.mean():.1f}  max={spd.max():.1f}"
              f"  (drone caps ~9)")
        if acc:
            ai = np.array([d["innov"] for d in acc])
            print(f"    ACCEPTED innovation: mean={ai.mean():.2f}  max={ai.max():.2f} (these corrected the state)")

    # Attitude (FULL). Report per-axis; pitch has the known ATTITUDE sign flip.
    ea = np.array([r["full_euler"] for r in full])
    roll_e = np.array([abs(_wrap_pi(ea[i, 0] - ta[i, 0])) for i in range(len(ea))])
    yaw_e = np.array([abs(_wrap_pi(ea[i, 2] - ta[i, 2])) for i in range(len(ea))])
    pitch_e_raw = np.array([abs(_wrap_pi(ea[i, 1] - ta[i, 1])) for i in range(len(ea))])
    pitch_e_flip = np.array([abs(_wrap_pi(ea[i, 1] + ta[i, 1])) for i in range(len(ea))])
    print("\n  ATTITUDE error (deg, FULL)")
    print(f"    roll  mean={math.degrees(roll_e.mean()):.1f}  max={math.degrees(roll_e.max()):.1f}")
    print(f"    yaw   mean={math.degrees(yaw_e.mean()):.1f}  max={math.degrees(yaw_e.max()):.1f}  "
          f"(mag-free -> expect slow drift)")
    print(f"    pitch mean={math.degrees(pitch_e_raw.mean()):.1f} (raw)   "
          f"{math.degrees(pitch_e_flip.mean()):.1f} (sign-corrected)  "
          f"<- use the smaller; ATTITUDE msg flips pitch sign")

    # Drift over time: velocity-error growth (FULL), position-error growth.
    vspeed_err_full = np.linalg.norm(
        np.array([r["full_vel"] for r in full]) - tv, axis=1)
    fp = np.array([r["full_pos"] for r in full])
    perr_full = np.linalg.norm((fp - fp[0]) - (tp - tp[0]), axis=1)  # displacement drift
    if len(t) > 20:
        half = len(t) // 2
        print("\n  DRIFT over time (FULL)")
        print(f"    velocity |err|: 1st half mean={vspeed_err_full[:half].mean():.2f}  "
              f"2nd half mean={vspeed_err_full[half:].mean():.2f} m/s")
        dur = max(1e-3, t[-1] - t[0])
        print(f"    position drift: {perr_full[0]:.1f} -> {perr_full[-1]:.1f} m "
              f"over {dur:.1f}s  ({perr_full[-1]/dur:.2f} m/s avg drift rate)")

    # Verdict heuristics on the FULL velocity estimate.
    v_rmse = _rmse(vspeed_err_full)
    v_rmse_imu = _rmse(np.linalg.norm(
        np.array([r["imu_vel"] for r in imu]) - np.array([r["truth_vel"] for r in imu]),
        axis=1)) if imu else float("nan")
    print("\n  VERDICT")
    print(f"    FULL velocity |v| RMSE = {v_rmse:.2f} m/s   IMU-only = {v_rmse_imu:.2f} m/s")
    if v_rmse < 1.0:
        print("    -> velocity estimate tracks truth well (<1 m/s): usable for VQ2.")
    elif v_rmse < 3.0:
        print("    -> moderate velocity error (1-3 m/s): borderline; see drift/attitude.")
    else:
        print("    -> velocity error large (>3 m/s): NOT yet accurate enough for VQ2.")
    if not math.isnan(v_rmse_imu):
        print(f"    -> vision {'HELPS' if v_rmse < v_rmse_imu else 'HURTS'} "
              f"(FULL {'<' if v_rmse < v_rmse_imu else '>='} IMU-only).")
    print("=" * 72 + "\n")


def _load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    if "--report" in sys.argv:
        analyze(_load(sys.argv[sys.argv.index("--report") + 1]))
        return

    v = ShadowValidator()
    v.install_tap()
    v.start()

    boot = int(time.time())
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(logs_dir, f"shadow_{boot}.jsonl"))

    try:
        v.run_abhay()  # blocks until Abhay's flight completes
    except KeyboardInterrupt:
        print("\n[shadow] interrupted.", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        v.full.stop()
        v.imu.stop()
        try:
            with open(log_path, "w") as fh:
                for r in v.rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"[shadow] synchronized log ({len(v.rows)} rows): {log_path}", flush=True)
        except Exception as exc:
            print(f"[shadow] log write failed: {exc}", flush=True)
        try:
            analyze(v.rows)
        except Exception:
            traceback.print_exc()
        print(f"[shadow] re-report: uv run -m simulator.shadow_validate --report {log_path}")


if __name__ == "__main__":
    main()
