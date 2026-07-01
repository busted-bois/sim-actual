"""Vision-only gate guidance — fly the course from detected gates alone.

No gate-map file, no GATE_INFO. Each detection's body-frame PnP pose is turned
into a WORLD NED gate position (drone_pos + R_wb @ gate_body). Confirmed gates
(seen >= min_hits times, deduped) accumulate into a persistent map, so the drone
locks the next gate the instant one exists -- even after it leaves the FOV.

Control (yaw HELD; translate with pitch+roll). Driven by the gate's BODY-frame
offset (reliable position) and its time-derivative -- NOT odometry velocity
(its body components are unreliable: under-reads forward -> speed runaway; as a
damping term it cancelled the lateral correction).
  - LATERAL roll: PD on body-right offset `lat` and its rate d(lat)/dt.
  - FORWARD pitch: regulate CLOSING SPEED = -d(fwd_to_gate)/dt toward v_des.
  - Derivatives come from position, EMA-smoothed.

All world<->body conversions use the SAME matrix R_wb and its transpose -- never
a scalar yaw (this sim's reported yaw sign disagrees with the quaternion).
"""

from dataclasses import dataclass

import numpy as np


def _quat_to_R(q):
    """Quaternion (w,x,y,z) -> body->world rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


@dataclass
class Cmd:
    tgt_roll: float
    tgt_pitch: float
    yaw_err: float
    tgt_z: float
    status: str


class VisionGuidance:
    def __init__(
        self,
        max_speed=1.5,  # max approach (closing) speed (m/s)
        thru_speed=0.8,  # min desired closing speed near a gate -> fly THROUGH
        k_fwd=0.25,  # forward lean per m/s of closing-speed error
        max_lean=0.18,  # forward tilt cap (rad)
        kp_lat=0.12,  # roll per metre of lateral offset (P)
        kd_lat=0.25,  # roll per m/s of lateral rate (D, damping) -- raised: was oscillating
        roll_dir=-1.0,  # measured: +roll moves LEFT, so right gate needs -roll
        max_tilt=0.30,  # tilt clamp (rad, ~17 deg)
        zoff=0.0,  # aim AT the opening centre (gate_pnp now targets the inner hole)
        box_conf=0.5,  # min YOLO box confidence to use a detection
        merge_radius=8.0,  # map entry within this of a PASSED gate is dropped
        reassoc_radius=10.0,  # detection within this of a map entry refines it (else new)
        min_hits=3,  # a map entry must be seen this many times before we chase it
        ema=0.3,  # smoothing when refining a map entry / target
        rate_ema=0.4,  # smoothing for the position-derivative rates
        pass_behind=0.5,  # gate counted passed once this far behind (body-fwd, m)
        pass_plane_radius=3.0,  # ...but only if this near the gate laterally
        pass_radius=1.0,  # ...or this close to gate centre regardless
        pass_cooldown=2.0,  # min seconds between counted passes
        min_pass_sep=6.0,  # ...and the drone must have moved this far since the last
        vel_dt=0.05,  # min interval for the position-derivative rates
        min_acq_fwd=1.5,  # only lock a mapped gate at least this far ahead (m)
        cone_half=0.7,  # ...and within this bearing (rad) -- else yaw-scan to it
        scan_yaw=0.35,  # yaw error injected while scanning for the next gate
    ):
        self.p = dict(
            max_speed=max_speed,
            thru_speed=thru_speed,
            k_fwd=k_fwd,
            max_lean=max_lean,
            kp_lat=kp_lat,
            kd_lat=kd_lat,
            roll_dir=roll_dir,
            max_tilt=max_tilt,
            zoff=zoff,
            box_conf=box_conf,
            merge_radius=merge_radius,
            reassoc_radius=reassoc_radius,
            min_hits=min_hits,
            ema=ema,
            rate_ema=rate_ema,
            pass_behind=pass_behind,
            pass_plane_radius=pass_plane_radius,
            pass_radius=pass_radius,
            pass_cooldown=pass_cooldown,
            min_pass_sep=min_pass_sep,
            vel_dt=vel_dt,
            min_acq_fwd=min_acq_fwd,
            cone_half=cone_half,
            scan_yaw=scan_yaw,
        )
        self.target = None  # world NED of the gate being flown
        self.gates_map = []  # [{"p": world NED, "n": hits}, ...]
        self.passed = []  # world NED of gates already flown through
        self.hold_z = None
        self.scan_dir = 1.0
        self._last_pass_t = -1e9
        self._last_pass_pos = None
        self._prev_fwd = None
        self._prev_lat = 0.0
        self._prev_t = None
        self._closing = 0.0
        self._lat_rate = 0.0

    @property
    def n_passed(self):
        return len(self.passed)

    def _rates(self, fwd_to_t, lat, now):
        p = self.p
        if self._prev_t is None or (now - self._prev_t) > 0.5:
            self._prev_fwd, self._prev_lat, self._prev_t = fwd_to_t, lat, now
            self._closing, self._lat_rate = 0.0, 0.0
        elif (now - self._prev_t) >= p["vel_dt"]:
            dt = now - self._prev_t
            a = p["rate_ema"]
            cl = float(np.clip(-(fwd_to_t - self._prev_fwd) / dt, -15, 15))
            lr = float(np.clip((lat - self._prev_lat) / dt, -15, 15))
            self._closing = a * cl + (1 - a) * self._closing
            self._lat_rate = a * lr + (1 - a) * self._lat_rate
            self._prev_fwd, self._prev_lat, self._prev_t = fwd_to_t, lat, now
        return self._closing, self._lat_rate

    def _near_passed(self, pt):
        r = self.p["merge_radius"]
        return any(np.linalg.norm(pt - q) < r for q in self.passed)

    def update(self, gates, pos, vel, quat, yaw, now):
        p = self.p
        pos = np.asarray(pos, float)
        R_wb = _quat_to_R(quat)
        Rt = R_wb.T
        if self.hold_z is None:
            self.hold_z = pos[2]

        # --- build/confirm the persistent gate map from detections ----------
        for g in gates or []:
            if g.get("conf", 0.0) < p["box_conf"] or g.get("pose") is None:
                continue
            gw = pos + R_wb @ np.asarray(g["pose"]["gate_pos_body"], float)
            if self._near_passed(gw):
                continue  # already flew this one
            best, bd = None, 1e9
            for e in self.gates_map:
                d = np.linalg.norm(gw - e["p"])
                if d < bd:
                    bd, best = d, e
            if best is not None and bd < p["reassoc_radius"]:
                best["p"] = (1 - p["ema"]) * best["p"] + p["ema"] * gw
                best["n"] += 1
            else:
                self.gates_map.append({"p": gw, "n": 1})
        self.gates_map = [e for e in self.gates_map if not self._near_passed(e["p"])]

        # --- target: keep current (snap to its refined entry) or acquire the
        # nearest CONFIRMED unpassed gate that is ahead and within the cone ---
        if self.target is not None:
            best, bd = None, 1e9
            for e in self.gates_map:
                d = np.linalg.norm(self.target - e["p"])
                if d < bd:
                    bd, best = d, e
            self.target = (
                best["p"] if (best is not None and bd < p["reassoc_radius"]) else None
            )
        if self.target is None:
            ahead = []
            for e in self.gates_map:
                if e["n"] < p["min_hits"]:
                    continue
                eb = Rt @ (e["p"] - pos)
                if eb[0] > p["min_acq_fwd"] and abs(eb[1]) < eb[0] * np.tan(
                    p["cone_half"]
                ):
                    ahead.append(e["p"])
            if ahead:
                self.target = min(ahead, key=lambda m: np.linalg.norm(m - pos))

        # --- no target in view: yaw-SCAN to bring the next gate into FOV -----
        if self.target is None:
            self._prev_t = None
            return Cmd(
                0.0,
                0.02,
                self.scan_dir * p["scan_yaw"],
                self.hold_z,
                f"SCAN n={self.n_passed} map={len(self.gates_map)}",
            )

        # --- body-frame offset to the target --------------------------------
        err_body = Rt @ (self.target - pos)
        fwd_to_t = float(err_body[0])
        lat = float(err_body[1])
        dh = float(np.hypot(fwd_to_t, lat))

        crossed = fwd_to_t < -p["pass_behind"] and dh < p["pass_plane_radius"]
        if crossed or dh < p["pass_radius"]:
            # Count a pass only if it's a NEW gate (cooldown + moved far enough)
            # -- stops re-counting one gate while circling it (the map=108 bug).
            moved = self._last_pass_pos is None or (
                np.linalg.norm(pos - self._last_pass_pos) > p["min_pass_sep"]
            )
            if (now - self._last_pass_t) > p["pass_cooldown"] and moved:
                self.passed.append(self.target.copy())
                self._last_pass_t = now
                self._last_pass_pos = pos.copy()
            self.gates_map = [
                e
                for e in self.gates_map
                if np.linalg.norm(e["p"] - self.target) > p["merge_radius"]
            ]
            self.hold_z = self.target[2]
            self.target = None
            return Cmd(0.0, -0.05, 0.0, self.hold_z, f"PASSED g{self.n_passed}")

        closing, lat_rate = self._rates(fwd_to_t, lat, now)

        # --- FORWARD: closing-speed regulation; LATERAL: PD on offset -------
        v_des = float(np.clip(dh, p["thru_speed"], p["max_speed"]))
        lean = float(
            np.clip(p["k_fwd"] * (v_des - closing), -p["max_tilt"], p["max_lean"])
        )
        tgt_pitch = -lean
        tgt_roll = float(
            np.clip(
                p["roll_dir"] * (p["kp_lat"] * lat + p["kd_lat"] * lat_rate),
                -p["max_tilt"],
                p["max_tilt"],
            )
        )
        return Cmd(
            tgt_roll,
            tgt_pitch,
            0.0,
            self.target[2] + p["zoff"],
            f"GO d={dh:.1f} fwd={fwd_to_t:+.1f} lat={lat:+.1f} c={closing:+.1f} passed={self.n_passed} map={len(self.gates_map)}",
        )
