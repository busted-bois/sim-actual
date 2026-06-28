"""Vision-only gate guidance — fly the course from detected gates alone.

No gate-map file, no GATE_INFO. Each detection's body-frame PnP pose is turned
into a WORLD NED gate position (drone_pos + R_wb @ gate_body) and remembered, so
the drone keeps flying to a gate even after it leaves the camera FOV.

Control (yaw HELD; translate with pitch+roll). Everything is driven by the
gate's BODY-frame offset (from reliable position) and its time-derivative --
NO odometry velocity (its body components are unreliable: under-reading forward
let speed run to ~7 m/s, and as a damping term it cancelled the lateral
correction so the drone never banked).
  - LATERAL roll: PD on the body-right offset `lat` and its rate d(lat)/dt.
  - FORWARD pitch: regulate CLOSING SPEED = -d(fwd_to_gate)/dt toward v_des.
  - Both derivatives come from position and are EMA-smoothed (raw 20 Hz position
    differences are noisy).

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
        max_lean=0.18,  # forward tilt cap (rad) -- limits speed build-up
        kp_lat=0.15,  # roll per metre of lateral offset (P)
        kd_lat=0.15,  # roll per m/s of lateral rate (D, damping)
        roll_dir=-1.0,  # measured: +roll moves LEFT, so right gate needs -roll
        # (fly2's inner loop already inverts roll via SIGN_ROLL=-1; ref uses +1)
        max_tilt=0.30,  # tilt clamp (rad, ~17 deg)
        zoff=0.8,  # aim BELOW detection centre (NED +down): drop into the opening,
        # not the AI-GP box on top of the gate (centre estimate biases high)
        box_conf=0.5,  # min YOLO box confidence to use a detection
        merge_radius=4.0,  # detections within this of a passed gate are ignored
        reassoc_radius=5.0,  # detection re-associates with the target if within this
        ema=0.3,  # smoothing when refining the target from a new detection
        rate_ema=0.4,  # smoothing for the position-derivative rates
        pass_behind=0.5,  # gate counted passed once this far behind (body-fwd, m)
        pass_plane_radius=3.0,  # ...but only if this near the gate laterally
        pass_radius=1.0,  # ...or this close to gate centre regardless
        vel_dt=0.05,  # min interval for the position-derivative rates
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
            ema=ema,
            rate_ema=rate_ema,
            pass_behind=pass_behind,
            pass_plane_radius=pass_plane_radius,
            pass_radius=pass_radius,
            vel_dt=vel_dt,
        )
        self.target = None
        self.passed = []
        self.hold_z = None
        self._prev_fwd = None
        self._prev_lat = 0.0
        self._prev_t = None
        self._closing = 0.0  # smoothed -d(fwd)/dt
        self._lat_rate = 0.0  # smoothed d(lat)/dt

    @property
    def n_passed(self):
        return len(self.passed)

    def _rates(self, fwd_to_t, lat, now):
        """Closing speed -d(fwd)/dt and lateral rate d(lat)/dt from reliable
        position, EMA-smoothed. Held between updates; reset on gaps."""
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

    def update(self, gates, pos, vel, quat, yaw, now):
        p = self.p
        pos = np.asarray(pos, float)
        R_wb = _quat_to_R(quat)
        Rt = R_wb.T
        if self.hold_z is None:
            self.hold_z = pos[2]

        # --- detections -> candidate world gate positions -------------------
        cands = []
        for g in gates or []:
            if g.get("conf", 0.0) < p["box_conf"] or g.get("pose") is None:
                continue
            gw = pos + R_wb @ np.asarray(g["pose"]["gate_pos_body"], float)
            if any(np.linalg.norm(gw - q) < p["merge_radius"] for q in self.passed):
                continue
            cands.append(gw)
        if cands:
            if self.target is None:
                self.target = min(cands, key=lambda c: np.linalg.norm(c - pos))
            else:
                near = min(cands, key=lambda c: np.linalg.norm(c - self.target))
                if np.linalg.norm(near - self.target) < p["reassoc_radius"]:
                    self.target = (1 - p["ema"]) * self.target + p["ema"] * near

        # --- no target: hold position/altitude ------------------------------
        if self.target is None:
            self._prev_t = None  # reset rate tracker
            return Cmd(0.0, 0.03, 0.0, self.hold_z, f"SEARCH n={self.n_passed}")

        # --- body-frame offset to the target (position; reliable) -----------
        err_body = Rt @ (self.target - pos)
        fwd_to_t = float(err_body[0])
        lat = float(err_body[1])
        dh = float(np.hypot(fwd_to_t, lat))

        crossed = fwd_to_t < -p["pass_behind"] and dh < p["pass_plane_radius"]
        if crossed or dh < p["pass_radius"]:
            self.passed.append(self.target.copy())
            self.hold_z = self.target[2]
            self.target = None
            return Cmd(0.0, -0.05, 0.0, self.hold_z, f"PASSED g{self.n_passed}")

        closing, lat_rate = self._rates(fwd_to_t, lat, now)

        # --- FORWARD: regulate closing speed (reliable, lateral-independent) -
        v_des = float(np.clip(dh, p["thru_speed"], p["max_speed"]))
        lean = float(
            np.clip(p["k_fwd"] * (v_des - closing), -p["max_tilt"], p["max_lean"])
        )
        tgt_pitch = -lean

        # --- LATERAL: PD on body-right offset (position + its rate) ----------
        # lat>0 = gate body-RIGHT. roll_dir sets which roll moves toward it.
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
            f"GO d={dh:.1f} fwd={fwd_to_t:+.1f} lat={lat:+.1f} c={closing:+.1f} roll={tgt_roll:+.2f} passed={self.n_passed}",
        )
