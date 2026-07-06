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

import math
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


class VisualServo:
    """Body-frame visual servoing for the VQ2 estimator regime.

    Chase the gate the camera SEES. No world map in the control path, so
    world-position drift cannot create phantom targets (the failure mode of
    the map-based guidance without odometry). Between vision frames (YOLO at
    3-10 Hz on CPU) the target vector is propagated by the attitude delta
    (gyro -- reliable) plus estimated-velocity translation (fine over 0.3 s).
    Only DELTAS of the estimator's position enter control -- never absolutes.
    """

    def __init__(
        self,
        max_speed=1.0,
        thru_speed=0.45,
        k_fwd=0.25,
        max_lean=0.15,
        kp_lat=0.12,
        kd_lat=0.25,
        roll_dir=-1.0,
        max_tilt=0.30,
        zoff=0.15,  # aim slightly LOW in the 1.5 m opening (camera evidence
        # 2026-07-05: approaches carry ~1.2 m/s residual climb through the
        # gate, so a centre entry EXITS through the top of the opening ~0.3 s
        # later; the crossing must be aimed where the climb-through ends up
        # centred. +0.3 was too low (bottom-bar scrapes), 0.0 too high.)
        box_conf=0.5,
        pass_range=2.8,  # target this close + lost/behind -> passed
        lost_t=1.2,  # forget a target unseen this long (s)
        dash_t=0.4,  # fly straight this long after a pass (through + clear)
        brake_t=1.4,  # then brake this long -- kill speed so the CNN can
        # re-acquire the next gate (measured: too fast after a pass)
        k_brake=0.25,  # brake lean per m/s of estimated velocity
        scan_yaw=0.5,
        rate_ema=0.4,
        vel_dt=0.05,
        min_fwd=0.3,  # ignore detections closer than this ahead
        acq_max=16.0,  # only ACQUIRE a new target within this range (m).
        # Gate 2 sits ~13.8 m from the gate-1 exit (12 locked the servo out
        # of it entirely -- flight log 2026-07-05); nearest-acquire below
        # keeps 15 m background gates from stealing the lock
        match_dist=3.0,  # a detection within this of the held target updates
        # it; scaled up with range in update() (PnP depth noise alone exceeds
        # 3 m at 15 m range -- fixed matches thrash far targets)
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
            pass_range=pass_range,
            lost_t=lost_t,
            dash_t=dash_t,
            brake_t=brake_t,
            k_brake=k_brake,
            scan_yaw=scan_yaw,
            rate_ema=rate_ema,
            vel_dt=vel_dt,
            min_fwd=min_fwd,
            acq_max=acq_max,
            match_dist=match_dist,
        )
        self.t_b = None  # body-frame vector to the target gate centre
        self.last_seen = None
        self.n_passed = 0
        self.dash_until = 0.0
        self.brake_until = 0.0
        self.scan_dir = 1.0
        # (anchor_world, gate_body) pairs for the estimator's landmark update.
        # The anchor is the target's world position FROZEN at acquisition:
        # re-observing the same gate then pins the estimate to the acquisition
        # frame, so drift accumulates per-approach instead of per-flight.
        # (Without this the VQ2 estimator dead-reckons the whole flight --
        # measured 2026-07-05: z drifted +12 m and no fix ever arrived.)
        self.last_matches = []
        self._anchor_w = None
        self._prev_R = None
        self._prev_pos = None
        self._prev_fwd = None
        self._prev_lat = 0.0
        self._prev_t = None
        self._closing = 0.0
        self._lat_rate = 0.0

    def _rates(self, fwd, lat, now):
        p = self.p
        if self._prev_t is None or (now - self._prev_t) > 0.5:
            self._prev_fwd, self._prev_lat, self._prev_t = fwd, lat, now
            self._closing, self._lat_rate = 0.0, 0.0
        elif (now - self._prev_t) >= p["vel_dt"]:
            dt = now - self._prev_t
            a = p["rate_ema"]
            cl = float(np.clip(-(fwd - self._prev_fwd) / dt, -15, 15))
            lr = float(np.clip((lat - self._prev_lat) / dt, -15, 15))
            self._closing = a * cl + (1 - a) * self._closing
            self._lat_rate = a * lr + (1 - a) * self._lat_rate
            self._prev_fwd, self._prev_lat, self._prev_t = fwd, lat, now
        return self._closing, self._lat_rate

    def _brake_lean(self, R, vel):
        """Roll/pitch leaning AGAINST the estimated velocity -- used by every
        non-translating state (BRAKE, SCAN, AIM). Yaw-only states that let
        residual drift ride carried the drone into walls mid-turn (measured
        2026-07-05: collisions at 3.2 s and 7.8 s, both while rotating)."""
        v_b = R.T @ np.asarray(vel, float)
        bp = float(np.clip(self.p["k_brake"] * v_b[0], -0.06, 0.18))
        br = float(
            np.clip(-self.p["roll_dir"] * self.p["k_brake"] * v_b[1], -0.15, 0.15)
        )
        return br, bp

    def update(self, gates, pos, vel, quat, yaw, now):
        p = self.p
        pos = np.asarray(pos, float)
        R = _quat_to_R(quat)

        # --- propagate the held target through our own motion ---------------
        if self.t_b is not None and self._prev_R is not None:
            t_w = self._prev_R @ self.t_b  # world offset from previous pose
            dp = pos - self._prev_pos  # estimator DELTA (drift-free short-term)
            self.t_b = R.T @ (t_w - dp)
        self._prev_R, self._prev_pos = R, pos

        # --- ingest fresh detections (gates is [] on stale ticks) ------------
        # Track discipline: while a target is HELD, only the detection that
        # MATCHES it may update it (switching to whichever detection is most
        # confident each frame thrashes between near and 80 m background
        # gates -- measured). Acquire fresh targets only within acq_max.
        dets = []
        self.last_matches = []
        for g in gates or []:
            if g.get("conf", 0.0) < p["box_conf"] or g.get("pose") is None:
                continue
            gb = np.asarray(g["pose"]["gate_pos_body"], float)
            if gb[0] < p["min_fwd"]:
                continue
            dets.append((float(g["conf"]), gb))
        if dets and now >= self.dash_until:
            if self.t_b is not None:
                # Sticky match in BEARING space: the same gate re-observed
                # keeps its direction even when PnP depth is noisy, while a
                # 3D radius either drops far matches (fixed 3 m) or lets a
                # background gate steal the lock (range-scaled -- measured
                # d 15.5 -> 24.1 m theft, flight log 2026-07-05).
                tn = float(np.linalg.norm(self.t_b))
                match, best_ang = None, 1e9
                for _, gb in dets:
                    gn = float(np.linalg.norm(gb))
                    if gn < 1e-6 or tn < 1e-6:
                        continue
                    ratio = gn / tn
                    if not (0.6 < ratio < 1.5):
                        continue  # wildly different range = different gate
                    cosang = float(np.dot(gb, self.t_b)) / (gn * tn)
                    ang = math.acos(max(-1.0, min(1.0, cosang)))
                    if ang < best_ang:
                        best_ang, match = ang, gb
                near_ok = tn < p["match_dist"]  # close in: fall back to 3D
                if match is not None and (
                    best_ang < math.radians(12.0)
                    or (
                        near_ok
                        and float(np.linalg.norm(match - self.t_b)) < p["match_dist"]
                    )
                ):
                    self.t_b = match
                    self.last_seen = now
                    if self._anchor_w is not None:
                        # Same gate re-observed: landmark fix vs its frozen
                        # acquisition-frame position.
                        self.last_matches = [(self._anchor_w.copy(), match)]
            else:
                near = [(c, gb) for c, gb in dets if np.linalg.norm(gb) < p["acq_max"]]
                if near:
                    # NEAREST, not max-conf: a far centre-frame gate often
                    # out-scores the near one and steals the acquisition.
                    self.t_b = min(near, key=lambda x: float(np.linalg.norm(x[1])))[1]
                    self.last_seen = now
                    self._anchor_w = pos + R @ self.t_b  # freeze the anchor

        # --- dash straight through after a pass, then BRAKE ------------------
        if now < self.dash_until:
            return Cmd(0.0, -0.03, 0.0, pos[2], f"DASH passed={self.n_passed}")
        if now < self.brake_until:
            if self.t_b is not None and self.last_seen == now:
                self.brake_until = now  # next gate acquired -- resume servoing
            else:
                # Lean AGAINST the estimated velocity to actively kill the
                # carried speed -- at speed the CNN can't line up the next
                # gate in time (measured). Velocity here is the thrust-model
                # estimate: exactly the motion we commanded, so it is the
                # right thing to cancel.
                br, bp = self._brake_lean(R, vel)
                return Cmd(br, bp, 0.0, pos[2], f"BRAKE passed={self.n_passed}")

        # --- pass detection: the PROPAGATED target must actually be crossed --
        # (measured 2026-07-05: the old "close + lost" trigger counted a pass
        # at fwd=+2.2 m when YOLO lost the gate near the FOV edge -- the dash
        # then flew INTO the gate frame. Losing sight is normal at close
        # range; the propagated t_b keeps tracking through it, so require the
        # plane crossing, or lost with the target nearly crossed.)
        if self.t_b is not None:
            dist = float(np.linalg.norm(self.t_b))
            lost = self.last_seen is not None and (now - self.last_seen) > 0.3
            crossed = self.t_b[0] < 0.0 or (lost and self.t_b[0] < 0.5)
            if dist < p["pass_range"] and crossed:
                self.n_passed += 1
                self.t_b = None
                self._anchor_w = None
                self.dash_until = now + p["dash_t"]
                self.brake_until = now + p["dash_t"] + p["brake_t"]
                return Cmd(0.0, -0.06, 0.0, pos[2], f"PASSED g{self.n_passed}")

        # --- stale target -> forget it and scan ------------------------------
        if self.t_b is not None and (now - self.last_seen) > p["lost_t"]:
            self.t_b = None
            self._anchor_w = None
        if self.t_b is None:
            self._prev_t = None
            br, bp = self._brake_lean(R, vel)
            return Cmd(
                br,
                bp,
                self.scan_dir * p["scan_yaw"],
                pos[2],
                f"SCAN passed={self.n_passed}",
            )

        # --- servo on the body-frame target ----------------------------------
        fwd, lat = float(self.t_b[0]), float(self.t_b[1])
        dist = float(np.hypot(fwd, lat))
        # AIM: far target well off the nose -> yaw in place FIRST. Translating
        # while 30-40 deg off-bearing orbits the gate and drifts into walls
        # (measured: every far approach circled, closing speed ~0, then hit a
        # wall). Translate only once the gate is roughly ahead.
        bearing = math.atan2(lat, max(fwd, 0.3))
        if dist > 4.0 and abs(bearing) > 0.35:
            self._prev_t = None  # rates stale after a pure-yaw phase
            br, bp = self._brake_lean(R, vel)
            return Cmd(
                br,
                bp,
                bearing,
                pos[2],
                f"AIM b={math.degrees(bearing):+.0f} d={dist:.1f}",
            )
        closing, lat_rate = self._rates(fwd, lat, now)
        v_des = float(np.clip(dist, p["thru_speed"], p["max_speed"]))
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
        yaw_err = float(np.arctan2(lat, max(fwd, 0.3)))
        dz_world = float((R @ self.t_b)[2])
        tgt_z = pos[2] + dz_world + p["zoff"]
        return Cmd(
            tgt_roll,
            tgt_pitch,
            yaw_err,
            tgt_z,
            f"GO d={dist:.1f} fwd={fwd:+.1f} lat={lat:+.1f} c={closing:+.1f} passed={self.n_passed}",
        )


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
        entry_ttl=4.0,  # drop a map entry unseen for this long (s) -- under a
        # drifting pose estimate stale entries become phantoms
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
            entry_ttl=entry_ttl,
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
        # (map_pos, gate_body) pairs for detections that re-associated with a
        # CONFIRMED map entry this update -- landmark fixes for the estimator.
        self.last_matches = []

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
        self.last_matches = []
        for g in gates or []:
            if g.get("conf", 0.0) < p["box_conf"] or g.get("pose") is None:
                continue
            gate_body = np.asarray(g["pose"]["gate_pos_body"], float)
            gw = pos + R_wb @ gate_body
            if self._near_passed(gw):
                continue  # already flew this one
            best, bd = None, 1e9
            for e in self.gates_map:
                d = np.linalg.norm(gw - e["p"])
                if d < bd:
                    bd, best = d, e
            if best is not None and bd < p["reassoc_radius"]:
                # Landmark pair BEFORE the ema refine: the stable map position
                # is the reference the estimator corrects against.
                if best["n"] >= p["min_hits"]:
                    self.last_matches.append((best["p"].copy(), gate_body))
                best["p"] = (1 - p["ema"]) * best["p"] + p["ema"] * gw
                best["n"] += 1
                best["t"] = now
            else:
                self.gates_map.append({"p": gw, "n": 1, "t": now})
        self.gates_map = [
            e
            for e in self.gates_map
            if not self._near_passed(e["p"]) and (now - e["t"]) < p["entry_ttl"]
        ]

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
