"""Vision-only gate guidance — fly the course from detected gates alone.

No gate-map file, no GATE_INFO. Each detection's body-frame PnP pose is turned
into a WORLD NED gate position (drone_pos + R_wb @ gate_body). Confirmed gates
(seen >= min_hits times, deduped) accumulate into a persistent map, so the drone
locks the next gate the instant one exists -- even after it leaves the FOV.

TRAVERSAL is two-phase (Falanga ICRA'17 narrow-gap; AlphaPilot). Detection
always dies at point-blank range -- the corners leave the frame -- so guidance
that must keep SEEING the gate can never finish a pass (measured: every run
stalled or scanned right in front of the opening). Instead:
  APPROACH  carrot-track the fly-through AXIS (gate centre + PnP plane normal)
            toward a pre-gate point d_pre in front of the opening, camera held
            on the gate so the map keeps refining.
  COMMIT    once on-axis and yaw-aligned, freeze gate+axis and dash straight
            through on pose-delta dead reckoning, ignoring vision. The pass is
            declared from GEOMETRY (crossing the gate plane), never from sight.

Control (translate with pitch+roll; yaw tracks the gate / the axis). Driven by
the aim point's BODY-frame offset (reliable position) and its time-derivative
-- NOT odometry velocity (its body components are unreliable: under-reads
forward -> speed runaway; as a damping term it cancelled the lateral
correction).
  - LATERAL roll: PD on body-right offset `lat` and its rate d(lat)/dt.
  - FORWARD pitch: regulate CLOSING SPEED = -d(fwd)/dt toward v_des.
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


# --- traversal-axis geometry (shared by both guidance classes) ----------------


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else np.asarray(v, float) / n


def _flat(v):
    """Horizontal unit vector (z zeroed). Gates stand upright, so flattening
    the fly-through axis discards the PnP tilt component (IPPE's weak axis)."""
    return _unit(np.array([v[0], v[1], 0.0]))


def _blend_axis(cur, sample, ema):
    """EMA a fly-through axis; drop samples >60 deg off the running estimate
    (IPPE's planar two-fold ambiguity shows up as exactly such flips)."""
    if sample is None:
        return cur
    if cur is None:
        return sample
    if float(np.dot(cur, sample)) < 0.5:
        return cur
    return _unit((1.0 - ema) * cur + ema * sample)


def _axis_coords(to_gate_b, axis_b):
    """Drone position in the gate's axis frame, in CURRENT body coordinates.
    s: along-axis coordinate (gate plane at 0; negative = before the gate).
    cross: off-axis vector (y ~ lateral, z ~ vertical at small tilt)."""
    r = -np.asarray(to_gate_b, float)
    s = float(np.dot(r, axis_b))
    return s, r - s * axis_b


class VisualServo:
    """Body-frame visual servoing for the VQ2 estimator regime.

    Chase the gate the camera SEES. No world map in the control path, so
    world-position drift cannot create phantom targets (the failure mode of
    the map-based guidance without odometry). Between vision frames (YOLO at
    3-10 Hz on CPU) the target vector is propagated by the attitude delta
    (gyro -- reliable) plus estimated-velocity translation (fine over 0.3 s).
    Only DELTAS of the estimator's position enter control -- never absolutes.
    The traversal axis is held in the body frame too, rotated (never
    translated -- it's a direction) through the same propagation.
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
        zoff=0.3,  # aim below the detected centre (measured high bias)
        box_conf=0.5,
        lost_t=1.2,  # forget a FAR target unseen this long (s)
        blind_max=3.0,  # ...but hold a NEAR (<6m) one blind up to this long:
        # detection always dies at point-blank range
        dash_t=0.4,  # fly straight this long after a pass (through + clear)
        brake_t=1.4,  # then brake this long -- kill speed so the CNN can
        # re-acquire the next gate (measured: too fast after a pass)
        k_brake=0.25,  # brake lean per m/s of estimated velocity
        max_brake=0.08,  # backward-tilt cap when braking (rad) -- closing-rate
        # spikes at max_tilt pitched the drone into full-reverse flight
        scan_yaw=0.5,
        rate_ema=0.4,
        vel_dt=0.05,
        min_fwd=0.3,  # ignore detections closer than this ahead
        acq_max=18.0,  # only ACQUIRE a new target within this range (m)
        match_dist=3.0,  # a detection within this of the held target updates it
        d_pre=2.5,  # pre-gate point this far in front, on the axis (m)
        d_post=2.0,  # commit carries this far past the plane (m)
        d_carrot=1.5,  # pure-pursuit lookahead along the axis (m)
        align_lat=0.4,  # cross-track bound to enter COMMIT (m)
        align_z=0.4,  # vertical bound to enter COMMIT (m)
        align_yaw=0.26,  # yaw-to-axis bound to enter COMMIT (rad, ~15 deg)
        commit_speed=1.0,  # desired closing speed through the gate (m/s)
        pass_behind=0.7,  # plane crossed by this much -> PASSED (m)
        abort_lat=0.9,  # off-axis beyond this during COMMIT -> abort (m)
        axis_ema=0.3,  # smoothing for the fly-through axis
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
            lost_t=lost_t,
            blind_max=blind_max,
            dash_t=dash_t,
            brake_t=brake_t,
            k_brake=k_brake,
            max_brake=max_brake,
            scan_yaw=scan_yaw,
            rate_ema=rate_ema,
            vel_dt=vel_dt,
            min_fwd=min_fwd,
            acq_max=acq_max,
            match_dist=match_dist,
            d_pre=d_pre,
            d_post=d_post,
            d_carrot=d_carrot,
            align_lat=align_lat,
            align_z=align_z,
            align_yaw=align_yaw,
            commit_speed=commit_speed,
            pass_behind=pass_behind,
            abort_lat=abort_lat,
            axis_ema=axis_ema,
        )
        self.t_b = None  # body-frame vector to the target gate centre
        self.a_b = None  # body-frame fly-through axis (unit)
        self.committing = False
        self.last_seen = None
        self.n_passed = 0
        self.dash_until = 0.0
        self.brake_until = 0.0
        self.scan_dir = 1.0
        self.last_matches = []  # API compat (no map -> no landmark pairs)
        self.debug = {}
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

    def _drop(self):
        self.t_b = None
        self.a_b = None
        self.committing = False
        self._prev_t = None

    def _servo(self, aim_b, axis_b, v_cap, R, pos, now, tag, yaw_on_axis, s, cross):
        p = self.p
        fwd, lat = float(aim_b[0]), float(aim_b[1])
        dist = float(np.hypot(fwd, lat))
        closing, lat_rate = self._rates(fwd, lat, now)
        v_des = float(np.clip(dist, min(p["thru_speed"], v_cap), v_cap))
        lean = float(
            np.clip(p["k_fwd"] * (v_des - closing), -p["max_brake"], p["max_lean"])
        )
        tgt_pitch = -lean
        tgt_roll = float(
            np.clip(
                p["roll_dir"] * (p["kp_lat"] * lat + p["kd_lat"] * lat_rate),
                -p["max_tilt"],
                p["max_tilt"],
            )
        )
        if yaw_on_axis:
            # COMMIT: the gate is at point-blank bearing -- yaw to the axis.
            yaw_err = float(np.clip(np.arctan2(axis_b[1], axis_b[0]), -0.5, 0.5))
        else:
            # APPROACH: camera stays on the GATE (vision-only: never go blind
            # earlier than the commit demands).
            yaw_err = float(np.arctan2(self.t_b[1], max(float(self.t_b[0]), 0.3)))
        dz_world = float((R @ aim_b)[2])
        tgt_z = pos[2] + dz_world + p["zoff"]
        self.debug = {
            "phase": tag.split()[0],
            "s": s,
            "lat": float(cross[1]),
            "vert": float(cross[2]),
            "dist": float(np.linalg.norm(self.t_b)),
        }
        return Cmd(
            tgt_roll,
            tgt_pitch,
            yaw_err,
            tgt_z,
            f"{tag} c={closing:+.1f} passed={self.n_passed}",
        )

    def update(self, gates, pos, vel, quat, yaw, now):
        p = self.p
        pos = np.asarray(pos, float)
        R = _quat_to_R(quat)

        # --- propagate the held target + axis through our own motion ---------
        if self._prev_R is not None:
            if self.t_b is not None:
                t_w = self._prev_R @ self.t_b  # world offset from previous pose
                dp = pos - self._prev_pos  # estimator DELTA (drift-free short-term)
                self.t_b = R.T @ (t_w - dp)
            if self.a_b is not None:  # direction: rotate only, never translate
                self.a_b = R.T @ (self._prev_R @ self.a_b)
        self._prev_R, self._prev_pos = R, pos

        # --- ingest fresh detections (gates is [] on stale ticks). Blocked
        # while COMMITTED: the dash is deliberately blind -- a last-moment
        # misdetection must not bend it. ---------------------------------------
        # Track discipline: while a target is HELD, only the detection that
        # MATCHES it may update it (switching to whichever detection is most
        # confident each frame thrashes between near and 80 m background
        # gates -- measured). Acquire fresh targets only within acq_max.
        dets = []
        if not self.committing:
            for g in gates or []:
                if g.get("conf", 0.0) < p["box_conf"] or g.get("pose") is None:
                    continue
                gb = np.asarray(g["pose"]["gate_pos_body"], float)
                if gb[0] < p["min_fwd"]:
                    continue
                nb = g["pose"].get("normal_body")
                ab = None
                if nb is not None:
                    aw = _flat(R @ -np.asarray(nb, float))  # fly-through, level
                    ab = None if aw is None else R.T @ aw
                # Canonicalize: fly-through must roughly agree with the line of
                # sight, else the sample is an IPPE/back-face flip -- correct it
                # rather than letting one bad FIRST sample poison the EMA (its
                # 60 deg outlier reject would then block every GOOD sample).
                if ab is not None and float(np.dot(ab, gb)) < 0:
                    ab = -ab
                dets.append((float(g["conf"]), gb, ab))
        if dets and now >= self.dash_until:
            if self.t_b is not None:
                match, ma, md = None, None, 1e9
                for _, gb, ab in dets:
                    d = float(np.linalg.norm(gb - self.t_b))
                    if d < md:
                        md, match, ma = d, gb, ab
                if match is not None and md < p["match_dist"]:
                    self.t_b = match
                    self.a_b = _blend_axis(self.a_b, ma, p["axis_ema"])
                    self.last_seen = now
            else:
                near = [
                    (c, gb, ab)
                    for c, gb, ab in dets
                    if np.linalg.norm(gb) < p["acq_max"]
                ]
                if near:
                    _, self.t_b, self.a_b = max(near, key=lambda x: x[0])
                    self.last_seen = now

        # --- dash straight through after a pass, then BRAKE ------------------
        if now < self.dash_until:
            self.debug = {"phase": "DASH"}
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
                v_b = R.T @ np.asarray(vel, float)
                bp = float(np.clip(self.p["k_brake"] * v_b[0], -0.06, 0.18))
                br = float(
                    np.clip(
                        -self.p["roll_dir"] * self.p["k_brake"] * v_b[1], -0.15, 0.15
                    )
                )
                self.debug = {"phase": "BRAKE"}
                return Cmd(br, bp, 0.0, pos[2], f"BRAKE passed={self.n_passed}")

        # --- COMMIT: blind dash through the frozen, dead-reckoned gate --------
        if self.committing and (self.t_b is None or self.a_b is None):
            self.committing = False  # frozen state lost: unblock ingestion
        if self.committing:
            s, cross = _axis_coords(self.t_b, self.a_b)
            if s > p["pass_behind"]:
                self.n_passed += 1
                lat_at_plane = float(np.hypot(cross[1], cross[2]))
                self._drop()
                self.dash_until = now + p["dash_t"]
                self.brake_until = now + p["dash_t"] + p["brake_t"]
                self.debug = {"phase": "PASSED", "lat": lat_at_plane}
                return Cmd(
                    0.0,
                    -0.06,
                    0.0,
                    pos[2],
                    f"PASSED g{self.n_passed} off={lat_at_plane:.2f}",
                )
            if s < 0 and (
                abs(cross[1]) > p["abort_lat"] or abs(cross[2]) > p["abort_lat"]
            ):
                # Drifted outside the safe aperture before the plane: bail out
                # rather than clip the frame; brake, re-acquire, re-approach.
                self._drop()
                self.brake_until = now + 1.0
                self.debug = {"phase": "ABORT", "s": s}
                return Cmd(0.0, 0.05, 0.0, pos[2], f"ABORT s={s:+.1f}")
            aim_b = self.t_b + min(s + p["d_carrot"], p["d_post"]) * self.a_b
            return self._servo(
                aim_b,
                self.a_b,
                p["commit_speed"],
                R,
                pos,
                now,
                f"COMMIT s={s:+.1f}",
                True,
                s,
                cross,
            )

        # --- stale target: hold a NEAR one blind (detection always dies at
        # point-blank range), forget a FAR one and scan ------------------------
        if self.t_b is not None and self.last_seen is not None:
            unseen = now - self.last_seen
            near = float(np.linalg.norm(self.t_b)) < 6.0
            if unseen > (p["blind_max"] if near else p["lost_t"]):
                self._drop()
        if self.t_b is None:
            self._prev_t = None
            self.debug = {"phase": "SCAN"}
            return Cmd(
                0.0,
                0.02,
                self.scan_dir * p["scan_yaw"],
                pos[2],
                f"SCAN passed={self.n_passed}",
            )

        # --- APPROACH the pre-gate point on the fly-through axis --------------
        axis = self.a_b
        if axis is None:
            # No usable PnP normal yet: approach head-on along the line of
            # sight (flattened) -- degenerates to the old point-chase, but
            # still gains the commit phase.
            aw = _flat(R @ self.t_b)
            axis = (
                self.t_b / max(float(np.linalg.norm(self.t_b)), 1e-9)
                if aw is None
                else R.T @ aw
            )
        s, cross = _axis_coords(self.t_b, axis)
        yaw_axis = float(np.arctan2(axis[1], axis[0]))
        aligned = (
            abs(cross[1]) < p["align_lat"]
            and abs(cross[2]) < p["align_z"]
            and abs(yaw_axis) < p["align_yaw"]
        )
        if s >= -(p["d_pre"] + 0.5) and aligned:
            self.committing = True
            self.a_b = axis
            self._prev_t = None  # aim point jumps: restart the rate estimator
            aim_b = self.t_b + min(s + p["d_carrot"], p["d_post"]) * axis
            return self._servo(
                aim_b,
                axis,
                p["commit_speed"],
                R,
                pos,
                now,
                f"COMMIT s={s:+.1f}",
                True,
                s,
                cross,
            )
        if s < -(p["d_pre"] + 0.5):
            # Far out: carrot along the axis, never past the pre-gate point.
            d_aim = min(s + p["d_carrot"], -p["d_pre"])
            v_cap = p["max_speed"]
            if abs(cross[2]) > 2.0 * p["align_z"]:
                # Big vertical offset: descend/climb onto the axis BEFORE
                # charging -- measured (nav log 13:38): closing at full speed
                # with 4.4 m of vertical error reached the plane unaligned,
                # aborted, and lost the gate.
                v_cap = 0.6
        else:
            if s > -0.4:
                # At the plane but never aligned: do not creep into the frame.
                self._drop()
                self.brake_until = now + 1.0
                self.debug = {"phase": "ABORT", "s": s}
                return Cmd(0.0, 0.05, 0.0, pos[2], f"ABORT s={s:+.1f}")
            # Inside the pre-gate zone but off-axis: slide onto the axis,
            # barely creeping forward, until the alignment gate opens.
            d_aim = s + 0.5
            v_cap = 0.4
        aim_b = self.t_b + d_aim * axis
        return self._servo(
            aim_b,
            axis,
            v_cap,
            R,
            pos,
            now,
            f"GO s={s:+.1f} lat={float(cross[1]):+.1f}",
            False,
            s,
            cross,
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
        reassoc_radius=4.0,  # detection within this of a map entry refines it (else
        # new). Must stay well under the gate spacing: at 10.0 the NEXT gate's
        # detections re-associated with the current entry and EMA-dragged it to
        # a phantom point between the two gates (caught in closed-loop test)
        min_hits=3,  # a map entry must be seen this many times before we chase it
        ema=0.3,  # smoothing when refining a map entry / target
        rate_ema=0.4,  # smoothing for the position-derivative rates
        pass_cooldown=2.0,  # min seconds between counted passes
        min_pass_sep=6.0,  # ...and the drone must have moved this far since the last
        vel_dt=0.05,  # min interval for the position-derivative rates
        min_acq_fwd=1.5,  # only lock a mapped gate at least this far ahead (m)
        cone_half=0.7,  # ...and within this bearing (rad) -- else yaw-scan to it
        scan_yaw=0.35,  # yaw error injected while scanning for the next gate
        entry_ttl=4.0,  # drop a map entry unseen for this long (s) -- under a
        # drifting pose estimate stale entries become phantoms
        near_keep=6.0,  # ...EXCEPT the held target within this range: detection
        # always dies at point-blank; expiring it strands the drone at the gate
        d_pre=2.5,  # pre-gate point this far in front, on the axis (m)
        d_post=2.0,  # commit carries this far past the plane (m)
        d_carrot=1.5,  # pure-pursuit lookahead along the axis (m)
        align_lat=0.4,  # cross-track bound to enter COMMIT (m)
        align_z=0.4,  # vertical bound to enter COMMIT (m)
        align_yaw=0.26,  # yaw-to-axis bound to enter COMMIT (rad, ~15 deg)
        commit_speed=1.0,  # desired closing speed through the gate (m/s)
        pass_behind=0.7,  # plane crossed by this much -> PASSED (m)
        abort_lat=0.9,  # off-axis beyond this during COMMIT -> abort (m)
        dash_t=0.4,  # fly straight this long after a pass (through + clear)
        brake_t=1.4,  # then brake this long before chasing the next gate
        k_brake=0.25,  # brake lean per m/s of velocity
        max_brake=0.08,  # backward-tilt cap when braking (rad) -- was max_tilt:
        #                  closing-rate spikes then pitched the drone into
        #                  full-reverse flight
        yaw_track_clip=0.5,  # yaw toward the target (rad cap) so the gate stays
        #                      in the camera FOV all the way through
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
            pass_cooldown=pass_cooldown,
            min_pass_sep=min_pass_sep,
            vel_dt=vel_dt,
            min_acq_fwd=min_acq_fwd,
            cone_half=cone_half,
            scan_yaw=scan_yaw,
            entry_ttl=entry_ttl,
            near_keep=near_keep,
            d_pre=d_pre,
            d_post=d_post,
            d_carrot=d_carrot,
            align_lat=align_lat,
            align_z=align_z,
            align_yaw=align_yaw,
            commit_speed=commit_speed,
            pass_behind=pass_behind,
            abort_lat=abort_lat,
            dash_t=dash_t,
            brake_t=brake_t,
            k_brake=k_brake,
            max_brake=max_brake,
            yaw_track_clip=yaw_track_clip,
        )
        self.target = None  # world NED of the gate being flown
        self.gates_map = []  # [{"p", "n" hits, "t", "axis", "obs"}, ...]
        self.passed = []  # world NED of gates already flown through
        self.hold_z = None
        self.scan_dir = 1.0
        self.dash_until = 0.0
        self.brake_until = 0.0
        self.debug = {}
        self._commit = None  # frozen {"g": world pos, "a": world axis}
        self._target_ent = None
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
        # Tests bump n_passed by appending None -- skip those sentinels.
        return any(np.linalg.norm(pt - q) < r for q in self.passed if q is not None)

    def _finish_pass(self, g, off, pos, now):
        """Plane crossed during COMMIT: book the pass (guarded against
        re-counting one gate) and set up the dash/brake exit."""
        p = self.p
        moved = self._last_pass_pos is None or (
            np.linalg.norm(pos - self._last_pass_pos) > p["min_pass_sep"]
        )
        if (now - self._last_pass_t) > p["pass_cooldown"] and moved:
            self.passed.append(np.asarray(g, float).copy())
            self._last_pass_t = now
            self._last_pass_pos = pos.copy()
        self.gates_map = [
            e for e in self.gates_map if np.linalg.norm(e["p"] - g) > p["merge_radius"]
        ]
        self.hold_z = float(g[2])
        self.target = None
        self._target_ent = None
        self._commit = None
        self._prev_t = None
        self.dash_until = now + p["dash_t"]
        self.brake_until = now + p["dash_t"] + p["brake_t"]
        self.debug = {"phase": "PASSED", "lat": off}
        return Cmd(
            0.0, -0.05, 0.0, self.hold_z, f"PASSED g{self.n_passed} off={off:.2f}"
        )

    def _abort(self, s, pos, now):
        self.target = None
        self._target_ent = None
        self._commit = None
        self._prev_t = None
        self.brake_until = now + 1.0
        self.debug = {"phase": "ABORT", "s": s}
        # Hold CURRENT altitude: hold_z can be metres away (spawn / last gate)
        # and this fires right next to a gate frame -- no dive/climb here.
        return Cmd(0.0, 0.05, 0.0, pos[2], f"ABORT s={s:+.1f}")

    def _servo(
        self, aim_b, axis_b, to_g_b, v_cap, R_wb, pos, now, tag, yaw_on_axis, s, cross
    ):
        p = self.p
        fwd, lat = float(aim_b[0]), float(aim_b[1])
        dist = float(np.hypot(fwd, lat))
        closing, lat_rate = self._rates(fwd, lat, now)
        v_des = float(np.clip(dist, min(p["thru_speed"], v_cap), v_cap))
        lean = float(
            np.clip(p["k_fwd"] * (v_des - closing), -p["max_brake"], p["max_lean"])
        )
        tgt_pitch = -lean
        tgt_roll = float(
            np.clip(
                p["roll_dir"] * (p["kp_lat"] * lat + p["kd_lat"] * lat_rate),
                -p["max_tilt"],
                p["max_tilt"],
            )
        )
        if yaw_on_axis:
            yaw_err = float(np.arctan2(axis_b[1], axis_b[0]))
        else:
            # Camera on the GATE while approaching -- keeps detections flowing.
            yaw_err = float(np.arctan2(to_g_b[1], max(float(to_g_b[0]), 0.5)))
        yaw_err = float(np.clip(yaw_err, -p["yaw_track_clip"], p["yaw_track_clip"]))
        aim_w = pos + R_wb @ aim_b
        tgt_z = float(aim_w[2]) + p["zoff"]
        self.debug = {
            "phase": tag.split()[0],
            "s": s,
            "lat": float(cross[1]),
            "vert": float(cross[2]),
            "dist": float(np.linalg.norm(to_g_b)),
        }
        return Cmd(
            tgt_roll,
            tgt_pitch,
            yaw_err,
            tgt_z,
            f"{tag} c={closing:+.1f} passed={self.n_passed} map={len(self.gates_map)}",
        )

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
            nb = g["pose"].get("normal_body")
            thru_w = _flat(R_wb @ -np.asarray(nb, float)) if nb is not None else None
            obs_w = _flat(gw - pos)  # bearing fallback for the axis
            # Canonicalize: fly-through must roughly agree with the line of
            # sight, else the sample is an IPPE/back-face flip -- correct it
            # rather than letting one bad FIRST sample poison the EMA (its
            # 60 deg outlier reject would then block every GOOD sample).
            if thru_w is not None and obs_w is not None:
                if float(np.dot(thru_w, obs_w)) < 0:
                    thru_w = -thru_w
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
                best["axis"] = _blend_axis(best["axis"], thru_w, p["ema"])
                best["obs"] = _blend_axis(best["obs"], obs_w, p["ema"])
            else:
                self.gates_map.append(
                    {"p": gw, "n": 1, "t": now, "axis": thru_w, "obs": obs_w}
                )
        self.gates_map = [
            e
            for e in self.gates_map
            if not self._near_passed(e["p"])
            and (
                (now - e["t"]) < p["entry_ttl"]
                # The HELD target never expires at close range: detection
                # always dies at point-blank; expiring the entry here used to
                # strand the drone scanning right in front of the gate.
                or (
                    self.target is not None
                    and np.linalg.norm(e["p"] - self.target) < p["reassoc_radius"]
                    and np.linalg.norm(e["p"] - pos) < p["near_keep"]
                )
            )
        ]

        # --- target: keep current (snap to its refined entry) or acquire the
        # nearest CONFIRMED unpassed gate that is ahead and within the cone ---
        if self.target is not None:
            best, bd = None, 1e9
            for e in self.gates_map:
                d = np.linalg.norm(self.target - e["p"])
                if d < bd:
                    bd, best = d, e
            if best is not None and bd < p["reassoc_radius"]:
                self.target, self._target_ent = best["p"], best
            else:
                self.target, self._target_ent = None, None
        if self.target is None and self._commit is None:
            ahead = []
            for e in self.gates_map:
                if e["n"] < p["min_hits"]:
                    continue
                eb = Rt @ (e["p"] - pos)
                if eb[0] > p["min_acq_fwd"] and abs(eb[1]) < eb[0] * np.tan(
                    p["cone_half"]
                ):
                    ahead.append(e)
            if ahead:
                ent = min(ahead, key=lambda e: np.linalg.norm(e["p"] - pos))
                self.target, self._target_ent = ent["p"], ent

        # --- dash straight through after a pass, then BRAKE ------------------
        if now < self.dash_until:
            self.debug = {"phase": "DASH"}
            return Cmd(0.0, -0.03, 0.0, self.hold_z, f"DASH passed={self.n_passed}")
        if now < self.brake_until and self.target is None:
            v_b = Rt @ np.asarray(vel, float)
            bp = float(np.clip(p["k_brake"] * v_b[0], -0.06, 0.18))
            br = float(np.clip(-p["roll_dir"] * p["k_brake"] * v_b[1], -0.15, 0.15))
            self.debug = {"phase": "BRAKE"}
            # Current altitude, not hold_z: braking often happens beside a gate.
            return Cmd(br, bp, 0.0, pos[2], f"BRAKE passed={self.n_passed}")

        # --- COMMIT: blind dash through the frozen gate on dead reckoning ----
        if self._commit is not None:
            g, a = self._commit["g"], self._commit["a"]
            to_g_b = Rt @ (g - pos)
            axis_b = Rt @ a
            s, cross = _axis_coords(to_g_b, axis_b)
            if s > p["pass_behind"]:
                return self._finish_pass(
                    g, float(np.hypot(cross[1], cross[2])), pos, now
                )
            if s < 0 and (
                abs(cross[1]) > p["abort_lat"] or abs(cross[2]) > p["abort_lat"]
            ):
                return self._abort(s, pos, now)
            aim_b = to_g_b + min(s + p["d_carrot"], p["d_post"]) * axis_b
            return self._servo(
                aim_b,
                axis_b,
                to_g_b,
                p["commit_speed"],
                R_wb,
                pos,
                now,
                f"COMMIT s={s:+.1f}",
                True,
                s,
                cross,
            )

        # --- no target in view: yaw-SCAN to bring the next gate into FOV -----
        if self.target is None:
            self._prev_t = None
            self.debug = {"phase": "SCAN"}
            return Cmd(
                0.0,
                0.02,
                self.scan_dir * p["scan_yaw"],
                self.hold_z,
                f"SCAN n={self.n_passed} map={len(self.gates_map)}",
            )

        # --- APPROACH the pre-gate point on the fly-through axis -------------
        ent = self._target_ent or {}
        a = ent.get("axis")
        if a is None:
            a = ent.get("obs")
        if a is None:
            a = _flat(self.target - pos)
        if a is None:  # gate directly above/below: fly the line of sight
            a = _unit(self.target - pos)
        if float(np.dot(a, self.target - pos)) < 0:
            # We are BEHIND the gate plane relative to its fly-through axis.
            # Never flip the axis to "fix" it -- that commits the drone
            # backwards through the gate. Drop the target and re-scan.
            self.target, self._target_ent = None, None
            self._prev_t = None
            self.debug = {"phase": "SCAN"}
            return Cmd(
                0.0,
                0.02,
                self.scan_dir * p["scan_yaw"],
                self.hold_z,
                f"SCAN n={self.n_passed} map={len(self.gates_map)} (behind)",
            )
        to_g_b = Rt @ (self.target - pos)
        axis_b = Rt @ a
        s, cross = _axis_coords(to_g_b, axis_b)
        yaw_axis = float(np.arctan2(axis_b[1], axis_b[0]))
        aligned = (
            abs(cross[1]) < p["align_lat"]
            and abs(cross[2]) < p["align_z"]
            and abs(yaw_axis) < p["align_yaw"]
        )
        if s >= -(p["d_pre"] + 0.5) and aligned:
            self._commit = {"g": self.target.copy(), "a": a.copy()}
            self._prev_t = None  # aim point jumps: restart the rate estimator
            aim_b = to_g_b + min(s + p["d_carrot"], p["d_post"]) * axis_b
            return self._servo(
                aim_b,
                axis_b,
                to_g_b,
                p["commit_speed"],
                R_wb,
                pos,
                now,
                f"COMMIT s={s:+.1f}",
                True,
                s,
                cross,
            )
        if s < -(p["d_pre"] + 0.5):
            # Far out: carrot along the axis, never past the pre-gate point.
            d_aim = min(s + p["d_carrot"], -p["d_pre"])
            v_cap = p["max_speed"]
            if abs(cross[2]) > 2.0 * p["align_z"]:
                # Big vertical offset: descend/climb onto the axis BEFORE
                # charging -- measured (nav log 13:38): closing at full speed
                # with 4.4 m of vertical error reached the plane unaligned,
                # aborted, and lost the gate.
                v_cap = 0.6
        else:
            if s > -0.4:
                # At the plane but never aligned: do not creep into the frame.
                return self._abort(s, pos, now)
            # Inside the pre-gate zone but off-axis: slide onto the axis,
            # barely creeping forward, until the alignment gate opens.
            d_aim = s + 0.5
            v_cap = 0.4
        aim_b = to_g_b + d_aim * axis_b
        return self._servo(
            aim_b,
            axis_b,
            to_g_b,
            v_cap,
            R_wb,
            pos,
            now,
            f"GO s={s:+.1f} lat={float(cross[1]):+.1f}",
            False,
            s,
            cross,
        )
