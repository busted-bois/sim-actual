"""VQ2-legal observation — everything the policy is allowed to see at race time.

Why this exists alongside rl/core/observation.py
------------------------------------------------
The 24-D vector in observation.py is PRIVILEGED: it is built from world
position, world velocity, orientation and an absolute gate map. Measured on
the live simulator (2026-08-01 probe, Qualification session): ODOMETRY,
ATTITUDE and LOCAL_POSITION_NED never arrive even when explicitly requested at
100 Hz after a GCS heartbeat, and the track burst reports track_info=0 -- no
gate poses. So a policy trained on that vector cannot be fed at race time. It
is not a plumbing gap; the observation contract itself is unusable.

This module defines what IS available:

  * ONE gate estimate per tick from YOLO-pose + PnP (gp_vision publishes a
    single best gate -- there is no next-gate lookahead anywhere in the stack),
  * gyro (measured PRESENT at 114 Hz),
  * a gravity direction from an AHRS (encodes roll/pitch; yaw is NOT
    observable -- mag is NaN -- and is deliberately absent),
  * our own last action,
  * the gate index from race_status (measured PRESENT at 4.4 Hz).

Deliberate omissions, each for a measured reason
------------------------------------------------
* No velocity. The only estimator ever graded against truth had velocity RMSE
  28.9 m/s, and fusing vision made it 2.7x WORSE than IMU alone. Instead the
  policy gets the honest time-derivatives of the vision features it can
  actually measure (bearing rate, log-range rate) and infers the rest through
  memory -- which is the whole reason the policy is recurrent.
* No yaw / heading. No magnetometer; measured yaw error was 177-179 deg.
* No next-gate vector. The previous attempt hardcoded those four dimensions to
  constants, so 4 of its 24 inputs carried zero information.
* No absolute metric range as a decision threshold. PnP scale rests on an
  assumed gate width; apparent_size is carried separately because angular size
  is measured directly and survives a wrong metric assumption.

Discontinuity handling is load-bearing
--------------------------------------
A detector re-lock onto a different gate is a jump, not motion. Scoring it as
motion is what produced the previous attempt's mean per-episode progress of
-42 while it was flying forward at +26. Feature rates are zeroed across any
gate-index change or implausible jump.

    uv run python -m unittest tests.test_vq2_observation
"""

from __future__ import annotations

import numpy as np

# Gate geometry. simulator/gate_pnp.py -- the code that actually flies -- uses
# a 1.5 m inner opening inside a 2.7 m outer frame, and that is corroborated by
# measurement: over 551 detected gates the projected inner/outer width ratio is
# 0.560 +/- 0.014 against the 1.5/2.7 = 0.556 the model assumes. Note that
# rl/core/spec.py's GATE_SIZE_M = 2.72 is applied as the OPENING, which is the
# outer width and therefore scales range by 1.81x. Do not use it here.
GATE_OPENING_M = 1.5

RANGE_REF_M = 10.0  # log-range is measured about this
GYRO_SCALE = 3.0  # rad/s -> O(1); racing rates far exceed spec's 0.6 clamp
STALE_MAX_S = 1.0  # detection age at which staleness saturates
CONF_DECAY_S = 1.0  # held confidence reaches 0 after this long with no fix
SINCE_GATE_MAX_S = 10.0  # time-since-gate normalizer (also a stall signal)
MIN_RANGE_M = 0.5  # guard against divide-by-zero on a gate in our face
MIN_FORWARD_M = 0.1  # a gate at or behind the lens is not a target
MIN_KP_FOR_NORMAL = 4  # IPPE needs 4 coplanar points for a real plane normal
MAX_LOG_RANGE_JUMP = 0.35  # |dlog(range)| above this in one tick = re-lock
OBS_ABS_MAX = 3.0  # every field is clipped to +/- this

OBS_LAYOUT = {
    "gate_dir_body": slice(0, 3),  # unit vector to gate, body FRD
    "gate_dir_rate": slice(3, 6),  # d(gate_dir)/dt, 1/s
    "log_range": slice(6, 7),  # log(range / RANGE_REF_M)
    "log_range_rate": slice(7, 8),  # d(log_range)/dt, 1/s
    "apparent_size": slice(8, 9),  # angular size, metric-assumption-free
    "gate_normal_body": slice(9, 12),  # gate plane normal (zeroed if untrusted)
    "normal_valid": slice(12, 13),  # 1 if the normal came from real keypoints
    "n_visible": slice(13, 14),  # keypoints used, / 8
    "conf": slice(14, 15),  # detector confidence, decayed while stale
    "detected": slice(15, 16),  # 1 = fresh valid fix this tick
    "staleness": slice(16, 17),  # age of the held fix, / STALE_MAX_S
    "gyro": slice(17, 20),  # body rates / GYRO_SCALE
    "gravity_body": slice(20, 23),  # gravity direction in body (roll/pitch)
    "last_action": slice(23, 27),  # our previous command, [-1, 1]
    "gate_idx": slice(27, 28),  # active_gate_index / (n_gates - 1)
    "since_gate": slice(28, 29),  # time since last pass / SINCE_GATE_MAX_S
}

OBS_DIM = 29


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(3)


def _valid_estimate(est) -> bool:
    """Accept only a fix we would be willing to steer on."""
    if not est:
        return False
    gb = np.asarray(est.get("gate_pos_body"), dtype=float).reshape(-1)
    if gb.size != 3 or not np.all(np.isfinite(gb)):
        return False
    # gate_pnp reconstructs lateral/vertical from the centre pixel, so a
    # non-positive forward component means the solve collapsed.
    return bool(gb[0] > MIN_FORWARD_M)


class GateFeatureTracker:
    """Per-tick vision + IMU -> the VQ2 observation vector.

    Stateful on purpose: staleness, feature rates and time-since-gate all need
    history, and holding the last fix through a dropout is what keeps the
    policy input continuous at 12-16 Hz perception against a 30-50 Hz loop.
    """

    def __init__(self, n_gates: int = 17):
        self.n_gates = max(int(n_gates), 2)
        self.reset()

    def reset(self) -> None:
        self._dir = np.zeros(3)
        self._normal = np.zeros(3)
        self._normal_valid = 0.0
        self._log_range = 0.0
        self._apparent = 0.0
        self._n_visible = 0.0
        self._conf = 0.0
        self._dir_rate = np.zeros(3)
        self._log_range_rate = 0.0
        self._t_fix = None  # time of the last accepted fix
        self._t_prev = None  # time of the previous accepted fix
        self._gate_idx = 0
        self._t_gate = None  # time the gate index last advanced
        self._have_fix = False

    def update(
        self,
        t: float,
        vision_est: dict | None,
        gyro,
        gravity_body,
        last_action,
        gate_idx: int,
    ) -> np.ndarray:
        """Build the observation for this tick.

        t            : monotonic seconds. HIGHRES_IMU.time_usec is frozen on
                       this sim (measured: 1141 samples, 0 us span), so callers
                       must pass an arrival-time clock, never a sensor stamp.
        vision_est   : gate_pnp._result_dict-shaped dict, or None if no fix.
        gyro         : body rates (roll, pitch, yaw) rad/s.
        gravity_body : gravity direction in body frame (need not be unit).
        last_action  : our previous normalized command, 4-D in [-1, 1].
        gate_idx     : active_gate_index from race_status.
        """
        t = float(t)
        gate_idx = int(np.clip(gate_idx, 0, self.n_gates - 1))

        if self._t_gate is None:
            self._t_gate = t

        # A gate advance invalidates the held fix: it describes the gate we
        # just went through, not the one ahead.
        advanced = gate_idx != self._gate_idx
        if advanced:
            self._gate_idx = gate_idx
            self._t_gate = t
            self._have_fix = False
            self._dir_rate = np.zeros(3)
            self._log_range_rate = 0.0
            self._t_prev = None

        fresh = _valid_estimate(vision_est)
        if fresh:
            self._ingest(t, vision_est, discontinuous=advanced)

        return self._assemble(t, fresh, gyro, gravity_body, last_action)

    def _ingest(self, t: float, est: dict, discontinuous: bool) -> None:
        gb = np.asarray(est["gate_pos_body"], dtype=float).reshape(3)
        rng = max(float(np.linalg.norm(gb)), MIN_RANGE_M)
        new_dir = _unit(gb)
        new_log_range = float(np.log(rng / RANGE_REF_M))

        dt = t - self._t_fix if self._t_fix is not None else 0.0
        jumped = (
            discontinuous
            or not self._have_fix
            or dt <= 1e-6
            or abs(new_log_range - self._log_range) > MAX_LOG_RANGE_JUMP
        )
        if jumped:
            # Not motion — a re-lock or a first sighting. Rates stay at zero
            # rather than reporting a metres-per-tick spike.
            self._dir_rate = np.zeros(3)
            self._log_range_rate = 0.0
        else:
            self._dir_rate = (new_dir - self._dir) / dt
            self._log_range_rate = (new_log_range - self._log_range) / dt

        self._dir = new_dir
        self._log_range = new_log_range
        self._apparent = GATE_OPENING_M / rng

        n_vis = int(est.get("n_visible", 0) or 0)
        self._n_visible = n_vis / 8.0
        self._conf = float(est.get("conf", 0.0) or 0.0)

        # The edge-pair fallback fabricates a straight-facing normal AND
        # reports reproj_px = 0.0, so it outranks every real solve on a naive
        # quality filter. Trust the normal only with enough real keypoints.
        nrm = np.asarray(est.get("normal_body", (0, 0, 0)), dtype=float).reshape(-1)
        if n_vis >= MIN_KP_FOR_NORMAL and nrm.size == 3 and np.all(np.isfinite(nrm)):
            self._normal = _unit(nrm)
            self._normal_valid = 1.0
        else:
            self._normal = np.zeros(3)
            self._normal_valid = 0.0

        self._t_prev = self._t_fix
        self._t_fix = t
        self._have_fix = True

    def _assemble(self, t, fresh, gyro, gravity_body, last_action) -> np.ndarray:
        age = (
            (t - self._t_fix) if (self._have_fix and self._t_fix is not None) else None
        )
        staleness = 1.0 if age is None else min(age / STALE_MAX_S, 1.0)
        decay = 0.0 if age is None else max(0.0, 1.0 - age / CONF_DECAY_S)
        since_gate = min(max(t - self._t_gate, 0.0) / SINCE_GATE_MAX_S, 1.0)

        gyro = np.asarray(gyro, dtype=float).reshape(-1)[:3]
        gyro = np.nan_to_num(gyro, nan=0.0, posinf=0.0, neginf=0.0)
        grav = np.asarray(gravity_body, dtype=float).reshape(-1)[:3]
        grav = _unit(np.nan_to_num(grav, nan=0.0, posinf=0.0, neginf=0.0))
        act = np.asarray(last_action, dtype=float).reshape(-1)[:4]
        act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        L = OBS_LAYOUT
        obs[L["gate_dir_body"]] = self._dir
        obs[L["gate_dir_rate"]] = self._dir_rate
        obs[L["log_range"]] = self._log_range
        obs[L["log_range_rate"]] = self._log_range_rate
        obs[L["apparent_size"]] = self._apparent
        obs[L["gate_normal_body"]] = self._normal
        obs[L["normal_valid"]] = self._normal_valid
        obs[L["n_visible"]] = self._n_visible
        obs[L["conf"]] = self._conf * decay
        obs[L["detected"]] = 1.0 if fresh else 0.0
        obs[L["staleness"]] = staleness
        obs[L["gyro"]] = gyro / GYRO_SCALE
        obs[L["gravity_body"]] = grav
        obs[L["last_action"]] = np.clip(act, -1.0, 1.0)
        obs[L["gate_idx"]] = self._gate_idx / (self.n_gates - 1)
        obs[L["since_gate"]] = since_gate

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(obs, -OBS_ABS_MAX, OBS_ABS_MAX, out=obs)
        return obs
