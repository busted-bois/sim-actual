"""Classical HSV dual-cyan corridor detection (no YOLO).

Two parallel blue ribbons define left/right track edges. The frame is split
into horizontal bands; within each band the mask's contiguous column RUNS are
the rails, so left/right come from the corridor's own geometry rather than
from which half of the image a blob happens to sit in.

Replaces an earlier centroid detector that split each ROI at a fixed w//2.
Four defects that split caused, all reproduced on synthetic frames:
  * corridor wholly inside one image half -> both rails scored as one side,
    and the "only one side" fallback (mid = rail +- 0.25*w) pulled the answer
    back PAST centre: true cx +0.44 came out as -0.06, i.e. sign-inverted
    exactly when the drone is off the corridor and the correction matters.
  * a hard bend puts both far rails in one half, so the far/near mid-x
    comparison inverted too: true +36 deg bend read as -1 deg.
  * MORPH_OPEN 5x5 erased the far ribbon entirely (830 px -> 0), which is why
    heading was 0 in 75-88% of logged frames -- the far band was empty.
  * the cyan band reached H=140, so blue sky (H~113) entered the mask wholesale.

The mask now comes from track_line.track_mask, whose ranges and CC filter were
measured on real flight frames (H capped at 106, close-only morphology).
heading_err uses detect_track's polyfit/sign convention so the two detectors
stay interchangeable where GPPilot.TrackVirtualGate consumes either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from simulator.gp_vision import env_flag
from simulator.track_line import track_mask

_DOWNSCALE = 2  # detection resolution; outputs are full-res / normalized

# Band layout mirrors detect_track: skip the timer strip, keep bands fine
# enough that a near-horizontal ribbon at a sharp turn still spans several.
_N_BANDS = 12
_BAND_TOP_FRAC = 0.08

# --- Measured on runs/videos/vision.mp4 frames 900-1500 (scripts/bl_replay.py)
# The 20 deg up-tilted camera puts the corridor in the BOTTOM THIRD of the
# frame: mask rows run p05=0.29 .. p95=0.97 with the median at 0.73, so of the
# 12 evenly-spread bands the top four hold any pixels on only 9-12% of frames.
# Four consequences, all reproduced by the replay harness:
#   * band centres span a median of 0.24 of frame height, so the polyfit that
#     produces heading_err has almost no lever arm -> +-70 deg swings, 2.9
#     sign flips/s, and |d hdg| pinned at the tracker's 25 deg/frame limiter at
#     BOTH p90 and max (the limiter was hiding the estimate, not smoothing it).
#   * conf, being band-count driven, sits at p50 0.60 and falls below the
#     pilot's TURN_FF_MIN_CONF on 40% of found frames.
#   * fit_l and fit_r, fitted independently, cross at a row a band actually
#     occupies on 15.4% of frames -> the inferred side flips and the centre
#     lands a full corridor-width wrong.
#   * cx_norm had no filter of any kind while heading had a rate limiter, and
#     jumps up to 0.709 in one frame straight into TRACK_LAT_GAIN.
# MEASURED NEGATIVE RESULT — default OFF, kept so it is not retried blind.
# Concentrating the 12 bands on the rows that hold ribbon looks obviously right
# (the top four are empty ~90% of the time) and does raise the band count 4 -> 9.
# It makes every quality metric WORSE: span p50 unchanged at 0.24, heading valid
# 25.2% -> 22.0%, hdg flips 2.25 -> 2.57/s, cx flips 2.33 -> 2.64/s, |d cx| max
# 0.709 -> 1.028. Packing the same rows into more, thinner bands adds no
# information — the span is set by where the ribbon is VISIBLE, not by how it is
# sliced — while each thinner band carries fewer pixels and is noisier.
_ADAPTIVE_BANDS = env_flag("GP_BL_ADAPTIVE_BANDS", False)
_CENTRE_FIT = env_flag("GP_BL_CENTRE_FIT", True)
_HDG_GATE = env_flag("GP_BL_HDG_GATE", True)
_CX_FILTER = env_flag("GP_BL_CX_FILTER", True)
_CONF_V2 = env_flag("GP_BL_CONF_V2", True)
_CONF_V2_FULL_BANDS = 4.0  # measured p50; the old 5.0 was never reachable

# Heading is only reported when the band centres span enough of the frame to
# constrain a slope, and the fit actually explains them. Below that the honest
# answer is "no heading", not a number the pilot will bank on.
#
# 0.25 chosen by sweeping the threshold over the replay clip. It is not a
# quality/availability trade — it DOMINATES the neighbours on both axes:
#     span_thr  heading valid  |d hdg| p90  sign flips/s
#       0.00        77.4%        25.00        3.50     <- pinned at the limiter
#       0.20        49.7%        10.10        3.16
#       0.25        34.4%         6.29        2.01     <- best flip rate
#       0.30        25.2%         4.34        2.25
#       0.40        19.1%         1.25        2.31
# Below 0.20 the fit is so unconstrained it rides the rate limiter; above 0.25
# the surviving frames are fewer without flipping any less often.
_MIN_HDG_SPAN_FRAC = 0.25
_MAX_HDG_RESID_PX = 12.0  # downscaled px RMS about the fitted centreline
# Sign is what the pilot's turn feed-forward consumes, so flips cost more than
# lag. Measured at the 0.25 threshold: EMA 1.0 -> 2.01 flips/s, 0.35 -> 1.28,
# 0.25 -> 0.91, for ~0.13 s of lag against turns that last seconds.
_HDG_EMA = 0.25

# cx guards, mirroring the heading pair. 0.15 is just above the measured p90
# per-frame step (0.133), so ordinary motion passes and the 0.709 outlier does not.
_MAX_CX_STEP = 0.15
_CX_EMA = 0.6  # on the accepted value; 1.0 disables

# Pixel thresholds are quoted at FULL resolution and scaled by _DOWNSCALE.
_MIN_TOTAL_PX_FULLRES = 120  # whole-frame floor: below this there is no line
_MIN_BAND_PX_FULLRES = 24
_MIN_RUN_PX_FULLRES = 12  # a thinner column run is speckle, not a rail
_RUN_MERGE_GAP_FULLRES = 8  # bridge gaps this small INSIDE one rail
_MIN_RAIL_SEP_FULLRES = 30  # two runs closer than this are one broken rail

# Glow-flood reject (detect_track's rule): wide AND densely filled is a bloom;
# wide-but-thin is the real ribbon seen side-on when banked, so it is kept.
_MAX_BAND_SPAN_FRAC = 0.60
_FLOOD_FILL_FRAC = 0.20

_MIN_VALID_BANDS = 2
_CONF_FULL_BANDS = 5.0  # this many valid bands already means a solid lock

# Altitude cue reads the mid-frame ribbon (elevated path), not near floor paint.
_ALT_BAND_LO, _ALT_BAND_HI = 0.25, 0.65

# Last-resort corridor half-width when a single rail is visible and the tracker
# has no memory yet (first frames after a reset).
_DEFAULT_HALF_WIDTH_FRAC = 0.22

# Tracker guards.
_MAX_HDG_STEP_DEG = 25.0  # per-frame heading jump beyond this is not physical
_HALF_WIDTH_EMA = 0.3


@dataclass(frozen=True)
class BlueLineEstimate:
    found: bool
    cx_norm: float = 0.0  # corridor mid, -1 left .. +1 right
    cy_norm: float = 0.0  # avg near height, -1 top .. +1 bottom
    heading_err: float = 0.0  # rad, + = corridor vanishes right of center
    width_norm: float = 0.0  # left-right separation / image width
    left_found: bool = False
    right_found: bool = False
    conf: float = 0.0  # 0-1 detection quality (bands seen, rails paired)
    single_rail: bool = False  # no band saw BOTH rails; centre is inferred
    frame_id: int = 0
    points: tuple[tuple[float, float], ...] = field(default=())  # corridor centres
    # heading_err is meaningless unless the band centres span enough rows to
    # constrain a slope. Consumers must gate on this rather than on
    # left_found/right_found, which are frame-global and go True as soon as ONE
    # band pairs (28.6% of centres are inferred from a single rail + memory).
    heading_valid: bool = False
    span_frac: float = 0.0  # band-centre vertical span / frame height
    paired_bands: int = 0  # bands that resolved BOTH rails themselves


def cyan_mask(bgr: np.ndarray) -> np.ndarray:
    """Full-res ribbon mask (thin wrapper over the shared track_line mask)."""
    mask = track_mask(bgr, _DOWNSCALE)
    h, w = bgr.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _band_runs(
    band: np.ndarray, y0: int, sw: int, min_px: int, min_run_px: int, gap: int, sep: int
) -> tuple[float, list[tuple[float, int]]] | None:
    """One band -> (row, [(rail_x, px), ...]) with rails ordered left to right."""
    counts = np.count_nonzero(band, axis=0)
    nz = np.flatnonzero(counts)
    if nz.size == 0:
        return None
    total = int(counts.sum())
    if total < min_px:
        return None
    span = int(nz[-1] - nz[0])
    if span > _MAX_BAND_SPAN_FRAC * sw and total > _FLOOD_FILL_FRAC * band.size:
        return None  # glow flood

    runs: list[tuple[float, int]] = []
    for grp in np.split(nz, np.flatnonzero(np.diff(nz) > gap) + 1):
        c = counts[grp]
        n = int(c.sum())
        if n < min_run_px:
            continue
        runs.append((float((grp * c).sum() / n), n))
    if not runs:
        return None

    # Two runs closer together than a rail width apart are one broken rail.
    merged = [runs[0]]
    for cx, n in runs[1:]:
        pcx, pn = merged[-1]
        if cx - pcx < sep:
            merged[-1] = ((pcx * pn + cx * n) / (pn + n), pn + n)
        else:
            merged.append((cx, n))

    rows = np.count_nonzero(band, axis=1)
    row_sum = int(rows.sum())
    v = y0 + (float((np.arange(rows.size) * rows).sum() / row_sum) if row_sum else 0.0)
    return v, merged


def _fit(vs: list[float], us: list[float]):
    """u(v) through the samples: linear when possible, else constant."""
    if len(vs) >= 2:
        a, b = np.polyfit(np.asarray(vs), np.asarray(us), 1)
        return lambda v: float(a * v + b)
    u0 = us[0]
    return lambda v: float(u0)


def _band_edges(mask: np.ndarray, sh: int) -> np.ndarray:
    """Row boundaries for the N bands, placed where the ribbon actually is.

    A fixed 0.08..1.0 spread wastes the top four bands on rows that hold no
    ribbon on ~90% of frames, leaving ~4 usable samples out of 12 crowded into
    the bottom third. Spanning the mask's own 5th-95th row percentiles puts all
    12 where they can resolve something. Falls back to the fixed layout when
    the mask is too sparse or too thin to define a span.
    """
    fixed = np.linspace(int(sh * _BAND_TOP_FRAC), sh, _N_BANDS + 1).astype(int)
    if not _ADAPTIVE_BANDS:
        return fixed
    rows = np.count_nonzero(mask, axis=1).astype(np.float64)
    total = rows.sum()
    if total <= 0:
        return fixed
    cum = np.cumsum(rows) / total
    lo = int(np.searchsorted(cum, 0.05))
    # ONLY the top edge adapts. Moving the bottom edge too re-cuts the nearest
    # bands every frame, and cx_norm is the pixel-weighted mean of the nearest
    # THREE — a band grid that breathes makes cx non-stationary (measured: cx
    # sign flips rose 2.33 -> 2.45/s when both edges moved). Anchoring the
    # bottom keeps cx sampling the same rows while the top still reaches up to
    # collect the far samples the heading fit needs.
    lo = max(lo, int(sh * _BAND_TOP_FRAC))
    if sh - lo < max(_N_BANDS, int(0.25 * sh)):
        return fixed  # too little ribbon to define a span; keep the fixed grid
    return np.linspace(lo, sh, _N_BANDS + 1).astype(int)


def detect_blue_lines(
    bgr: np.ndarray,
    frame_id: int = 0,
    *,
    return_mask: bool = False,
    hint_half_width_px: float | None = None,
    hint_side: int | None = None,
) -> tuple[BlueLineEstimate, np.ndarray | None]:
    """Detect the dual-cyan corridor; return estimate and optional mask.

    Pure function. `hint_half_width_px` / `hint_side` (-1 left rail, +1 right
    rail, full-res px) let BlueLineTracker carry corridor width across frames so
    a single visible rail does not snap the centre to a fixed guess.
    """
    h, w = bgr.shape[:2]
    ds = _DOWNSCALE
    mask = track_mask(bgr, ds)
    sh, sw = mask.shape[:2]

    def _out(est: BlueLineEstimate) -> tuple[BlueLineEstimate, np.ndarray | None]:
        if not return_mask:
            return est, None
        full = mask
        if (sh, sw) != (h, w):
            full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return est, full

    area = ds * ds
    if cv2.countNonZero(mask) * area < _MIN_TOTAL_PX_FULLRES:
        return _out(BlueLineEstimate(found=False, frame_id=frame_id))

    min_px = max(4, _MIN_BAND_PX_FULLRES // area)
    min_run_px = max(3, _MIN_RUN_PX_FULLRES // area)
    gap = max(2, _RUN_MERGE_GAP_FULLRES // ds)
    sep = max(4, _MIN_RAIL_SEP_FULLRES // ds)

    edges = _band_edges(mask, sh)
    bands = []  # near (bottom) first
    for i in range(_N_BANDS - 1, -1, -1):
        r = _band_runs(
            mask[edges[i] : edges[i + 1]], edges[i], sw, min_px, min_run_px, gap, sep
        )
        if r is not None:
            bands.append(r)
    if not bands:
        return _out(BlueLineEstimate(found=False, frame_id=frame_id))

    # Bands that resolved BOTH rails define the corridor; they also calibrate
    # the half-width used to place the centre on single-rail bands.
    pv, pl, pr, phw = [], [], [], []
    for v, runs in bands:
        if len(runs) >= 2 and (runs[-1][0] - runs[0][0]) >= sep:
            pv.append(v)
            pl.append(runs[0][0])
            pr.append(runs[-1][0])
            phw.append(0.5 * (runs[-1][0] - runs[0][0]))

    single_rail = not pv
    if pv:
        fit_hw = _fit(pv, phw)
        if _CENTRE_FIT:
            # Fit the CENTRELINE and the half-width, then rebuild the rails as
            # centre +- half_width. Fitting the two rails independently lets
            # them cross (15.4% of frames), and where they cross the nearest-fit
            # side test below inverts — putting the corridor centre a full width
            # on the wrong side, the exact failure this detector replaced.
            fit_c = _fit(pv, [0.5 * (a + b) for a, b in zip(pl, pr)])

            def fit_l(v):
                return fit_c(v) - abs(fit_hw(v))

            def fit_r(v):
                return fit_c(v) + abs(fit_hw(v))
        else:
            fit_l, fit_r = _fit(pv, pl), _fit(pv, pr)
    else:
        hw0 = (
            hint_half_width_px / ds
            if hint_half_width_px
            else _DEFAULT_HALF_WIDTH_FRAC * sw
        )
        fit_l = fit_r = None
        fit_hw = lambda v: hw0  # noqa: E731 - constant fallback

    centres: list[tuple[float, float, int]] = []  # (v, u, px)
    left_found = right_found = False
    for v, runs in bands:
        half_w = max(fit_hw(v), sep * 0.5)
        px = sum(n for _, n in runs)
        if len(runs) >= 2 and (runs[-1][0] - runs[0][0]) >= sep:
            centres.append((v, 0.5 * (runs[0][0] + runs[-1][0]), px))
            left_found = right_found = True
            continue
        u = runs[0][0]
        if fit_l is not None:
            # Side comes from whichever fitted rail this run continues.
            side = 1 if abs(u - fit_r(v)) < abs(u - fit_l(v)) else -1
        elif hint_side is not None:
            side = hint_side
        else:
            # No memory at all: a rail left of centre is the left rail.
            side = -1 if u < sw * 0.5 else 1
        centres.append((v, u + (-half_w if side > 0 else half_w), px))
        if side > 0:
            right_found = True
        else:
            left_found = True

    if len(centres) < _MIN_VALID_BANDS:
        return _out(BlueLineEstimate(found=False, frame_id=frame_id))

    vs = [c[0] for c in centres]
    us = [c[1] for c in centres]
    ws = [float(c[2]) for c in centres]

    # Lateral error from the NEAREST bands only, pixel-weighted. detect_track's
    # geometry review: a single near band can flap +-0.5 between near-identical
    # frames when it catches a fragment of one rail, so average three.
    k = min(3, len(us))
    cx_px = float(np.average(us[:k], weights=ws[:k]))
    cx_norm = float(np.clip((cx_px - sw * 0.5) / (sw * 0.5), -1.5, 1.5))

    # Altitude cue: mid-frame ribbon height, near floor paint excluded.
    alt = [
        (v, wt)
        for (v, _u, wt) in zip(vs, us, ws)
        if _ALT_BAND_LO * sh <= v <= _ALT_BAND_HI * sh
    ]
    if alt:
        alt_v = float(np.average([a[0] for a in alt], weights=[a[1] for a in alt]))
    else:
        alt_v = float(np.average(vs, weights=ws))
    cy_norm = float(np.clip((alt_v - sh * 0.5) / (sh * 0.5), -1.5, 1.5))

    # Heading: slope of the corridor centreline. u = a*v + b; near = large v, so
    # the far end lies right of the near end when a < 0 (detect_track's sign).
    # Only meaningful if the centres span enough rows to constrain the slope AND
    # the line explains them: a 0.24-of-frame span (the measured median) puts the
    # slope's uncertainty far above the +-25 deg the tracker rate-limits it to,
    # which is why heading used to swing +-70 deg and flip sign 2.9 times/s.
    span_frac = (max(vs) - min(vs)) / sh if len(vs) >= 2 else 0.0
    heading_err = 0.0
    heading_valid = False
    if len(vs) >= 2:
        av = np.asarray(vs)
        au = np.asarray(us)
        a, b = np.polyfit(av, au, 1)
        heading_err = float(np.arctan2(-float(a), 1.0))
        resid = float(np.sqrt(np.mean((au - (a * av + b)) ** 2)))
        heading_valid = span_frac >= _MIN_HDG_SPAN_FRAC and resid <= _MAX_HDG_RESID_PX
        if _HDG_GATE and not heading_valid:
            heading_err = 0.0

    width_norm = float(np.mean(phw) * 2.0 / sw) if phw else 0.0

    paired_frac = len(pv) / len(centres)
    conf = float(
        np.clip(len(centres) / _CONF_FULL_BANDS, 0.0, 1.0) * (0.5 + 0.5 * paired_frac)
    )
    if _CONF_V2:
        # Band count alone says how much was sampled, not how well. Two frames
        # with 4 bands are not equally trustworthy when one has them stacked in
        # the bottom eighth of the image and the other spread over a third. Fold
        # in the geometry the pilot's lateral gain actually depends on, so
        # `conf` can drive blend (WEAK_BLEND_SCALE) as a real quality number
        # rather than a band tally. _CONF_FULL_BANDS is rebased to the count
        # that is genuinely reachable here (measured p50 = 4 of 12, and the
        # top four bands are empty ~90% of frames) instead of an unreachable 5.
        span_q = float(np.clip(span_frac / _MIN_HDG_SPAN_FRAC, 0.0, 1.0))
        conf = float(
            np.clip(len(centres) / _CONF_V2_FULL_BANDS, 0.0, 1.0)
            * (0.5 + 0.5 * paired_frac)
            * (0.6 + 0.4 * span_q)
        )

    est = BlueLineEstimate(
        found=True,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        heading_err=heading_err,
        width_norm=width_norm,
        left_found=left_found,
        right_found=right_found,
        conf=conf,
        single_rail=single_rail,
        frame_id=frame_id,
        points=tuple((u * ds, v * ds) for v, u, _ in centres),
        heading_valid=heading_valid,
        span_frac=float(span_frac),
        paired_bands=len(pv),
    )
    return _out(est)


class BlueLineTracker:
    """Frame-to-frame memory around the pure detector.

    Carries corridor half-width and rail side so single-rail frames keep a
    continuous centre estimate, and rate-limits heading so no single bad frame
    can saturate the pilot's yaw command (5% of logged frames used to exceed
    20 deg, peaking at 85 deg).
    """

    def __init__(self):
        self._half_width_px: float | None = None
        self._side: int | None = None
        self._hdg: float | None = None
        self._cx: float | None = None

    def reset(self) -> None:
        self._half_width_px = None
        self._side = None
        self._hdg = None
        self._cx = None

    def update(
        self, bgr: np.ndarray, frame_id: int = 0, *, return_mask: bool = False
    ) -> tuple[BlueLineEstimate, np.ndarray | None]:
        est, mask = detect_blue_lines(
            bgr,
            frame_id,
            return_mask=return_mask,
            hint_half_width_px=self._half_width_px,
            hint_side=self._side,
        )
        if not est.found:
            self._hdg = None
            self._cx = None
            return est, mask

        w = bgr.shape[1]
        if not est.single_rail and est.width_norm > 0.0:
            hw = 0.5 * est.width_norm * w
            self._half_width_px = (
                hw
                if self._half_width_px is None
                else _HALF_WIDTH_EMA * hw
                + (1.0 - _HALF_WIDTH_EMA) * self._half_width_px
            )
        if est.left_found != est.right_found:
            self._side = -1 if est.left_found else 1
        elif not est.single_rail:
            self._side = None

        cx, hdg, conf = est.cx_norm, est.heading_err, est.conf

        # cx had NO filter while heading had a rate limiter, yet cx is the term
        # that reaches the roll command through TRACK_LAT_GAIN: measured jumps
        # of up to 0.709 in a single frame, sign-flipping 2.3 times/s. Same
        # treatment as heading — clamp an implausible step and mark the frame
        # suspect — then a light EMA on what survives.
        if _CX_FILTER and self._cx is not None:
            if abs(cx - self._cx) > _MAX_CX_STEP:
                cx = self._cx + math.copysign(_MAX_CX_STEP, cx - self._cx)
                conf *= 0.5
            cx = _CX_EMA * cx + (1.0 - _CX_EMA) * self._cx

        # Only rate-limit a heading we are actually going to publish; a gated
        # heading is 0.0 by definition and must not drag the limiter's memory.
        if est.heading_valid or not _HDG_GATE:
            max_step = math.radians(_MAX_HDG_STEP_DEG)
            if self._hdg is not None:
                if abs(hdg - self._hdg) > max_step:
                    hdg = self._hdg + math.copysign(max_step, hdg - self._hdg)
                    conf *= 0.5  # the frame is suspect, not unusable
                hdg = _HDG_EMA * hdg + (1.0 - _HDG_EMA) * self._hdg
            self._hdg = hdg
        else:
            self._hdg = None

        self._cx = cx
        if (cx, hdg, conf) != (est.cx_norm, est.heading_err, est.conf):
            est = BlueLineEstimate(
                found=est.found,
                cx_norm=cx,
                cy_norm=est.cy_norm,
                heading_err=hdg,
                width_norm=est.width_norm,
                left_found=est.left_found,
                right_found=est.right_found,
                conf=conf,
                single_rail=est.single_rail,
                frame_id=est.frame_id,
                points=est.points,
                heading_valid=est.heading_valid,
                span_frac=est.span_frac,
                paired_bands=est.paired_bands,
            )
        return est, mask


def estimate_to_dict(est: BlueLineEstimate) -> dict:
    return {
        "found": est.found,
        "cx_norm": est.cx_norm,
        "cy_norm": est.cy_norm,
        "heading_err": est.heading_err,
        "width_norm": est.width_norm,
        "left_found": est.left_found,
        "right_found": est.right_found,
        "conf": est.conf,
        "single_rail": est.single_rail,
        "frame_id": est.frame_id,
        "points": [(float(u), float(v)) for u, v in est.points],
        "heading_valid": est.heading_valid,
        "span_frac": est.span_frac,
        "paired_bands": est.paired_bands,
    }


def estimate_from_dict(d: dict) -> BlueLineEstimate:
    """Rebuild a BlueLineEstimate from estimate_to_dict output."""
    return BlueLineEstimate(
        found=bool(d.get("found")),
        cx_norm=float(d.get("cx_norm", 0.0)),
        cy_norm=float(d.get("cy_norm", 0.0)),
        heading_err=float(d.get("heading_err", 0.0)),
        width_norm=float(d.get("width_norm", 0.0)),
        left_found=bool(d.get("left_found")),
        right_found=bool(d.get("right_found")),
        conf=float(d.get("conf", 0.0)),
        single_rail=bool(d.get("single_rail")),
        frame_id=int(d.get("frame_id", 0) or 0),
        points=tuple((float(u), float(v)) for u, v in d.get("points", ())),
        heading_valid=bool(d.get("heading_valid")),
        span_frac=float(d.get("span_frac", 0.0)),
        paired_bands=int(d.get("paired_bands", 0) or 0),
    )


def annotate_blue_lines(
    bgr: np.ndarray, est: BlueLineEstimate, mask: np.ndarray | None
) -> np.ndarray:
    """Overlay cyan mask tint + corridor centreline for the live vision window."""
    out = bgr.copy()
    if mask is not None:
        if mask.shape[:2] != out.shape[:2]:
            mask = cv2.resize(
                mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        tint = np.zeros_like(out)
        tint[:, :] = (255, 200, 0)  # cyan-ish BGR highlight
        m = mask > 0
        out[m] = (0.45 * out[m] + 0.55 * tint[m]).astype(np.uint8)

    h, w = out.shape[:2]
    hud = "no blue line"
    if est.found:
        for u, v in est.points:
            cv2.circle(out, (int(u), int(v)), 3, (0, 165, 255), -1)
        cx = int((est.cx_norm * 0.5 + 0.5) * w)
        cy = int((est.cy_norm * 0.5 + 0.5) * h)
        cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)
        cv2.line(out, (w // 2, h), (cx, cy), (0, 255, 255), 2)
        hdg_txt = (
            f"{math.degrees(est.heading_err):+.0f}d" if est.heading_valid else "--"
        )
        hud = (
            f"BLUE cx={est.cx_norm:+.2f} cy={est.cy_norm:+.2f} "
            f"hdg={hdg_txt} conf={est.conf:.2f} span={est.span_frac:.2f} "
            f"{est.paired_bands}/{len(est.points)}b "
            f"{'1RAIL' if est.single_rail else 'L+R'}"
        )
    cv2.putText(
        out,
        hud,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
