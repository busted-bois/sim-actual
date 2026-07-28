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

from simulator.track_line import track_mask

_DOWNSCALE = 2  # detection resolution; outputs are full-res / normalized

# Band layout mirrors detect_track: skip the timer strip, keep bands fine
# enough that a near-horizontal ribbon at a sharp turn still spans several.
_N_BANDS = 12
_BAND_TOP_FRAC = 0.08

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

    edges = np.linspace(int(sh * _BAND_TOP_FRAC), sh, _N_BANDS + 1).astype(int)
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
        fit_l, fit_r, fit_hw = _fit(pv, pl), _fit(pv, pr), _fit(pv, phw)
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
    heading_err = 0.0
    if len(vs) >= 2:
        a = float(np.polyfit(np.asarray(vs), np.asarray(us), 1)[0])
        heading_err = float(np.arctan2(-a, 1.0))

    width_norm = float(np.mean(phw) * 2.0 / sw) if phw else 0.0

    paired_frac = len(pv) / len(centres)
    conf = float(
        np.clip(len(centres) / _CONF_FULL_BANDS, 0.0, 1.0) * (0.5 + 0.5 * paired_frac)
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

    def reset(self) -> None:
        self._half_width_px = None
        self._side = None
        self._hdg = None

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

        max_step = math.radians(_MAX_HDG_STEP_DEG)
        if self._hdg is not None and abs(est.heading_err - self._hdg) > max_step:
            clamped = self._hdg + math.copysign(max_step, est.heading_err - self._hdg)
            est = BlueLineEstimate(
                found=est.found,
                cx_norm=est.cx_norm,
                cy_norm=est.cy_norm,
                heading_err=clamped,
                width_norm=est.width_norm,
                left_found=est.left_found,
                right_found=est.right_found,
                conf=est.conf * 0.5,  # the frame is suspect, not unusable
                single_rail=est.single_rail,
                frame_id=est.frame_id,
                points=est.points,
            )
        self._hdg = est.heading_err
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
        hud = (
            f"BLUE cx={est.cx_norm:+.2f} cy={est.cy_norm:+.2f} "
            f"hdg={math.degrees(est.heading_err):+.0f}d conf={est.conf:.2f} "
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
