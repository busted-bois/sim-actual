"""Classical HSV dual-cyan corridor detection (no YOLO).

Two parallel blue ribbons define left/right track edges.

Two estimators run on the same mask:

* centroid (legacy) — split the frame at w//2, take each side's blob centroid,
  corridor mid = midpoint of the two centroids.
* inner-edge (default) — per-row scanlines track each ribbon's INNER edge from
  the bottom row upward; corridor mid = midpoint of the inner edges, fitted by
  least squares over the scanned rows.

Inner edges are the better geometric cue: the centroid of a ribbon moves with
its apparent THICKNESS, so unequal bloom (one ribbon nearer/brighter) shifts
the corridor mid even when the drone is centered. An inner edge is a
cyan->road HUE transition, unlike the outer edge's exposure-sensitive
cyan->black glow fade. Scanline tracking also drops the fixed w//2 split,
which mis-assigns pixels whenever the drone banks or the corridor curves.

BL_INNER_EDGE=0 reverts to the centroid estimator. Either way both values are
reported (cx_norm vs cx_centroid) so a live run is a direct A/B.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# OpenCV HSV: H 0-179. Bright cyan/electric-blue glow from the AI-GP ribbon.
_CYAN_LOWER = np.array([85, 70, 50], dtype=np.uint8)
_CYAN_UPPER = np.array([140, 255, 255], dtype=np.uint8)

_MORPH_KSIZE = 5
_NEAR_FRAC = 0.40  # lower fraction of frame = near field
_FAR_FRAC = 0.55  # upper fraction used for heading (from top)
_MIN_SIDE_AREA = 80.0
_MIN_TOTAL_AREA = 120.0

# --- inner-edge scanline parameters (all fractions of frame size) ------------
_SCAN_ROWS = 28  # sampled rows, bottom -> up
_SCAN_TOP_FRAC = 0.35  # stop here; above the vanishing point there is no ribbon
_MIN_RUN_FRAC = 0.005  # ignore cyan runs thinner than this (speckle, reflections)
# ...and wider than this. A ribbon is THIN in every row (measured 9-13 px at
# 640 wide, even at the bottom of frame); the lit track surface between the
# ribbons also passes the cyan mask on ~24% of live frames and shows up as one
# 276-395 px blob that swallows the corridor. Width is what separates them.
_MAX_RUN_FRAC = 0.15
_MAX_JUMP_FRAC = 0.10  # per-row inner-edge continuity limit
_MAX_MISS_ROWS = 4  # consecutive unusable rows before the scan stops
# A corridor gap narrower than this is not the corridor. On frames where the
# lit track surface itself passes the cyan mask, the ribbons and the floor
# merge into one blob and the only remaining "gaps" are slivers at the frame
# edge — seeding on one of those put the corridor centre on a ribbon and
# produced the +-0.5 cx excursions (24% of live frames). Refusing to seed
# there falls back to the centroid estimate, which survives a filled corridor.
_MIN_SEED_GAP_FRAC = 0.10
_MIN_FIT_ROWS = 4  # samples needed for a center fit
_MIN_FIT_SPAN_FRAC = 0.12  # vertical span needed before the slope means anything
# Row where cx is evaluated = middle of the near band, so BOTH estimators report
# cx at the same image row and the A/B stays comparable if _NEAR_FRAC is retuned.
_CX_REF_FRAC = 1.0 - _NEAR_FRAC / 2.0


def _norm(v: float, half: float) -> float:
    """Pixel coord -> normalized image error, -1 .. +1 (clamped)."""
    return float(np.clip((v - half) / half, -1.5, 1.5))


@dataclass(frozen=True)
class BlueLineEstimate:
    found: bool
    cx_norm: float = 0.0  # corridor mid, -1 left .. +1 right
    cy_norm: float = 0.0  # avg near height, -1 top .. +1 bottom
    heading_err: float = 0.0  # rad, + = corridor vanishes right of center
    width_norm: float = 0.0  # left-right separation / image width
    left_found: bool = False
    right_found: bool = False
    frame_id: int = 0
    # TODO(inner-edge-ab): cx_centroid/heading_centroid/edge_rows are scaffolding
    # for the inner-edge-vs-centroid comparison — delete them (and the matching
    # bl_log columns + probe reads) once the A/B concludes. `source` stays: it is
    # the only signal that the scan silently fell back to the centroid path.
    source: str = "centroid"  # "edge" | "centroid"
    cx_centroid: float = 0.0
    heading_centroid: float = 0.0
    edge_rows: int = 0
    # Per-row scan samples (y, x_left_inner|None, x_right_inner|None, x_center),
    # for the live overlay only — not published in estimate_to_dict.
    samples: tuple = field(default=(), repr=False)


def _inner_edge_enabled() -> bool:
    return os.environ.get("BL_INNER_EDGE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def cyan_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _CYAN_LOWER, _CYAN_UPPER)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_MORPH_KSIZE, _MORPH_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _centroid(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, area) or None if empty."""
    area = float(cv2.countNonZero(mask))
    if area < _MIN_SIDE_AREA:
        return None
    m = cv2.moments(mask)
    m00 = m["m00"]
    if m00 < 1e-6:
        return None
    return m["m10"] / m00, m["m01"] / m00, area


def _side_centroids(
    mask: np.ndarray, y0: int, y1: int
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Left/right centroids in horizontal band [y0, y1)."""
    h, w = mask.shape[:2]
    y0 = max(0, min(h, y0))
    y1 = max(y0, min(h, y1))
    band = np.zeros_like(mask)
    band[y0:y1, :] = mask[y0:y1, :]
    mid = w // 2
    left = _centroid(band[:, :mid])
    right = _centroid(band[:, mid:])
    if left is not None:
        left = (left[0], left[1], left[2])
    if right is not None:
        right = (right[0] + mid, right[1], right[2])
    return left, right


# --- inner-edge scanline estimator -------------------------------------------


def _row_runs(row: np.ndarray, min_len: int, max_len: int) -> list[tuple[int, int]]:
    """Ribbon-plausible runs [start, end] inclusive, ascending.

    Keeps contiguous nonzero runs whose width is in [min_len, max_len] — see
    _MIN_RUN_FRAC / _MAX_RUN_FRAC. Zero-padding both ends makes every rise pair
    with a fall, so runs touching the frame edge need no special casing.
    """
    d = np.diff(np.concatenate(([0], (row > 0).astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # exclusive
    return [
        (int(s), int(e) - 1)
        for s, e in zip(starts, ends)
        if min_len <= e - s <= max_len
    ]


def _seed_gap(
    runs: list[tuple[int, int]], center_x: float, min_gap: float
) -> tuple[float, float] | None:
    """Corridor = the gap BETWEEN two ribbons; seed on the one we fly inside.

    Prefers the inter-run gap containing the image center (the drone normally
    sits between the ribbons), else the widest gap. Unlike a w//2 split this
    still works when both ribbons are on the same side of the frame. Gaps
    narrower than min_gap are rejected outright — see _MIN_SEED_GAP_FRAC.
    """
    gaps = [
        (float(a[1]), float(b[0]))
        for a, b in zip(runs, runs[1:])
        if b[0] - a[1] >= min_gap
    ]
    if not gaps:
        return None
    return max(gaps, key=lambda g: (g[0] <= center_x <= g[1], g[1] - g[0]))


def _nearest(cands: list[float], predicted: float, max_jump: float) -> float | None:
    """Candidate closest to `predicted`, or None if none is within max_jump."""
    return min(
        (c for c in cands if abs(c - predicted) <= max_jump),
        key=lambda c: abs(c - predicted),
        default=None,
    )


def _scan_inner_edges(
    mask: np.ndarray,
) -> list[tuple[float, float | None, float | None, float]]:
    """Track both inner edges bottom-up. Returns (y, xl, xr, xc) per row.

    xl/xr are None when that side was not measured on the row (xc is then
    carried across from the tracked half-width). Continuity against the
    previous row is what rejects detached blobs — floor reflections, and the
    gate's own cyan panel near the vanishing point.
    """
    h, w = mask.shape[:2]
    min_len = max(3, int(w * _MIN_RUN_FRAC))
    max_len = int(w * _MAX_RUN_FRAC)
    max_jump = max(4.0, w * _MAX_JUMP_FRAC)
    y_top = int(h * _SCAN_TOP_FRAC)
    y_bot = h - 1
    step = max(1, (y_bot - y_top) // max(_SCAN_ROWS - 1, 1))

    samples: list[tuple[float, float | None, float | None, float]] = []
    center_p: float | None = None
    half_p = 0.0
    miss = 0

    for y in range(y_bot, y_top - 1, -step):
        runs = _row_runs(mask[y], min_len, max_len)
        xl = xr = None
        if center_p is None:
            gap = _seed_gap(runs, w * 0.5, w * _MIN_SEED_GAP_FRAC)
            if gap is not None:
                xl, xr = gap
        elif runs:
            # Predict this row's inner edges from the row below, then take the
            # nearest candidate on each side.
            pred_l, pred_r = center_p - half_p, center_p + half_p
            xl = _nearest([float(r[1]) for r in runs], pred_l, max_jump)
            xr = _nearest([float(r[0]) for r in runs], pred_r, max_jump)
            # A single run gives one edge, never both: drop the weaker match.
            if xl is not None and xr is not None and xr <= xl:
                if abs(xl - pred_l) <= abs(xr - pred_r):
                    xr = None
                else:
                    xl = None

        if xl is not None and xr is not None:
            center = 0.5 * (xl + xr)
            half_p = 0.5 * (xr - xl)
        elif xl is not None and half_p > 0.0:
            center = xl + half_p
        elif xr is not None and half_p > 0.0:
            center = xr - half_p
        else:
            miss += 1
            # Rows before the seed do not count toward the abort: the ribbons
            # can start well above the bottom of the frame (nose-down view, or
            # a bloomed bottom band), and giving up there means never reaching
            # them at all. After seeding, a miss streak is a genuine end-of-line.
            if samples and miss >= _MAX_MISS_ROWS:
                break
            continue

        miss = 0
        center_p = center
        samples.append((float(y), xl, xr, center))

    return samples


def _estimate_inner_edge(mask: np.ndarray) -> dict | None:
    """cx/heading/width from the inner-edge scan, or None if it did not track."""
    h, w = mask.shape[:2]
    samples = _scan_inner_edges(mask)
    if len(samples) < _MIN_FIT_ROWS:
        return None

    ys = np.array([s[0] for s in samples], dtype=np.float64)
    xs = np.array([s[3] for s in samples], dtype=np.float64)
    slope, intercept = np.polyfit(ys, xs, 1)

    y_ref = h * _CX_REF_FRAC
    mid_x = float(slope * y_ref + intercept)

    # Heading from the fitted slope. The legacy form is
    # arctan2(far_mid_x - near_mid_x, 0.3h); over a 0.3h baseline a corridor of
    # slope b (dx per dy, y down) gives dx = -b*0.3h, so this is the SAME
    # quantity — just least-squares over ~28 rows instead of two centroids.
    # Pilot gains (K_*_HDG) therefore need no retuning.
    heading_err = 0.0
    if float(ys.max() - ys.min()) >= h * _MIN_FIT_SPAN_FRAC:
        heading_err = float(np.arctan(-slope))

    # Single pass: per-side measurement counts, plus the two-sided sample
    # closest to y_ref (corridor width is read off that row).
    n_left = n_right = 0
    nearest = None
    for s in samples:
        n_left += s[1] is not None
        n_right += s[2] is not None
        if s[1] is not None and s[2] is not None:
            if nearest is None or abs(s[0] - y_ref) < abs(nearest[0] - y_ref):
                nearest = s
    width_norm = 0.0 if nearest is None else float(abs(nearest[2] - nearest[1]) / w)

    return {
        "cx_norm": _norm(mid_x, w / 2.0),
        "heading_err": heading_err,
        "width_norm": width_norm,
        "left_found": n_left >= 2,
        "right_found": n_right >= 2,
        "rows": len(samples),
        "samples": tuple(samples),
    }


def detect_blue_lines(
    bgr: np.ndarray, frame_id: int = 0, *, return_mask: bool = False
) -> tuple[BlueLineEstimate, np.ndarray | None]:
    """Detect dual cyan lines; return estimate and optional mask."""
    h, w = bgr.shape[:2]
    mask = cyan_mask(bgr)
    total = float(cv2.countNonZero(mask))
    if total < _MIN_TOTAL_AREA:
        est = BlueLineEstimate(found=False, frame_id=frame_id)
        return est, (mask if return_mask else None)

    near_y0 = int(h * (1.0 - _NEAR_FRAC))
    left_n, right_n = _side_centroids(mask, near_y0, h)

    far_y1 = int(h * _FAR_FRAC)
    left_f, right_f = _side_centroids(mask, 0, far_y1)

    left_found = left_n is not None
    right_found = right_n is not None
    if not left_found and not right_found:
        # Fall back to full-frame sides if near ROI empty (looking at ribbon).
        left_n, right_n = _side_centroids(mask, 0, h)
        left_found = left_n is not None
        right_found = right_n is not None

    if not left_found and not right_found:
        est = BlueLineEstimate(found=False, frame_id=frame_id)
        return est, (mask if return_mask else None)

    half_w = w / 2.0
    half_h = h / 2.0

    if left_found and right_found:
        mid_x = 0.5 * (left_n[0] + right_n[0])
        mid_y = 0.5 * (left_n[1] + right_n[1])
        width_norm = abs(right_n[0] - left_n[0]) / float(w)
    elif left_found:
        # Only left: bias corridor right of the line by ~0.25 of half-width.
        mid_x = left_n[0] + 0.25 * w
        mid_y = left_n[1]
        width_norm = 0.0
    else:
        mid_x = right_n[0] - 0.25 * w
        mid_y = right_n[1]
        width_norm = 0.0

    # Altitude cue: mid/far ribbon height (elevated path), not near floor paint.
    mid_band_y0 = int(h * 0.25)
    mid_band_y1 = int(h * 0.65)
    left_m, right_m = _side_centroids(mask, mid_band_y0, mid_band_y1)
    if left_m is not None and right_m is not None:
        alt_y = 0.5 * (left_m[1] + right_m[1])
    elif left_f is not None and right_f is not None:
        alt_y = 0.5 * (left_f[1] + right_f[1])
    else:
        alt_y = mid_y

    cx_norm = _norm(mid_x, half_w)
    cy_norm = _norm(alt_y, half_h)

    # Heading: vanishing of far mid relative to near mid (image x).
    heading_err = 0.0
    if left_f is not None and right_f is not None and left_found and right_found:
        far_mid_x = 0.5 * (left_f[0] + right_f[0])
        near_mid_x = 0.5 * (left_n[0] + right_n[0])
        # Positive = corridor bends/vanishes to the right → yaw right.
        heading_err = float(np.arctan2(far_mid_x - near_mid_x, max(h * 0.3, 1.0)))
    elif left_found and right_found:
        # Fit from near line tilt: average of left/right column slopes via far.
        if left_f is not None:
            heading_err = float(
                np.arctan2(left_f[0] - left_n[0], max(left_n[1] - left_f[1], 1.0))
            )
        elif right_f is not None:
            heading_err = float(
                np.arctan2(right_f[0] - right_n[0], max(right_n[1] - right_f[1], 1.0))
            )

    # cy (altitude) always stays on the centroid path — only the lateral/heading
    # cues move to inner edges, so a live A/B changes one thing at a time.
    cx_centroid, heading_centroid = cx_norm, heading_err
    source = "centroid"
    edge_rows = 0
    samples: tuple = ()
    if _inner_edge_enabled():
        edge = _estimate_inner_edge(mask)
        if edge is not None:  # else: keep the centroid result as the backstop
            cx_norm = edge["cx_norm"]
            heading_err = edge["heading_err"]
            width_norm = edge["width_norm"]
            left_found = edge["left_found"]
            right_found = edge["right_found"]
            edge_rows = edge["rows"]
            samples = edge["samples"]
            source = "edge"

    est = BlueLineEstimate(
        found=True,
        cx_norm=cx_norm,
        cy_norm=cy_norm,
        heading_err=heading_err,
        width_norm=float(width_norm),
        left_found=left_found,
        right_found=right_found,
        frame_id=frame_id,
        source=source,
        cx_centroid=cx_centroid,
        heading_centroid=heading_centroid,
        edge_rows=edge_rows,
        samples=samples,
    )
    return est, (mask if return_mask else None)


def estimate_to_dict(est: BlueLineEstimate) -> dict:
    return {
        "found": est.found,
        "cx_norm": est.cx_norm,
        "cy_norm": est.cy_norm,
        "heading_err": est.heading_err,
        "width_norm": est.width_norm,
        "left_found": est.left_found,
        "right_found": est.right_found,
        "frame_id": est.frame_id,
        "source": est.source,
        "cx_centroid": est.cx_centroid,
        "heading_centroid": est.heading_centroid,
        "edge_rows": est.edge_rows,
    }


def annotate_blue_lines(
    bgr: np.ndarray, est: BlueLineEstimate, mask: np.ndarray | None
) -> np.ndarray:
    """Overlay cyan mask tint + corridor mid for the live vision window."""
    out = bgr.copy()
    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :] = (255, 200, 0)  # cyan-ish BGR highlight
        m = mask > 0
        out[m] = (0.45 * out[m] + 0.55 * tint[m]).astype(np.uint8)

    h, w = out.shape[:2]
    hud = "no blue line"
    if est.found:
        # Tracked inner edges + fitted corridor mid, so the scan can be checked
        # by eye against the ribbons it is supposed to be riding.
        for y, xl, xr, xc in est.samples:
            if xl is not None:
                cv2.circle(out, (int(xl), int(y)), 2, (0, 0, 255), -1)
            if xr is not None:
                cv2.circle(out, (int(xr), int(y)), 2, (0, 255, 0), -1)
            cv2.circle(out, (int(xc), int(y)), 2, (255, 255, 255), -1)

        cx = int((est.cx_norm * 0.5 + 0.5) * w)
        cy = int((est.cy_norm * 0.5 + 0.5) * h)
        cv2.circle(out, (cx, cy), 6, (0, 255, 255), -1)
        cv2.line(out, (w // 2, h), (cx, cy), (0, 255, 255), 2)
        # Legacy centroid mid (magenta) — the live A/B against the yellow edge mid.
        cxc = int((est.cx_centroid * 0.5 + 0.5) * w)
        cv2.drawMarker(out, (cxc, cy), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        hud = (
            f"{est.source.upper()} cx={est.cx_norm:+.2f} (cent {est.cx_centroid:+.2f}) "
            f"cy={est.cy_norm:+.2f} hdg={est.heading_err:+.2f} "
            f"(cent {est.heading_centroid:+.2f}) rows={est.edge_rows} "
            f"L={int(est.left_found)} R={int(est.right_found)}"
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
