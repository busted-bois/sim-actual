"""Blue-track-line detector: find the cyan racing ribbon in a camera frame.

The VQ2 course is marked by a glowing cyan floor ribbon (with a white-hot
core and occasional gold start-ramp beam flanked by the same cyan rails).
detect_track() turns one BGR frame into a steering cue for fly2's 90 Hz
loop: lateral offset of the near end, heading angle across depth bands,
and a confidence. Must run < 3 ms, so it downscales internally.

Measured facts from a 15-frame survey of real annotated flight video
(10 with-track, 5 without; 640x360):
  * ribbon body:  H 80-106, S >= 90, V >= 110  -- this range alone gives
    0 false-positive px on all 5 no-track frames and 461-21094 px on all
    10 with-track frames.
  * white-hot core: S <= 95, V >= 220, any hue -- MUST be adjacency-gated
    to within ~5 px of the cyan mask; standalone it fires on ceiling
    lights and gate checker patterns (2300+ px on trackless frames).
  * H is capped at 106 because YOLO overlay boxes/labels (pure blue
    H~120) smear down to H 105-116 under video compression.
  * YOLO keypoint dots: small (< 8 px full-res) saturated round clusters
    -> killed by a connected-component filter (small AND non-elongated).
    A morphological open would also kill them, but it erases the 1-2 px
    thin line the ribbon becomes when far ahead (survey frame t=1.5 s),
    so the CC filter is used instead.
  * green timer text lives in rows 0-30, cols 0-120: masked explicitly.
  * magenta wash near red gate lights punches small GAPS in the ribbon
    mask (never false positives) -> closed with morphology.
  * gold beam segments are hue-identical to the yellow floor lane
    stripes, so no gold range is used; the flanking cyan rails always
    carry the detection (their centroid midpoint sits on the beam).

Geometry: rows below the timer strip (top 8 %) are split into 12
horizontal bands; each band contributes a mask centroid if it has enough
pixels and is not a glow flood. Flood test is span > 60 % of the width
AND dense fill ( > 20 % of the band area): a genuinely wide-but-THIN
stripe is the ribbon seen side-on when banked or crossing ahead, and is
kept -- measured on the survey frames, span-only rejection lost 4 of the
10 with-track frames. >= 3 valid bands are required (2 accepted when the
total mask is large -- a horizontal ribbon at a sharp turn fits in 2
bands, survey frame t=15.3 s). Bands reach almost to the top because the
ribbon shrinks to a thin line high in the frame when still far ahead
(survey frame t=1.5 s).

Bank/roll caveat for the integrator: when the drone is rolled the ribbon
crosses the image diagonally; the angle output absorbs moderate bank, but
a near-horizontal ribbon occupies few bands, so strength drops (0.17 in
the 2-band fallback) and angle saturates near +-pi/2 -- treat low
strength as "trust offset, not angle". Strength is n_valid_bands / 12.

Run me:  uv run python -m simulator.track_line --selftest
"""

import argparse
import os
import time

import cv2
import numpy as np

# --- HSV ranges (OpenCV H in 0..180), from the frame survey ------------------
CYAN_LO = np.array([80, 90, 110], np.uint8)
CYAN_HI = np.array([106, 255, 255], np.uint8)
CORE_LO = np.array([0, 0, 220], np.uint8)  # white-hot core, adjacency-gated
CORE_HI = np.array([180, 95, 255], np.uint8)

# --- tuning -------------------------------------------------------------------
TIMER_ROWS, TIMER_COLS = 34, 130  # full-res region of the green timer text
BAND_TOP_FRAC = 0.08  # skip the timer strip; ribbon can sit high when far
N_BANDS = 12  # finer bands so a near-horizontal ribbon still spans >= 3
MIN_BAND_PX_FULLRES = 24  # per-band pixel floor, scaled by downscale^2
MAX_BAND_SPAN_FRAC = 0.60  # span above this AND dense fill = glow flood
FLOOD_FILL_FRAC = 0.20  # dense-fill threshold for the flood reject
MIN_VALID_BANDS = 3
TWO_BAND_PX_FULLRES = 600  # 2-band fallback needs this much total mask
CORE_GATE_PX_FULLRES = 5  # dilate radius linking white core to cyan
MIN_CC_AREA_FULLRES = 60  # smaller + round = YOLO keypoint dot -> drop
CC_ELONGATION_KEEP = 3.0  # bbox aspect ratio that marks a real track sliver

_KERN3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERN5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Survey frames used by _selftest when present (written by the frame survey).
_SURVEY_DIR = (
    "C:/Users/kunal/AppData/Local/Temp/claude/"
    "C--Users-kunal-Desktop-sim-actual/"
    "777ec0ea-1fa4-4465-a6ee-da54a4af203e/scratchpad/survey/frames"
)


def _track_mask(small_bgr, downscale):
    """Binary ribbon mask at downscaled resolution."""
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    cyan = cv2.inRange(hsv, CYAN_LO, CYAN_HI)
    # white-hot core counts only next to cyan (ceiling lights otherwise)
    r = max(1, round(CORE_GATE_PX_FULLRES / downscale))
    near_cyan = cv2.dilate(cyan, _KERN3, iterations=r)
    core = cv2.inRange(hsv, CORE_LO, CORE_HI)
    mask = cv2.bitwise_or(cyan, cv2.bitwise_and(core, near_cyan))
    # timer text region
    mask[: TIMER_ROWS // downscale, : TIMER_COLS // downscale] = 0
    # close bridges magenta-wash gaps; NO open (it erases the far thin line)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERN5)
    # connected-component filter: small round blobs are YOLO keypoint dots
    min_area = max(4, MIN_CC_AREA_FULLRES // (downscale * downscale))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        area = stats[:, cv2.CC_STAT_AREA]
        bw = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        bh = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        elong = np.maximum(bw, bh) >= CC_ELONGATION_KEEP * np.minimum(bw, bh)
        keep = (area >= min_area) | elong
        keep[0] = False  # background
        mask = np.where(keep[lbl], mask, 0)
    return mask


def detect_track(frame_bgr, downscale=2):
    """Detect the cyan track ribbon in one BGR frame. -> dict | None.

    offset   horizontal position of the NEAREST detected band centre,
             normalized to [-1, 1]; + = track right of image centre.
    angle    rad; direction the track heads across bands, + = far end is
             right of the near end (track curves/heads right).
    strength confidence in [0, 1]: fraction of bands with a solid centroid.
    points   [(u, v), ...] band centroids in FULL-RES coords, near -> far.
    """
    if downscale > 1:
        small = frame_bgr[::downscale, ::downscale]
    else:
        small = frame_bgr
    mask = _track_mask(small, downscale)
    sh, sw = mask.shape

    min_px = max(6, MIN_BAND_PX_FULLRES // (downscale * downscale))
    y0 = int(sh * BAND_TOP_FRAC)
    edges = np.linspace(y0, sh, N_BANDS + 1).astype(int)

    pts = []  # (u, v) downscaled, near (bottom) first
    weights = []  # band pixel counts, aligned with pts
    total_px = 0
    for i in range(N_BANDS - 1, -1, -1):
        band = mask[edges[i] : edges[i + 1]]
        ys, xs = np.nonzero(band)
        if xs.size < min_px:
            continue
        span = int(xs.max() - xs.min())
        fill = xs.size / band.size
        if span > MAX_BAND_SPAN_FRAC * sw and fill > FLOOD_FILL_FRAC:
            continue  # glow flood; wide-but-thin = side-on ribbon, kept
        total_px += xs.size
        pts.append((float(xs.mean()), float(edges[i] + ys.mean())))
        weights.append(float(xs.size))

    if len(pts) < MIN_VALID_BANDS:
        # a ribbon crossing HORIZONTALLY (sharp turn ahead / banked) can
        # live in just 2 bands; accept only with strong pixel evidence
        strong = total_px >= TWO_BAND_PX_FULLRES // (downscale * downscale)
        if len(pts) < 2 or not strong:
            return None

    us = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    # Offset = pixel-weighted mean of the nearest 3 bands, NOT the single
    # nearest band: in the gold start-beam region that band can be a ~29 px
    # fragment of ONE flanking rail, flapping the offset +-0.5 between
    # consecutive near-identical frames (geometry review 2026-07-06).
    k = min(3, len(us))
    w = np.array(weights[:k])
    offset = (float(np.average(us[:k], weights=w)) - sw / 2.0) / (sw / 2.0)
    # u = a*v + b through the centroids; near = large v, so heading toward
    # the far end (v decreasing) goes right when a < 0.
    a = np.polyfit(vs, us, 1)[0]
    angle = float(np.arctan2(-a, 1.0))
    return {
        "offset": float(np.clip(offset, -1.0, 1.0)),
        "angle": angle,
        "strength": len(pts) / N_BANDS,
        "points": [(u * downscale, v * downscale) for u, v in pts],
    }


# --- selftest -----------------------------------------------------------------


def _synth(u_near, u_far, color=(255, 255, 0), thickness=46):
    """640x360 black frame with a ribbon from (u_near, bottom) to (u_far, top
    of band region). Cyan by default (BGR (255,255,0) -> H=90 S=255 V=255)."""
    img = np.zeros((360, 640, 3), np.uint8)
    p = np.array([(u_near, 359), ((u_near + u_far) // 2, 250), (u_far, 135)], np.int32)
    cv2.polylines(img, [p], False, color, thickness)
    return img


def _selftest():
    # 1) synthetic geometry: presence, offset sign, angle sign, rejections
    d = detect_track(_synth(320, 320))
    assert d is not None and abs(d["offset"]) < 0.1 and abs(d["angle"]) < 0.1, d
    assert d["strength"] >= 0.5, d
    print(f"[selftest] straight  off={d['offset']:+.2f} ang={d['angle']:+.2f}")

    d = detect_track(_synth(480, 500))
    assert d is not None and d["offset"] > 0.3, d
    print(f"[selftest] right-off off={d['offset']:+.2f}")

    d = detect_track(_synth(320, 520))
    assert d is not None and d["angle"] > 0.15, d
    print(f"[selftest] curve-R   ang={d['angle']:+.2f} (expect > 0)")

    d = detect_track(_synth(320, 120))
    assert d is not None and d["angle"] < -0.15, d
    print(f"[selftest] curve-L   ang={d['angle']:+.2f} (expect < 0)")

    blank = np.zeros((360, 640, 3), np.uint8)
    for k in range(30):  # fake ceiling lights: white must not fire alone
        cv2.circle(blank, (21 * k + 10, 40 + 9 * k), 3, (255, 255, 255), -1)
    for k in range(8):  # fake cyan-ish YOLO keypoint dots (small + round)
        cv2.circle(blank, (300 + 9 * k, 200 + 11 * k), 3, (255, 255, 0), -1)
    assert detect_track(blank) is None, "lights/dots w/o track must be None"

    red = np.zeros((360, 640, 3), np.uint8)
    red[:] = (0, 0, 255)  # full-frame red-glow crash frame
    assert detect_track(red) is None, "red glow frame must be None"
    print("[selftest] absent / red-glow -> None  OK")

    # 2) real survey frames, if present on disk (report, don't assert)
    for sub, want in (("with_track", True), ("no_track", False)):
        p = os.path.join(_SURVEY_DIR, sub)
        if not os.path.isdir(p):
            continue
        names = sorted(os.listdir(p))
        hits, dt = 0, 0.0
        for n in names:
            img = cv2.imread(os.path.join(p, n))
            t0 = time.perf_counter()
            r = detect_track(img)
            dt += time.perf_counter() - t0
            hits += r is not None
        rate = hits / max(1, len(names))
        kind = "detect" if want else "false-pos"
        print(
            f"[selftest] real {sub}: {kind} {hits}/{len(names)} "
            f"({rate:.0%}), {1e3 * dt / max(1, len(names)):.2f} ms/frame"
        )
    print("[selftest] OK -- track ribbon detector")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
