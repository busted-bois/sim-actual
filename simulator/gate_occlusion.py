"""Non-gate occluder (pillar / wall) PEEK cue for GPPilot.

Gate 3 fails on ~89% of runs because a dark pillar sits between the drone and the
gate. Guidance only knows the gate CENTRE bearing, so it banks straight at the
pillar; when the gate then vanishes behind it the pilot goes blind and hands off
to a search that sweeps in place instead of stepping around the obstruction.

This fuses the obstacle blobs already computed every frame in vision_rx with the
gate's visible position: find the blob that occludes the gate, decide which side
of it the gate's evidence lies on, and emit a PEEK side + strength so guidance can
bias its aim that way -- fly AROUND the pillar to re-open the view.

Design constraints (from flight-tested prior art, commit c0d51f9 reverted):
  * Inert on clean approaches -> None (an occluder must actually overlap / sit in
    front of the gate), so gates 1-2 are untouched.
  * NEVER hold a stale bias: strength DECAYS on dropout (mirrors ELEV_BLIND_DECAY)
    and disengages -- the c0d51f9 failure was a held aim-bias going stale at the
    crossing.
  * N-frame confirm before declaring a side (roll shouldn't chase per-frame noise).

Contract:
    detect_occlusion(gate, obstacles, img_shape, bgr=None, frame_id=None) -> dict|None
        gate       {cx, cy, x0, x1, y0, y1} px, or None
        obstacles  [{x0, x1, y0, y1, area_px, nx, ny, r_frac}, ...] (vision_rx)
        returns    {"side": +/-1, "strength": 0..1, "blob_x", "gate_lr",
                    "reason", "frame_id"} | None
    OcclusionTracker().update(...) adds the confirm + decay the pilot consumes.

    uv run python -m simulator.gate_occlusion --selftest
    uv run python -m simulator.gate_occlusion --frames a.jpg b.jpg   # gate=None (fallback)
"""

from __future__ import annotations

import argparse

import numpy as np

from simulator.pillar_detect import BAND_BOT_FRAC, BAND_TOP_FRAC, detect_pillar

# Blob must fill at least this fraction of the frame to count as an occluder.
OCC_MIN_AREA_FRAC = 0.010
# Gate-mass imbalance (right-left)/(right+left) below this -> use the tie-break.
SIDE_DEADBAND = 0.15
# Gate-3's pillar is consistently on the LEFT, gate visible to the RIGHT
# (user-confirmed). When gate mass is balanced about the blob, default to peek
# RIGHT. Tie-break ONLY -- never overrides real evidence, and only once an
# occluder is already detected, so it can't bias a clean approach elsewhere.
PEEK_TIEBREAK_SIDE = 1
# Confirm + decay (the c0d51f9 "never hold" lesson).
OCC_CONFIRM_FRAMES = 3
OCC_DECAY = 0.85          # strength * this per dropout frame
OCC_MIN_STRENGTH = 0.05   # below this, drop the cue entirely


def detect_occlusion(gate, obstacles, img_shape, bgr=None, frame_id=None):
    """Raw per-frame occlusion cue (no temporal state). See module docstring."""
    h, w = int(img_shape[0]), int(img_shape[1])

    # No gate in view -> fall back to the classical brightness pillar cue (this
    # is what finally wires the orphan pillar_detect in).
    if gate is None:
        return _brightness_fallback(bgr, frame_id)

    if not obstacles:
        return None

    gx0, gx1, gcx = float(gate["x0"]), float(gate["x1"]), float(gate["cx"])
    gate_w = max(1.0, gx1 - gx0)
    band_top, band_bot = BAND_TOP_FRAC * h, BAND_BOT_FRAC * h
    area_floor = OCC_MIN_AREA_FRAC * w * h
    cx_img = w / 2.0

    best = None
    for ob in obstacles:
        area = float(ob.get("area_px", ob.get("r_frac", 0.0) * w * h))
        if area < area_floor:
            continue
        bx0, bx1 = float(ob["x0"]), float(ob["x1"])
        bcx = 0.5 * (bx0 + bx1)
        bcy = 0.5 * (float(ob["y0"]) + float(ob["y1"]))
        if not (band_top <= bcy <= band_bot):
            continue

        # Occluder test: overlaps the gate horizontally, OR sits between the image
        # centre and the gate centroid (a pillar the drone is aimed through).
        horiz_overlap = min(bx1, gx1) - max(bx0, gx0)
        between = min(cx_img, gcx) <= bcx <= max(cx_img, gcx)
        if horiz_overlap <= 0.0 and not between:
            continue

        # PEEK side = which side of the blob carries more gate mass.
        gate_right = max(0.0, gx1 - bcx)
        gate_left = max(0.0, bcx - gx0)
        imbal = (gate_right - gate_left) / (gate_right + gate_left + 1e-6)
        if abs(imbal) >= SIDE_DEADBAND:
            side = 1 if imbal > 0 else -1
            gate_lr = "right" if side > 0 else "left"
        else:
            side = PEEK_TIEBREAK_SIDE  # balanced -> gate-3 prior
            gate_lr = "tiebreak"

        occ_frac = float(np.clip(max(0.0, horiz_overlap) / gate_w, 0.0, 1.0))
        centrality = float(np.clip(1.0 - abs(bcx - cx_img) / cx_img, 0.0, 1.0))
        strength = float(np.clip(occ_frac * centrality, 0.0, 1.0))
        if strength <= 0.0:
            strength = 0.15 * centrality  # 'between' the aim line but not overlapping yet

        cand = {
            "side": side, "strength": strength, "blob_x": bcx,
            "gate_lr": gate_lr, "reason": "pillar", "frame_id": frame_id,
        }
        if best is None or cand["strength"] > best["strength"]:
            best = cand
    return best


def obstacles_from_frame(img, detection=None, min_area=800):
    """Dark-blob obstacle contours (pillars / walls) from one BGR frame: gate
    orange (>80 gray) + bright objects + the lower ground band masked out, and
    the gate's own detection circle punched out. SINGLE source of truth shared by
    vision_rx (live) and peek_replay (offline) so both see identical blobs.
    Returns [{nx, ny, r_frac, x0, x1, y0, y1, area_px}, ...]."""
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    mask[gray > 80] = 0          # exclude gate orange (~100+) and bright objects
    mask[int(h * 0.55):, :] = 0  # exclude the dark floor band
    if detection is not None:
        cv2.circle(
            mask,
            (int(detection.centroid_x_px), int(detection.centroid_y_px)),
            int(max(detection.width_px, detection.height_px)),
            0, -1,
        )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for oc in contours:
        oa = cv2.contourArea(oc)
        if oa < min_area:
            continue
        m = cv2.moments(oc)
        m00 = max(m["m00"], 1e-6)
        ocx, ocy = m["m10"] / m00, m["m01"] / m00
        bx, by, bw, bh = cv2.boundingRect(oc)
        out.append({
            "nx": (ocx - w / 2.0) / (w / 2.0), "ny": (ocy - h / 2.0) / (h / 2.0),
            "r_frac": oa / (w * h),
            "x0": float(bx), "x1": float(bx + bw),
            "y0": float(by), "y1": float(by + bh), "area_px": float(oa),
        })
    return out


def _brightness_fallback(bgr, frame_id):
    """No gate mask -> classical dark-blob brightness cue (pillar_detect)."""
    if bgr is None:
        return None
    d = detect_pillar(bgr)
    if d is None:
        return None
    steer = int(d["steer"])
    return {
        "side": steer if steer != 0 else PEEK_TIEBREAK_SIDE,
        "strength": float(d["threat"]),
        "blob_x": None,
        "gate_lr": "brightness",
        "reason": "pillar_detect",
        "frame_id": frame_id,
    }


class OcclusionTracker:
    """Wraps detect_occlusion with the N-frame confirm + decay-on-dropout the
    pilot consumes. Held by VisionRX (one per stream), read as data['occlusion']."""

    def __init__(self):
        self._side = 0
        self._count = 0
        self._strength = 0.0

    def reset(self):
        self._side = 0
        self._count = 0
        self._strength = 0.0

    def update(self, gate, obstacles, img_shape, bgr=None, frame_id=None):
        raw = detect_occlusion(gate, obstacles, img_shape, bgr, frame_id)
        if raw is None:
            # DECAY the last cue toward zero -- never hold it at the crossing.
            self._strength *= OCC_DECAY
            if self._strength < OCC_MIN_STRENGTH:
                self.reset()
                return None
            return {
                "side": self._side, "strength": self._strength, "blob_x": None,
                "gate_lr": "decay", "reason": "decay", "frame_id": frame_id,
            }
        if raw["side"] == self._side:
            self._count += 1
        else:
            self._side, self._count = raw["side"], 1
        self._strength = raw["strength"]
        if self._count < OCC_CONFIRM_FRAMES:
            return None  # not yet confirmed -> no bias
        out = dict(raw)
        out["side"], out["strength"] = self._side, self._strength
        return out


def _selftest() -> None:
    shape = (360, 640)
    # Gate centred, wide (x 220..420, cx 320).
    gate = {"cx": 320, "cy": 180, "x0": 220, "x1": 420, "y0": 90, "y1": 270}

    def blob(x0, x1, y0=170, y1=210, area=6000):
        return [{"x0": x0, "x1": x1, "y0": y0, "y1": y1, "area_px": area,
                 "nx": 0.0, "ny": 0.0, "r_frac": area / (640 * 360)}]

    # pillar LEFT of gate centre (blob cx 250) -> gate mass mostly RIGHT -> peek +1
    d = detect_occlusion(gate, blob(220, 280), shape)
    assert d is not None and d["side"] == 1, d
    print(f"[selftest] pillar left  -> peek {d['side']:+d} ({d['gate_lr']}) str={d['strength']:.2f}")

    # pillar RIGHT of gate centre (blob cx 390) -> gate mass mostly LEFT -> peek -1
    d = detect_occlusion(gate, blob(360, 420), shape)
    assert d is not None and d["side"] == -1, d
    print(f"[selftest] pillar right -> peek {d['side']:+d} ({d['gate_lr']}) str={d['strength']:.2f}")

    # centred pillar, balanced gate -> tie-break (peek right)
    d = detect_occlusion(gate, blob(300, 340), shape)
    assert d is not None and d["side"] == PEEK_TIEBREAK_SIDE and d["gate_lr"] == "tiebreak", d
    print(f"[selftest] centred pillar -> tie-break peek {d['side']:+d}")

    # clean scene: gate visible, no obstacles -> None (gates 1-2 untouched)
    assert detect_occlusion(gate, [], shape) is None
    print("[selftest] clean scene -> None OK")

    # tiny blob under the area floor -> ignored
    assert detect_occlusion(gate, blob(300, 340, area=500), shape) is None
    print("[selftest] sub-floor blob -> None OK")

    # blob outside the danger band (near top) -> ignored
    assert detect_occlusion(gate, blob(300, 340, y0=10, y1=40), shape) is None
    print("[selftest] out-of-band blob -> None OK")

    # tracker: confirm needs N frames, then decays on dropout (never holds)
    tr = OcclusionTracker()
    outs = [tr.update(gate, blob(220, 280), shape) for _ in range(OCC_CONFIRM_FRAMES)]
    assert outs[0] is None and outs[-1] is not None and outs[-1]["side"] == 1, outs
    s0 = outs[-1]["strength"]
    dec = tr.update(gate, [], shape)  # dropout -> decayed, not held
    assert dec is not None and dec["reason"] == "decay" and dec["strength"] < s0, dec
    print(f"[selftest] confirm@{OCC_CONFIRM_FRAMES}f, decay {s0:.2f}->{dec['strength']:.2f} OK")

    print("[selftest] OK -- occlusion PEEK cue")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frames", nargs="*", help="JPEGs -> brightness fallback (gate=None)")
    args = ap.parse_args()
    if args.frames:
        import cv2
        for f in args.frames:
            img = cv2.imread(f)
            shape = img.shape[:2] if img is not None else (360, 640)
            d = detect_occlusion(None, [], shape, bgr=img)
            print(f"{f}: {d}" if d else f"{f}: clear")
    else:
        _selftest()
