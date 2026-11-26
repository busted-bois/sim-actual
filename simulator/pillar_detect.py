"""Pillar / wall avoidance cue for the indoor gate course.

The VQ2 course is an indoor structure (parking-garage-like) whose gates sit
between large DARK vertical pillars and perimeter walls. The gate-chasing pilot
flies straight gate-to-gate and clips these pillars on the way between gates
(the "collides with something external" misses). This turns one camera frame
into a lateral STEER cue so guidance can avoid a pillar/wall blocking the path.

Signature (measured on real flight frames): the lit lanes (gates, track) are
bright; pillars and perimeter walls occlude to near-black. In the forward danger
band (mid-rows, at flight level below the up-tilted ceiling) two failure shapes
appear: a CENTRAL pillar (dark middle, open sides) and a SIDE WALL (one whole
side dark). Split the band into left/centre/right thirds:
  * centre much darker than the open sides -> pillar dead ahead -> steer to the
    brighter side;
  * one side much darker than the other -> wall -> steer to the brighter side.
On a balanced (open) scene the steer is ~0, so a clean gate approach is
undisturbed.

    detect_pillar(bgr) -> {steer: -1..+1 (neg=left), threat: 0..1,
                           bands: (L,C,R)} or None when clear.

    uv run python -m simulator.pillar_detect --selftest
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

BAND_TOP_FRAC = 0.35  # forward danger band: below ceiling, at flight level
BAND_BOT_FRAC = 0.68
PILLAR_DARK_FRAC = 0.6  # centre below this * open-side brightness = pillar ahead
WALL_IMBAL = 0.30  # |L-R|/(L+R) above this = a dark wall on one side
DOWNSCALE = 4


def detect_pillar(bgr, downscale: int = DOWNSCALE) -> dict | None:
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(3, w // downscale), max(3, h // downscale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    roi = gray[int(BAND_TOP_FRAC * H) : int(BAND_BOT_FRAC * H), :]
    if roi.size == 0:
        return None
    col = roi.mean(axis=0)
    t = W // 3
    left = float(col[:t].mean())
    center = float(col[t : 2 * t].mean())
    right = float(col[2 * t :].mean())
    open_side = max(left, right)
    if open_side < 1e-3:
        return None

    steer = 0.0
    threat = 0.0
    # (1) central pillar: middle third much darker than the brighter open side.
    if center < PILLAR_DARK_FRAC * open_side:
        steer = 1.0 if right > left else -1.0  # toward the brighter open side
        threat = float(
            np.clip(
                (PILLAR_DARK_FRAC * open_side - center)
                / (PILLAR_DARK_FRAC * open_side),
                0.0,
                1.0,
            )
        )
    else:
        # (2) side wall: one side much darker than the other.
        imbal = (right - left) / (right + left + 1e-3)
        if abs(imbal) > WALL_IMBAL:
            steer = 1.0 if imbal > 0 else -1.0  # toward the brighter side
            threat = float(np.clip(abs(imbal), 0.0, 1.0))
    if threat <= 0.0:
        return None
    return {"steer": steer, "threat": threat, "bands": (left, center, right)}


def _selftest() -> None:
    def base():
        return np.full((180, 320, 3), 130, np.uint8)

    def dark(img, x0, x1):
        img[:, x0:x1] = 5
        return img

    # central pillar -> steer to a side (nonzero), strong threat
    d = detect_pillar(dark(base(), 120, 200))
    assert d is not None and d["threat"] > 0.2 and d["steer"] != 0, d
    print(f"[selftest] centre pillar: steer={d['steer']:+.0f} threat={d['threat']:.2f}")

    # wall on the RIGHT -> steer LEFT (this is the f4 collision case)
    d = detect_pillar(dark(base(), 210, 320))
    assert d is not None and d["steer"] < 0, d
    print(
        f"[selftest] right wall -> steer={d['steer']:+.0f} (left) threat={d['threat']:.2f} OK"
    )

    # wall on the LEFT -> steer RIGHT
    d = detect_pillar(dark(base(), 0, 110))
    assert d is not None and d["steer"] > 0, d
    print(f"[selftest] left wall -> steer={d['steer']:+.0f} (right) OK")

    # open scene -> None (undisturbed gate approach)
    assert detect_pillar(base()) is None
    print("[selftest] open scene -> None OK")
    print("[selftest] OK — pillar/wall avoidance cue")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--frames", nargs="*")
    args = ap.parse_args()
    if args.frames:
        for f in args.frames:
            d = detect_pillar(cv2.imread(f))
            if d:
                print(
                    f"{f}: steer={d['steer']:+.0f} threat={d['threat']:.2f} bands={tuple(round(b) for b in d['bands'])}"
                )
            else:
                print(f"{f}: clear")
    else:
        _selftest()
