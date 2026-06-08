"""
Tests for color detection. These run on synthetic frames, so no simulator is
needed. Run with either:

    uv run python -m pytest tests/test_vision_processing.py
    uv run python tests/test_vision_processing.py        # standalone, no pytest
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import DEFAULT_CONFIG  # noqa: E402
from simulator.vision_processing import analyze_frame  # noqa: E402

VISION_CFG = DEFAULT_CONFIG["vision"]

# The real gate color, in BGR (OpenCV's channel order) for drawing test shapes.
GATE_BGR = (15, 57, 243)  # #f3390f
PATH_BGR = (255, 0, 0)  # pure blue


def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detects_gate_color():
    img = _blank()
    cv2.rectangle(img, (400, 200), (520, 320), GATE_BGR, -1)  # blob in the right half
    a = analyze_frame(img, VISION_CFG)
    assert a.gate.found
    assert a.gate.cx_norm > 0.1, (
        f"expected gate to the right, got {a.gate.cx_norm:+.2f}"
    )
    assert a.gate.area_frac > 0.0


def test_gate_left_vs_right_sign():
    left = _blank()
    cv2.rectangle(left, (40, 200), (160, 320), GATE_BGR, -1)
    right = _blank()
    cv2.rectangle(right, (480, 200), (600, 320), GATE_BGR, -1)
    assert analyze_frame(left, VISION_CFG).gate.cx_norm < 0
    assert analyze_frame(right, VISION_CFG).gate.cx_norm > 0


def test_blank_frame_no_detection():
    a = analyze_frame(_blank(), VISION_CFG)
    assert not a.gate.found
    assert not a.path.found
    assert not a.any_detection


def test_detects_blue_path():
    img = _blank()
    cv2.rectangle(img, (260, 380), (380, 470), PATH_BGR, -1)
    a = analyze_frame(img, VISION_CFG)
    assert a.path.found
    assert abs(a.path.cx_norm) < 0.2  # roughly centered horizontally


def test_bigger_blob_has_bigger_area_frac():
    small = _blank()
    cv2.rectangle(small, (300, 220), (340, 260), GATE_BGR, -1)
    big = _blank()
    cv2.rectangle(big, (200, 120), (440, 360), GATE_BGR, -1)
    assert (
        analyze_frame(big, VISION_CFG).gate.area_frac
        > analyze_frame(small, VISION_CFG).gate.area_frac
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
