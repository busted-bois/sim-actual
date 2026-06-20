import numpy as np

from simulator.vision_processing import (
    GateTargetFilter,
    blue_ring_info_normalized,
    detect_gate_target,
    find_blue_rings,
)


def _synthetic_blue_ring(width=640, height=360, cx=400, cy=180, radius=40):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # BGR blue
    cv2 = __import__("cv2")
    cv2.circle(image, (cx, cy), radius, (255, 120, 40), thickness=8)
    return image


def test_find_blue_rings_detects_synthetic_target():
    image = _synthetic_blue_ring()
    circles = find_blue_rings(image)
    assert circles


def test_blue_ring_info_normalized_center_offset():
    image = _synthetic_blue_ring(cx=480, cy=180)
    info = blue_ring_info_normalized(image)
    assert info is not None
    nx, ny, r_frac = info
    assert nx > 0.1
    assert abs(ny) < 0.2
    assert r_frac > 0.02


def test_detect_gate_target_negative_when_empty():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = detect_gate_target(image)
    assert result["detected"] is False
    assert result["nx"] == 0.0


def test_detect_gate_target_positive_on_blue_ring():
    image = _synthetic_blue_ring()
    result = detect_gate_target(image)
    assert result["detected"] is True
    assert result["r_frac"] > 0.0


def test_gate_target_filter_smooths_jitter():
    filt = GateTargetFilter(alpha=0.5, max_jump=0.9)
    t1 = {"detected": True, "nx": 0.2, "ny": 0.0, "r_frac": 0.1}
    t2 = {"detected": True, "nx": 0.4, "ny": 0.0, "r_frac": 0.1}
    filt.apply(t1)
    out = filt.apply(t2)
    assert abs(out["nx"] - 0.3) < 1e-6


def test_gate_target_filter_rejects_outlier():
    filt = GateTargetFilter(alpha=0.5, max_jump=0.2)
    t1 = {"detected": True, "nx": 0.1, "ny": 0.0, "r_frac": 0.1}
    t2 = {"detected": True, "nx": 0.9, "ny": 0.0, "r_frac": 0.1}
    filt.apply(t1)
    out = filt.apply(t2)
    assert out["nx"] == 0.1


def test_desaturated_blue_ring_detected():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(image, (320, 180), 45, (180, 140, 120), thickness=10)
    result = detect_gate_target(image)
    assert result["detected"] is True
