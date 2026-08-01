"""Tests for the occlusion PEEK cue (simulator/gate_occlusion.py). unittest idiom."""

import unittest

import numpy as np

from simulator.gate_occlusion import (
    OCC_CONFIRM_FRAMES,
    PEEK_TIEBREAK_SIDE,
    OcclusionTracker,
    detect_occlusion,
)

_SHAPE = (360, 640)
# Gate centred and wide: x 220..420 (cx 320), y 90..270.
_GATE = {"cx": 320, "cy": 180, "x0": 220, "x1": 420, "y0": 90, "y1": 270}


def _blob(x0, x1, y0=170, y1=210, area=6000):
    return [{
        "x0": x0, "x1": x1, "y0": y0, "y1": y1, "area_px": area,
        "nx": 0.0, "ny": 0.0, "r_frac": area / (640 * 360),
    }]


class DetectOcclusionTests(unittest.TestCase):
    def test_pillar_left_of_gate_peeks_right(self):
        d = detect_occlusion(_GATE, _blob(220, 280), _SHAPE)
        self.assertIsNotNone(d)
        self.assertEqual(d["side"], 1)  # gate mass mostly right -> peek right
        self.assertGreater(d["strength"], 0.0)

    def test_pillar_right_of_gate_peeks_left(self):
        d = detect_occlusion(_GATE, _blob(360, 420), _SHAPE)
        self.assertIsNotNone(d)
        self.assertEqual(d["side"], -1)

    def test_balanced_pillar_uses_tiebreak(self):
        d = detect_occlusion(_GATE, _blob(300, 340), _SHAPE)
        self.assertIsNotNone(d)
        self.assertEqual(d["side"], PEEK_TIEBREAK_SIDE)
        self.assertEqual(d["gate_lr"], "tiebreak")

    def test_clean_scene_is_none(self):
        # Gate visible, no obstacle -> inert (gates 1-2 untouched).
        self.assertIsNone(detect_occlusion(_GATE, [], _SHAPE))

    def test_subfloor_blob_ignored(self):
        self.assertIsNone(detect_occlusion(_GATE, _blob(300, 340, area=500), _SHAPE))

    def test_out_of_band_blob_ignored(self):
        # Blob near the top of frame, above the forward danger band.
        self.assertIsNone(detect_occlusion(_GATE, _blob(300, 340, y0=10, y1=40), _SHAPE))

    def test_no_gate_uses_brightness_fallback(self):
        # gate=None -> classical pillar_detect on the frame. Central dark pillar.
        img = np.full((180, 320, 3), 130, np.uint8)
        img[:, 120:200] = 5
        d = detect_occlusion(None, [], (180, 320), bgr=img)
        self.assertIsNotNone(d)
        self.assertIn(d["side"], (-1, 1))
        self.assertEqual(d["reason"], "pillar_detect")

    def test_no_gate_open_scene_none(self):
        img = np.full((180, 320, 3), 130, np.uint8)
        self.assertIsNone(detect_occlusion(None, [], (180, 320), bgr=img))


class OcclusionTrackerTests(unittest.TestCase):
    def test_confirm_then_decay_never_holds(self):
        tr = OcclusionTracker()
        outs = [tr.update(_GATE, _blob(220, 280), _SHAPE) for _ in range(OCC_CONFIRM_FRAMES)]
        self.assertIsNone(outs[0])           # not yet confirmed
        self.assertIsNotNone(outs[-1])       # confirmed after N frames
        self.assertEqual(outs[-1]["side"], 1)
        s0 = outs[-1]["strength"]

        first = tr.update(_GATE, [], _SHAPE)  # dropout -> decays, keeps side
        self.assertIsNotNone(first)
        self.assertEqual(first["reason"], "decay")
        self.assertLess(first["strength"], s0)

        # Enough dropout frames -> fully released (never held forever).
        last = first
        for _ in range(30):
            last = tr.update(_GATE, [], _SHAPE)
        self.assertIsNone(last)

    def test_side_flip_requires_reconfirm(self):
        tr = OcclusionTracker()
        for _ in range(OCC_CONFIRM_FRAMES):
            tr.update(_GATE, _blob(220, 280), _SHAPE)      # confirmed side +1
        flipped = tr.update(_GATE, _blob(360, 420), _SHAPE)  # opposite side, 1 frame
        self.assertIsNone(flipped)  # must re-confirm before emitting the new side


if __name__ == "__main__":
    unittest.main()
