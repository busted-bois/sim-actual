"""
Offline color-detection preview / HSV tuning helper.

Runs the SAME detection pipeline the drone uses on a still image (or a generated
synthetic scene) and writes an annotated result plus the gate/path masks next to
it. Use this to sanity-check or tune the HSV ranges in settings.json without
needing the simulator running.

    # generate a synthetic gate+path scene and analyze it
    uv run python tools/vision_preview.py

    # analyze a real screenshot / saved frame
    uv run python tools/vision_preview.py path/to/frame.jpg

Outputs <name>_annotated.jpg, <name>_gatemask.jpg, <name>_pathmask.jpg.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import load_config  # noqa: E402
from simulator.vision_processing import analyze_frame, annotate, color_mask  # noqa: E402


def _synthetic_scene():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)  # dark gray background
    cv2.rectangle(img, (250, 380), (390, 470), (255, 0, 0), -1)  # blue path near bottom
    cv2.rectangle(
        img, (360, 160), (480, 300), (15, 57, 243), -1
    )  # orange gate, right of center
    return img


def main():
    cfg = load_config()
    vision_cfg = cfg["vision"]

    if len(sys.argv) > 1:
        src = sys.argv[1]
        img = cv2.imread(src)
        if img is None:
            print(f"Could not read image: {src}")
            sys.exit(1)
        base = os.path.splitext(src)[0]
    else:
        img = _synthetic_scene()
        base = "vision_preview_synthetic"
        print("No image given; using a synthetic gate+path scene.")

    analysis = analyze_frame(img, vision_cfg)
    print("Gate:", analysis.gate)
    print("Path:", analysis.path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gate_mask = color_mask(
        hsv, vision_cfg["gate_hsv_ranges"], vision_cfg.get("morph_ksize", 5)
    )
    path_mask = color_mask(
        hsv, vision_cfg["path_hsv_ranges"], vision_cfg.get("morph_ksize", 5)
    )

    cv2.imwrite(f"{base}_annotated.jpg", annotate(img, analysis))
    cv2.imwrite(f"{base}_gatemask.jpg", gate_mask)
    cv2.imwrite(f"{base}_pathmask.jpg", path_mask)
    print(f"Wrote {base}_annotated.jpg, {base}_gatemask.jpg, {base}_pathmask.jpg")


if __name__ == "__main__":
    main()
