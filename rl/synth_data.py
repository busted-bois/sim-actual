"""Synthetic gate-image generator — the Python stand-in for Matt's Unity data
factory. Renders thousands of gates with our known 3D geometry + camera model
onto randomized backgrounds, and auto-labels the 4 opening corners exactly
(we drew them, so we know where they are). Heavy domain randomization (pose,
lighting, colour, blur, noise, occluders, multi-gate) is what makes the trained
YOLO hold up on the real sim — same idea that got Matt to a flight-robust
detector without ever hand-labelling a frame.

Output goes into the pose dataset (rl/data/gate_pose_ds), so `make train-gatepose`
trains on it directly.

    uv run -m rl.synth_data                 # generate the default dataset
    uv run -m rl.synth_data --n 6000
    uv run -m rl.synth_data --preview 8     # write 8 sample renders to data/synth_preview
    uv run -m rl.synth_data --selftest      # offline geometry/label sanity
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from rl import spec
from rl.dataset import POSE_OUT_DIR, _yolo_pose_line, write_pose_data_yaml

W, H = spec.IMG_W, spec.IMG_H
FX, FY, CX, CY = spec.FX, spec.FY, spec.CX, spec.CY
GH = spec.GATE_HALF  # outer half-width 1.36 m (matches pnp._OBJ)
GH_IN = 0.75  # inner aperture half-width (~1.5 m gate, like Matt's inner square)

# Outer + inner corners in gate-local frame (x = normal, y = right, z = down),
# TL, TR, BR, BL — same order as spec.GATE_CORNERS_LOCAL so labels feed PnP.
_OUTER = np.array(
    [[0, -GH, -GH], [0, GH, -GH], [0, GH, GH], [0, -GH, GH]], dtype=np.float64
)
_INNER = np.array(
    [[0, -GH_IN, -GH_IN], [0, GH_IN, -GH_IN], [0, GH_IN, GH_IN], [0, -GH_IN, GH_IN]],
    dtype=np.float64,
)


def _rot(rng):
    """Random small rotation matrix (gate seen from varied angles)."""
    a = rng.uniform(-0.5, 0.5)  # yaw about z
    b = rng.uniform(-0.35, 0.35)  # pitch about y
    c = rng.uniform(-0.25, 0.25)  # roll about x
    ca, sa, cb, sb, cc, sc = (
        np.cos(a),
        np.sin(a),
        np.cos(b),
        np.sin(b),
        np.cos(c),
        np.sin(c),
    )
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return rz @ ry @ rx


def _project_cam(pts_cam):
    """Camera-frame points (N,3) -> pixels (N,2) + in-front mask."""
    z = np.clip(pts_cam[:, 2], 1e-3, None)
    u = FX * pts_cam[:, 0] / z + CX
    v = FY * pts_cam[:, 1] / z + CY
    return np.stack([u, v], 1), pts_cam[:, 2] > 0.2


def _gate_in_cam(rng, depth):
    """Place a gate at `depth` metres with random lateral/vertical offset and
    orientation; return outer+inner corner pixels (camera frame)."""
    # Lateral/vertical offset scaled so the gate roughly frames on-screen.
    spread = 0.45 * depth
    t = np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread), depth])
    R = _rot(rng)
    outer_cam = (_OUTER @ R.T) + t
    inner_cam = (_INNER @ R.T) + t
    op, of = _project_cam(outer_cam)
    ip, _ = _project_cam(inner_cam)
    return op, of, ip


# ---------------------------------------------------------------------------
# Background + gate rendering (rough match to the sim: dark scene, grey angular
# structures, a blue glowing track strip, orange/red gates)
# ---------------------------------------------------------------------------
def _background(rng):
    base = rng.integers(0, 40)
    img = np.full((H, W, 3), base, np.uint8)
    # grey angular "buildings" along the sides
    for _ in range(rng.integers(8, 22)):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        w = rng.integers(10, 90)
        h = rng.integers(20, 200)
        g = int(rng.integers(40, 150))
        cv2.rectangle(img, (x, y), (x + w, y + h), (g, g, g), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (min(g + 40, 255),) * 3, 1)
    # blue glowing track strip near the bottom (50% of frames)
    if rng.random() < 0.5:
        cx = rng.integers(W // 3, 2 * W // 3)
        pts = np.array(
            [
                [cx - 120, H],
                [cx - 40, H // 2 + 30],
                [cx + 40, H // 2 + 30],
                [cx + 120, H],
            ],
            np.int32,
        )
        cv2.fillPoly(
            img, [pts], (int(rng.integers(60, 130)), int(rng.integers(40, 90)), 10)
        )
    return img


def _draw_gate(img, outer_px, inner_px, rng):
    """Draw an orange/red square ring (outer filled, inner hole = background)."""
    o = np.round(outer_px).astype(np.int32)
    i = np.round(inner_px).astype(np.int32)
    # gate colour ~ orange/red (BGR) with variation
    col = (
        int(rng.integers(5, 40)),
        int(rng.integers(30, 90)),
        int(rng.integers(180, 255)),
    )
    ring = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ring, [o], 255)
    cv2.fillPoly(ring, [i], 0)  # cut the opening
    shade = rng.uniform(0.7, 1.0)
    img[ring > 0] = tuple(int(c * shade) for c in col)
    # thin bright edge on the frame
    cv2.polylines(img, [o], True, tuple(min(int(c * 1.2), 255) for c in col), 1)
    cv2.polylines(img, [i], True, tuple(min(int(c * 1.2), 255) for c in col), 1)


def _label_for(outer_px, outer_front, inner_px):
    """YOLO-pose label dict for a rendered gate (outer 4 corners), or None if
    too few corners are usable. Off-screen/behind corners -> visibility 0."""
    kpts, xs, ys = [], [], []
    for k in range(4):
        x, y = float(outer_px[k, 0]), float(outer_px[k, 1])
        on = bool(outer_front[k]) and 0 <= x < W and 0 <= y < H
        if on:
            kpts.append((x / W, y / H, 2))
            xs.append(x)
            ys.append(y)
        else:
            kpts.append((0.0, 0.0, 0))
    if len(xs) < 2:
        return None
    x0, x1 = max(0.0, min(xs)), min(float(W), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(H), max(ys))
    bw, bh = (x1 - x0) / W, (y1 - y0) / H
    if bw <= 0 or bh <= 0:
        return None
    return {"kpts": kpts, "bbox": ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, bw, bh)}


def render_scene(rng):
    """Render one synthetic frame + its YOLO-pose labels (1-3 gates)."""
    img = _background(rng)
    labels = []
    depths = sorted(
        rng.uniform(3.0, 24.0, size=int(rng.integers(1, 4))), reverse=True
    )  # far gates first so near ones draw on top
    for d in depths:
        op, of, ip = _gate_in_cam(rng, d)
        # cheap cull: skip gates entirely off-screen
        if (
            op[:, 0].max() < 0
            or op[:, 0].min() > W
            or op[:, 1].max() < 0
            or op[:, 1].min() > H
        ):
            continue
        _draw_gate(img, op, ip, rng)
        lab = _label_for(op, of, ip)
        if lab is not None:
            labels.append(lab)

    # --- domain randomization on the whole frame ---
    if rng.random() < 0.5:  # random occluders over the scene
        for _ in range(rng.integers(1, 4)):
            x, y = rng.integers(0, W), rng.integers(0, H)
            s = rng.integers(15, 70)
            g = int(rng.integers(0, 120))
            cv2.rectangle(img, (x, y), (x + s, y + s), (g, g, g), -1)
    # brightness / contrast
    a = rng.uniform(0.7, 1.3)
    b = rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    # hue shift
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.integers(-12, 12)) % 180
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # blur (motion/defocus proxy)
    if rng.random() < 0.5:
        kk = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (kk, kk), 0)
    # noise
    if rng.random() < 0.5:
        img = np.clip(
            img.astype(np.float32) + rng.normal(0, rng.uniform(3, 12), img.shape),
            0,
            255,
        ).astype(np.uint8)
    return img, labels


def generate(
    n: int = 5000, out_dir: str = POSE_OUT_DIR, val_every: int = 6, seed: int = 0
):
    rng = np.random.default_rng(seed)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)
    for i in range(n):
        img, labels = render_scene(rng)
        # keep ~25% gate-less frames as negatives
        if not labels and (i % 4) != 0:
            continue
        sp = "val" if (i % val_every == 0) else "train"
        cv2.imwrite(os.path.join(out_dir, "images", sp, f"s{i:06d}.png"), img)
        with open(os.path.join(out_dir, "labels", sp, f"s{i:06d}.txt"), "w") as f:
            f.write("\n".join(_yolo_pose_line(b) for b in labels))
        if (i + 1) % 500 == 0:
            print(f"[synth] {i + 1}/{n}", flush=True)
    write_pose_data_yaml(out_dir)
    print(f"[synth] done: {n} frames -> {out_dir}", flush=True)


def _preview(k: int):
    rng = np.random.default_rng(1)
    d = os.path.join(os.path.dirname(__file__), "data", "synth_preview")
    os.makedirs(d, exist_ok=True)
    for i in range(k):
        img, labels = render_scene(rng)
        for lab in labels:
            for x, y, v in lab["kpts"]:
                if v > 0:
                    cv2.circle(img, (int(x * W), int(y * H)), 4, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(d, f"preview_{i:02d}.png"), img)
    print(f"[synth] wrote {k} previews (with corner dots) -> {d}", flush=True)


def _selftest():
    rng = np.random.default_rng(3)
    # A head-on gate at a known depth projects to a sane, centred square.
    op, of, ip = _gate_in_cam(np.random.default_rng(0), depth=8.0)
    assert of.sum() == 4, "all 4 outer corners in front"
    img, labels = render_scene(rng)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    # render many scenes; most should produce at least one labelled gate
    hit = 0
    for _ in range(50):
        _, labs = render_scene(rng)
        if labs:
            hit += 1
            lab = labs[0]
            assert len(lab["kpts"]) == 4
            for x, y, v in lab["kpts"]:
                assert 0 <= x <= 1 and 0 <= y <= 1 and v in (0, 2)
            cx, cy, bw, bh = lab["bbox"]
            assert 0 < bw <= 1 and 0 < bh <= 1
    assert hit > 35, f"most scenes should have a gate, got {hit}/50"
    print(f"[selftest] rendered scenes OK — {hit}/50 had labelled gates")
    print("[selftest] OK — synthetic gates project + label correctly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument(
        "--preview", type=int, default=0, help="write N sample renders and exit"
    )
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.preview:
        _preview(args.preview)
    else:
        generate(args.n)
