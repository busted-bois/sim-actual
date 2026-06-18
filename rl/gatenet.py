"""Module 3 — GateNet (U-Net CNN) for gate segmentation.

Trains a small U-Net on the (image, mask) pairs from Module 2 and writes
``gatenet.pt``. cv2/numpy augmentation (flip, brightness, gamma, noise).
Pure torch so it runs on CPU.

    uv run -m rl.gatenet --data rl/data/gatenet_ds --epochs 30
    uv run -m rl.gatenet --selftest        # synthetic end-to-end smoke
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TRAIN_W, TRAIN_H = 320, 192  # divisible by 16 for 4 pool levels
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "gatenet.pt")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Compact 4-level U-Net, 1-channel logit output."""

    def __init__(self, in_ch=3, base=16):
        super().__init__()
        self.d1 = _DoubleConv(in_ch, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.d4 = _DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = _DoubleConv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = _DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = _DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = _DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        xb = self.bott(self.pool(x4))
        y = self.c4(torch.cat([self.u4(xb), x4], 1))
        y = self.c3(torch.cat([self.u3(y), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1))
        y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.head(y)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _augment(img: np.ndarray, mask: np.ndarray):
    if np.random.rand() < 0.5:  # horizontal flip
        img, mask = img[:, ::-1], mask[:, ::-1]
    if np.random.rand() < 0.7:  # brightness/contrast
        a = 1.0 + np.random.uniform(-0.3, 0.3)
        b = np.random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.4:  # gamma
        g = np.random.uniform(0.7, 1.4)
        lut = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if np.random.rand() < 0.3:  # gaussian noise
        img = np.clip(
            img.astype(np.float32) + np.random.randn(*img.shape) * 8, 0, 255
        ).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


class GateDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.imgs = sorted(glob.glob(os.path.join(root, "images", "*.png")))
        # Build mask paths from the basename + explicit masks dir; str.replace
        # would corrupt any earlier path segment that contains "images".
        self.masks = [
            os.path.join(root, "masks", os.path.basename(p)) for p in self.imgs
        ]
        self.train = train
        if not self.imgs:
            raise FileNotFoundError(f"no images under {root}/images")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = cv2.imread(self.imgs[i], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_NEAREST)
        if self.train:
            img, mask = _augment(img, mask)
        m = (mask > 127).astype(np.float32)[None]
        return _to_tensor(img), torch.from_numpy(m)


def _dice_bce(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(
    data_root: str,
    epochs: int = 30,
    bs: int = 8,
    lr: float = 1e-3,
    out: str = WEIGHTS_PATH,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = GateDataset(data_root, train=True)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"[gatenet] {len(ds)} samples, device={device}", flush=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for img, m in dl:
            img, m = img.to(device), m.to(device)
            opt.zero_grad()
            loss = _dice_bce(model(img), m)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        print(
            f"[gatenet] epoch {ep + 1}/{epochs} loss={tot / max(len(dl), 1):.4f}",
            flush=True,
        )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": "unet_base16",
            "train_wh": [TRAIN_W, TRAIN_H],
        },
        out,
    )
    print(f"[gatenet] saved -> {out}", flush=True)


# ----------------------------------------------------------------------------
# Inference (used live by Module 4)
# ----------------------------------------------------------------------------
class GateNetInfer:
    def __init__(self, weights: str = WEIGHTS_PATH, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=self.device)
        self.model = UNet().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def mask(self, img_bgr: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (TRAIN_W, TRAIN_H), interpolation=cv2.INTER_AREA)
        x = _to_tensor(small)[None].to(self.device)
        p = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        p = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
        return (p > thresh).astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Synthetic end-to-end smoke test (no real data needed)
# ----------------------------------------------------------------------------
def _make_synthetic_ds(root: str, n: int = 24):
    import numpy as np

    from rl import spec
    from rl.dataset import project_gate_mask

    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "masks"), exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        dist = rng.uniform(3, 8)
        lat = rng.uniform(-1.5, 1.5)
        alt = rng.uniform(-1.5, 1.5)
        gate = [
            {"id": 0, "pos": [dist, lat, alt], "quat": [0, 0, 0, 1], "w": 1.5, "h": 1.5}
        ]
        mask = project_gate_mask(
            (spec.IMG_H, spec.IMG_W), np.zeros(3), np.array([1.0, 0, 0, 0]), gate
        )
        img = rng.integers(20, 70, (spec.IMG_H, spec.IMG_W, 3), dtype=np.uint8)
        img[mask > 0] = (10, 60, 240)  # orange-ish gate on dark bg (BGR)
        cv2.imwrite(os.path.join(root, "images", f"{i:06d}.png"), img)
        cv2.imwrite(os.path.join(root, "masks", f"{i:06d}.png"), mask)


def _selftest():
    import tempfile

    # 1. forward/backward shape check
    m = UNet()
    x = torch.randn(2, 3, TRAIN_H, TRAIN_W)
    y = m(x)
    assert y.shape == (2, 1, TRAIN_H, TRAIN_W), y.shape
    loss = _dice_bce(y, torch.zeros_like(y))
    loss.backward()
    print(
        f"[selftest] UNet fwd/bwd OK out={tuple(y.shape)} "
        f"params={sum(p.numel() for p in m.parameters()) / 1e3:.0f}k"
    )
    # 2. one real training run on synthetic data + checkpoint round-trip
    tmp = tempfile.mkdtemp()
    _make_synthetic_ds(tmp, n=24)
    out = os.path.join(tmp, "gatenet.pt")
    train(tmp, epochs=2, bs=4, out=out, device="cpu")
    infer = GateNetInfer(out, device="cpu")
    test_img = cv2.imread(sorted(glob.glob(os.path.join(tmp, "images", "*.png")))[0])
    pred = infer.mask(test_img)
    print(f"[selftest] inference mask shape={pred.shape} fg_px={int((pred > 0).sum())}")
    print("[selftest] OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data", default=os.path.join(os.path.dirname(__file__), "data", "gatenet_ds")
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        train(args.data, epochs=args.epochs, bs=args.bs)
