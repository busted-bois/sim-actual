"""Module 4c — world-frame gate tracker (the "world_building" stage).

perception.py yields a gate-center position in WORLD NED each frame. This fuses
those noisy per-frame measurements into stable tracks via nearest-neighbour
association (world frame, so a gate stays one track as the drone moves) + EMA
smoothing. corrected_map() then snaps the broadcast gate map's POSITIONS onto
the matched tracks, keeping the broadcast map's gate order / orientation / size
but flying vision-refined positions.

Robust to drift / no-GPS: the policy observation is gate-minus-drone in the body
frame. Both the track and the observation are built from the SAME (possibly
drifting) drone pose, so common-mode drift cancels — relative geometry stays
correct even if absolute world pose is off. This is what removes the dependence
on the start-of-race broadcast (which corrupts on a fly-away).

    uv run -m rl.gate_tracker --selftest
"""

from __future__ import annotations

import argparse

import numpy as np

MATCH_DIST_M = 2.5  # a measurement within this of a track = same gate
EMA_ALPHA = 0.35  # weight of the newest measurement in the running mean
EXPIRE_S = 3.0  # drop tracks not seen for this long


class GateTracker:
    def __init__(self, match_dist=MATCH_DIST_M, alpha=EMA_ALPHA, expire_s=EXPIRE_S):
        self.match_dist = match_dist
        self.alpha = alpha
        self.expire_s = expire_s
        self._tracks: list[dict] = []  # {pos(3), n, last_t}

    def update(self, positions, t: float):
        """Fold a frame's gate world-positions into the tracks."""
        for p in positions:
            p = np.asarray(p, float)
            j, best = -1, self.match_dist
            for i, tr in enumerate(self._tracks):
                d = float(np.linalg.norm(tr["pos"] - p))
                if d < best:
                    best, j = d, i
            if j < 0:
                self._tracks.append({"pos": p.copy(), "n": 1, "last_t": t})
            else:
                tr = self._tracks[j]
                tr["pos"] = (1 - self.alpha) * tr["pos"] + self.alpha * p
                tr["n"] += 1
                tr["last_t"] = t
        self._tracks = [tr for tr in self._tracks if t - tr["last_t"] <= self.expire_s]

    def tracks(self, min_n: int = 1) -> list[np.ndarray]:
        return [tr["pos"].copy() for tr in self._tracks if tr["n"] >= min_n]

    def corrected_map(self, broadcast_map: list, min_n: int = 2) -> list:
        """Broadcast gate map with each position snapped onto its matched track."""
        out = []
        for g in broadcast_map:
            gp = np.asarray(g["pos"], float)
            j, best = -1, self.match_dist
            for i, tr in enumerate(self._tracks):
                if tr["n"] < min_n:
                    continue
                d = float(np.linalg.norm(tr["pos"] - gp))
                if d < best:
                    best, j = d, i
            ng = dict(g)
            if j >= 0:
                ng["pos"] = list(self._tracks[j]["pos"])
            out.append(ng)
        return out


# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(5)
    truth = [np.array([6.0, 0.0, 0.0]), np.array([11.0, 2.0, -1.0])]
    trk = GateTracker()
    dt = 1 / 30.0
    for k in range(120):
        t = k * dt
        # Each frame: noisy measurement of whichever gates are "seen".
        meas = [g + rng.normal(0, 0.25, 3) for g in truth]
        trk.update(meas, t)
    got = trk.tracks(min_n=2)
    assert len(got) == 2, f"should track exactly 2 gates, got {len(got)}"
    # Match tracks back to truth and check convergence.
    for g in truth:
        nearest = min(got, key=lambda p: np.linalg.norm(p - g))
        err = float(np.linalg.norm(nearest - g))
        assert err < 0.2, f"track should converge near truth, err={err:.3f}"

    # corrected_map: broadcast positions offset by ~0.6 m get snapped to tracks.
    bm = [
        {"pos": [6.6, 0.5, 0.2], "quat": [0, 0, 0, 1], "w": 2.72, "h": 2.72},
        {"pos": [11.5, 2.4, -0.7], "quat": [0, 0, 0, 1], "w": 2.72, "h": 2.72},
    ]
    cm = trk.corrected_map(bm)
    assert len(cm) == 2
    for g_in, g_out in zip(bm, cm):
        moved = np.linalg.norm(np.array(g_out["pos"]) - np.array(g_in["pos"]))
        assert moved > 0.2, "broadcast position should be corrected toward track"
        assert g_out["quat"] == g_in["quat"], "orientation/size preserved"
    # Expiry: nothing seen for > EXPIRE_S clears the tracks.
    trk.update([], t=1000.0)
    assert trk.tracks() == []
    print("[selftest] 2 gates tracked, converged <0.2m, map corrected, expiry OK")
    print("[selftest] OK — world-frame association + EMA smoothing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
