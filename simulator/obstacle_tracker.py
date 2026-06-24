"""Temporal tracking for obstacle detections — independent from gate KF."""

from __future__ import annotations

import time as _time

from simulator.config import ObstacleDetection


class ObstacleTracker:
    def __init__(self) -> None:
        self._tracks: list[dict] = []
        self._next_id = 0

    def update(
        self,
        detections: list[ObstacleDetection],
        frame_id: int,
        img_w: int,
        img_h: int,
    ) -> list[dict]:
        now = _time.monotonic()
        matched: set[int] = set()

        for det in detections:
            nx = (det.centroid_x_px - img_w / 2.0) / (img_w / 2.0)
            ny = (det.centroid_y_px - img_h / 2.0) / (img_h / 2.0)
            r_frac = det.area_px / max(img_w * img_h, 1)

            best_i = None
            best_dist = 0.20
            for i, tr in enumerate(self._tracks):
                if i in matched:
                    continue
                dist = abs(tr["nx"] - nx) + abs(tr["ny"] - ny)
                if dist < best_dist:
                    best_dist = dist
                    best_i = i

            if best_i is not None:
                tr = self._tracks[best_i]
                tr["nx"] = nx
                tr["ny"] = ny
                tr["r_frac"] = r_frac
                tr["streak"] = int(tr.get("streak", 0)) + 1
                tr["confidence"] = min(
                    1.0,
                    float(det.confidence) * (0.6 + 0.1 * min(tr["streak"], 4)),
                )
                tr["last_frame_id"] = frame_id
                tr["updated_at"] = now
                matched.add(best_i)
            else:
                self._tracks.append(
                    {
                        "id": self._next_id,
                        "nx": nx,
                        "ny": ny,
                        "r_frac": r_frac,
                        "confidence": float(det.confidence),
                        "streak": 1,
                        "last_frame_id": frame_id,
                        "updated_at": now,
                    }
                )
                self._next_id += 1

        self._tracks = [
            tr
            for i, tr in enumerate(self._tracks)
            if i in matched or (now - tr.get("updated_at", now)) < 0.5
        ]
        return [dict(tr) for tr in self._tracks]
