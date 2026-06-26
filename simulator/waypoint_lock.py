"""Simple waypoint lock — hold target until arrival, then advance sequentially."""

from __future__ import annotations

import numpy as np

ARRIVAL_RADIUS_M = 6.0


class WaypointLock:
    """Lock one waypoint at a time; advance on arrival within radius."""

    def __init__(self, arrival_radius_m: float = ARRIVAL_RADIUS_M):
        self.arrival_radius_m = arrival_radius_m
        self._locked: tuple[float, float, float] | None = None
        self._index = 0

    @property
    def waypoint_index(self) -> int:
        return self._index

    def reset(self, start_index: int = 0) -> None:
        self._locked = None
        self._index = max(0, start_index)

    def sync_index(self, index: int, n_waypoints: int) -> None:
        """Jump forward to sim active_gate_index without rewinding."""
        if 0 <= index < n_waypoints and index > self._index:
            self._index = index
            self._locked = None

    def compute_target(
        self,
        drone_ned: np.ndarray,
        waypoints: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        if not waypoints:
            return (float(drone_ned[0]), float(drone_ned[1]), float(drone_ned[2]))

        if self._index >= len(waypoints):
            last = waypoints[-1]
            return (float(last[0]), float(last[1]), float(last[2]))

        drone = np.asarray(drone_ned[:3], dtype=np.float64)

        if self._locked is not None:
            dist = float(np.linalg.norm(drone - np.asarray(self._locked)))
            if dist < self.arrival_radius_m:
                self._locked = None
                self._index += 1
                if self._index >= len(waypoints):
                    last = waypoints[-1]
                    return (float(last[0]), float(last[1]), float(last[2]))

        if self._locked is None and self._index < len(waypoints):
            wp = waypoints[self._index]
            self._locked = (float(wp[0]), float(wp[1]), float(wp[2]))

        if self._locked is None:
            return (float(drone_ned[0]), float(drone_ned[1]), float(drone_ned[2]))
        return self._locked
