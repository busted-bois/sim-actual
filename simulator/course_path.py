"""Course path (blue track) derived from sequential gate centers."""

from __future__ import annotations

import math


def rebuild_course_path(data: dict) -> None:
    """Build polyline segments between gates — the sim's blue path."""
    gates = data.get("track_gates") or []
    waypoints: list[tuple[float, float, float]] = []
    segments: list[dict] = []
    for i, gate in enumerate(gates):
        pos = gate.get("position_ned")
        if not pos or len(pos) < 3:
            continue
        p = (float(pos[0]), float(pos[1]), float(pos[2]))
        waypoints.append(p)
        if i > 0:
            prev = waypoints[i - 1]
            segments.append(
                {
                    "from_idx": i - 1,
                    "to_idx": i,
                    "start": prev,
                    "end": p,
                    "bearing_rad": math.atan2(p[1] - prev[1], p[0] - prev[0]),
                    "length_m": math.hypot(p[0] - prev[0], p[1] - prev[1]),
                }
            )
    data["course_path"] = {
        "waypoints": waypoints,
        "segments": segments,
        "gate_count": len(waypoints),
    }


def gate_at_index(data: dict, index: int) -> dict | None:
    gates = data.get("track_gates") or []
    if 0 <= index < len(gates):
        return gates[index]
    return None


def segment_for_index(data: dict, to_idx: int) -> dict | None:
    """Path segment leading into gate `to_idx` (from previous gate)."""
    for seg in (data.get("course_path") or {}).get("segments", []):
        if int(seg.get("to_idx", -1)) == to_idx:
            return seg
    return None


def cross_track_m(
    pos: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    """Signed lateral offset from path segment (positive = right of path)."""
    sx, sy = seg_start[0], seg_start[1]
    ex, ey = seg_end[0], seg_end[1]
    dx, dy = ex - sx, ey - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return 0.0
    px, py = pos[0] - sx, pos[1] - sy
    return (dx * py - dy * px) / math.sqrt(seg_len_sq)


def along_track_m(
    pos: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> float:
    """Meters along segment from start to pos projection ( > length = past end )."""
    sx, sy = seg_start[0], seg_start[1]
    ex, ey = seg_end[0], seg_end[1]
    dx, dy = ex - sx, ey - sy
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return 0.0
    px, py = pos[0] - sx, pos[1] - sy
    return (px * dx + py * dy) / math.sqrt(seg_len_sq)


def path_bearing_to(
    pos: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return math.atan2(target[1] - pos[1], target[0] - pos[0])
