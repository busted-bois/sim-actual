"""Gate adapter — sim track data / gate_map.json -> STUDYONLY-style tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from simulator.transforms import quat_to_R

# Fallback when sim has not broadcast track_gates yet.
DEFAULT_GATE_MAP = Path(__file__).resolve().parents[1] / "rl" / "data" / "gate_map.json"


@dataclass(frozen=True)
class GateTrack:
    gate_id: int
    tvec: np.ndarray  # world-origin NED position (3,)
    rvec: np.ndarray  # Rodrigues rotation (3, 1)


def _quat_wxyz_to_rvec(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quat
    rvec, _ = cv2.Rodrigues(quat_to_R(w, x, y, z))
    return rvec


def track_from_gate_map_entry(entry: dict) -> GateTrack:
    pos = np.asarray(entry["pos"], dtype=np.float64)
    quat = tuple(float(c) for c in entry["quat"])
    return GateTrack(
        gate_id=int(entry.get("id", 0)),
        tvec=pos,
        rvec=_quat_wxyz_to_rvec(quat),
    )


def track_from_sim_gate(gate: dict, gate_id: int = 0) -> GateTrack | None:
    pos = gate.get("position_ned")
    quat = gate.get("orientation_ned")
    if not pos or not quat or len(pos) < 3 or len(quat) < 4:
        return None
    return GateTrack(
        gate_id=gate_id,
        tvec=np.asarray(pos[:3], dtype=np.float64),
        rvec=_quat_wxyz_to_rvec(tuple(float(c) for c in quat[:4])),
    )


def load_gate_map(path: Path | str = DEFAULT_GATE_MAP) -> list[GateTrack]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    gates = payload.get("gates", payload)
    tracks = [track_from_gate_map_entry(g) for g in gates]
    return sorted(tracks, key=lambda t: t.gate_id)


def tracks_from_sim_data(track_gates: list) -> list[GateTrack]:
    tracks: list[GateTrack] = []
    for i, gate in enumerate(track_gates):
        if hasattr(gate, "pos_ned"):
            raw = {
                "position_ned": gate.pos_ned,
                "orientation_ned": gate.orient_quat,
            }
            gate_id = int(getattr(gate, "gate_id", i))
        else:
            raw = gate
            gate_id = int(gate.get("gate_id", gate.get("id", i)))
        track = track_from_sim_gate(raw, gate_id=gate_id)
        if track is not None:
            tracks.append(track)
    return sorted(tracks, key=lambda t: t.gate_id)


def get_gate_tracks(data: dict) -> list[GateTrack]:
    """Prefer live sim gates; fall back to captured gate_map.json."""
    track_gates = data.get("track_gates") or data.get("gates")
    if track_gates:
        tracks = tracks_from_sim_data(track_gates)
        if tracks:
            return tracks
    return load_gate_map()


def gate_centers(tracks: list[GateTrack]) -> list[tuple[float, float, float]]:
    """Ordered gate-center waypoints (no TGL)."""
    return [
        (float(t.tvec[0]), float(t.tvec[1]), float(t.tvec[2]))
        for t in sorted(tracks, key=lambda tr: tr.gate_id)
    ]
