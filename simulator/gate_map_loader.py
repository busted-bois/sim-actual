"""Load course gate map from rl/data/gate_map.json for early telemetry steering."""

from __future__ import annotations

import json
from pathlib import Path

_GATE_MAP_PATH = Path(__file__).resolve().parent.parent / "rl" / "data" / "gate_map.json"


def load_gate_map_into(data: dict) -> int:
    """Seed shared_data['track_gates'] if not already set. Returns gate count."""
    if data.get("track_gates"):
        from simulator.course_path import rebuild_course_path

        rebuild_course_path(data)
        return len(data["track_gates"])
    if not _GATE_MAP_PATH.is_file():
        return 0

    raw = json.loads(_GATE_MAP_PATH.read_text(encoding="utf-8"))
    gates = []
    for g in raw.get("gates", []):
        pos = g.get("pos")
        quat = g.get("quat")
        if not pos or not quat or len(pos) < 3 or len(quat) < 4:
            continue
        gates.append(
            {
                "gate_id": g.get("id", len(gates)),
                "position_ned": tuple(float(x) for x in pos[:3]),
                "orientation_ned": tuple(float(x) for x in quat[:4]),
                "width": float(g.get("w", 2.72)),
                "height": float(g.get("h", 2.72)),
            }
        )
    if gates:
        data["track_gates"] = gates
        from simulator.course_path import rebuild_course_path

        rebuild_course_path(data)
    return len(gates)
