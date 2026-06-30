from __future__ import annotations

import struct
from dataclasses import dataclass

RACE_STATUS_ID = 1
_RACE_STATUS_FMT = "<BQqqIq"


@dataclass(frozen=True)
class RaceStatus:
    race_start_boot_time_ms: int
    race_finish_time_ns: int
    active_gate_index: int

    @property
    def finished(self) -> bool:
        return self.race_finish_time_ns >= 0


def parse_race_status(raw: bytes) -> RaceStatus:
    _, _, start, finish, active, _ = struct.unpack_from(_RACE_STATUS_FMT, raw)
    return RaceStatus(
        race_start_boot_time_ms=start,
        race_finish_time_ns=finish,
        active_gate_index=active,
    )
