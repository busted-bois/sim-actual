"""Optional preflight checks before connecting to FlightSim."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.preflight import run_preflight_checks


def main() -> int:
    return 0 if run_preflight_checks() else 1


if __name__ == "__main__":
    sys.exit(main())
