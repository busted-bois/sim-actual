"""Analyze tracking CSV logs from live FlightSim runs (no sim required)."""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _analyze(path: Path) -> dict:
    rows = _load_rows(path)
    healthy = [r for r in rows if r.get("healthy") == "True"]
    if not healthy:
        return {"path": str(path), "rows": len(rows), "healthy_rows": 0}

    def col(name):
        return [float(r[name]) for r in healthy]

    z = col("z")
    vz = col("vz")
    yaw = col("yaw")
    imu = [int(r["imu_samples"]) for r in healthy]

    return {
        "path": str(path),
        "rows": len(rows),
        "healthy_rows": len(healthy),
        "z_min": min(z),
        "z_max": max(z),
        "z_mean": statistics.mean(z),
        "vz_std": statistics.pstdev(vz) if len(vz) > 1 else 0.0,
        "vz_max_abs": max(abs(v) for v in vz),
        "yaw_span_rad": max(yaw) - min(yaw),
        "imu_samples_max": max(imu),
        "altitude_stable": max(z) - min(z) < 8.0 and statistics.pstdev(z) < 3.0,
    }


def _print_report(results: list[dict]) -> int:
    if not results:
        print("No tracking logs found in logs/")
        return 1

    ok = True
    for r in results:
        print(f"\n=== {Path(r['path']).name} ===")
        if r["healthy_rows"] == 0:
            print("  No healthy tracking rows")
            ok = False
            continue
        print(f"  rows: {r['rows']} ({r['healthy_rows']} healthy)")
        print(f"  z range: {r['z_min']:.2f} .. {r['z_max']:.2f} (mean {r['z_mean']:.2f})")
        print(f"  vz std: {r['vz_std']:.2f}, max |vz|: {r['vz_max_abs']:.2f}")
        print(f"  yaw span: {r['yaw_span_rad']:.2f} rad")
        print(f"  imu samples peak: {r['imu_samples_max']}")
        stable = r["altitude_stable"]
        print(f"  altitude stable: {'YES' if stable else 'NO'}")
        if not stable:
            ok = False

    print("\nSummary:", "PASS" if ok else "NEEDS TUNING (see flight_config.py altitude/tracking blend)")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tracking CSV logs from live runs")
    parser.add_argument(
        "paths",
        nargs="*",
        help="CSV files or directories (default: logs/tracking_state_*.csv)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.paths:
        for raw in args.paths:
            p = Path(raw)
            if p.is_dir():
                paths.extend(Path(x) for x in glob.glob(str(p / "tracking_state_*.csv")))
            else:
                paths.append(p)
    else:
        paths = [Path(p) for p in glob.glob("logs/tracking_state_*.csv")]

    results = [_analyze(p) for p in sorted(paths)]
    return _print_report(results)


if __name__ == "__main__":
    sys.exit(main())
