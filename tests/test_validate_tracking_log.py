import csv

from scripts.validate_tracking_log import _analyze, _print_report


def test_analyze_healthy_log(tmp_path):
    path = tmp_path / "tracking_state_test.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sim_time_ns",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "roll",
                "pitch",
                "yaw",
                "status",
                "healthy",
                "imu_samples",
            ],
        )
        writer.writeheader()
        for z in (-5.0, -5.1, -4.9, -5.2):
            writer.writerow(
                {
                    "sim_time_ns": 1,
                    "x": 0,
                    "y": 0,
                    "z": z,
                    "vx": 0,
                    "vy": 0,
                    "vz": 0.1,
                    "roll": 0,
                    "pitch": 0,
                    "yaw": 0,
                    "status": "tracking",
                    "healthy": "True",
                    "imu_samples": 3,
                }
            )

    result = _analyze(path)
    assert result["healthy_rows"] == 4
    assert result["altitude_stable"] is True


def test_print_report_empty():
    assert _print_report([]) == 1
