from simulator.tracking.local_tracker import LocalTracker


def _armed_data(time_us=10_000, sim_time_ns=10_000_000):
    return {
        "armed": True,
        "highres_imu": {
            "xacc": 0.0,
            "yacc": 0.0,
            "zacc": -9.81,
            "xgyro": 0.0,
            "ygyro": 0.0,
            "zgyro": 0.0,
            "time_boot_us": time_us,
        },
        "local_position_ned": {
            "x": 1.0,
            "y": 0.0,
            "z": -2.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "attitude": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    }


def test_tracker_sets_origin_on_arm():
    tracker = LocalTracker(log_csv=False)
    data = {"armed": True}
    tracker.tick(data)
    assert data["tracking_snapshot"].status == "tracking"


def test_tracker_blends_local_position():
    tracker = LocalTracker(log_csv=False)
    data = _armed_data()
    tracker.tick(data)
    tracker.tick(
        {**data, "highres_imu": {**data["highres_imu"], "time_boot_us": 20_000}}
    )
    snapshot = data["tracking_snapshot"]
    assert snapshot.x > 0.0
    assert snapshot.z < 0.0


def test_tracker_applies_vision_yaw_correction():
    tracker = LocalTracker(log_csv=False)
    data = _armed_data()
    data["camera"] = {"sim_time_ns": 10_000_000}
    data["gate_target"] = {"detected": True, "nx": 0.4, "ny": 0.0, "r_frac": 0.05}
    tracker.tick(data)
    tracker.tick(
        {
            **data,
            "highres_imu": {**data["highres_imu"], "time_boot_us": 20_000},
            "camera": {"sim_time_ns": 20_000_000},
        }
    )
    assert data["tracking_snapshot"].yaw != 0.0
