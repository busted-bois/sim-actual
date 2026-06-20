from simulator.tracking.imu_propagator import (
    blend_attitude,
    propagate_position_velocity,
)


def test_propagate_integrates_roll_pitch_yaw_from_gyro():
    state = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "vx": 0.0,
        "vy": 0.0,
        "vz": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    result = propagate_position_velocity(
        state,
        accel_body=(0.0, 0.0, -9.81),
        gyro=(0.1, -0.2, 0.05),
        dt_s=0.01,
    )
    assert result["roll"] > 0.0
    assert result["pitch"] < 0.0
    assert result["yaw"] > 0.0


def test_blend_attitude_mixes_measured():
    state = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    blended = blend_attitude(state, 1.0, -1.0, 0.5)
    assert 0.0 < blended["roll"] < 1.0
    assert -1.0 < blended["pitch"] < 0.0
