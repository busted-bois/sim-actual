from simulator.gate_fusion import apply_pnp_to_state
from simulator.gate_pnp import order_corners, solve_gate_pnp


def test_order_corners_returns_four_points():
    points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    ordered = order_corners(points)
    assert len(ordered) == 4


def test_solve_gate_pnp_with_synthetic_corners():
    half = 120.0
    cx, cy = 320.0, 180.0
    corners = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]
    result = solve_gate_pnp(corners)
    assert result is not None
    assert result["range_m"] > 0.5


def test_apply_pnp_blends_position():
    gate = {
        "position_ned": (10.0, 0.0, -5.0),
        "orientation_ned": (1.0, 0.0, 0.0, 0.0),
    }
    state = {"x": 0.0, "y": 0.0, "z": -5.0, "yaw": 0.0}
    pnp = {"range_m": 8.0, "lateral_m": 0.0, "yaw_correction": 0.0}
    updated = apply_pnp_to_state(state, gate, pnp, blend=0.5)
    assert updated["x"] > state["x"]
