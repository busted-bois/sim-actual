"""Unit tests for the pure residual-RL core (no sim, no gym, no torch)."""

import math

import numpy as np
import pytest

from simulator import rl_core as rc


# ---------------------------------------------------------------- action mapping
def test_action_to_residual_bounds():
    lo = rc.action_to_residual([-1, -1, -1])
    hi = rc.action_to_residual([1, 1, 1])
    assert lo.speed_mult == pytest.approx(rc.SPEED_MULT_MIN)
    assert hi.speed_mult == pytest.approx(rc.SPEED_MULT_MAX)
    assert lo.lookahead_m == pytest.approx(rc.LOOKAHEAD_MIN_M)
    assert hi.lookahead_m == pytest.approx(rc.LOOKAHEAD_MAX_M)
    assert lo.lateral_offset_m == pytest.approx(-rc.LATERAL_OFFSET_MAX_M)
    assert hi.lateral_offset_m == pytest.approx(rc.LATERAL_OFFSET_MAX_M)


def test_action_clipped_when_out_of_range():
    a = rc.action_to_residual([5.0, -9.0, 100.0])
    assert a.speed_mult == pytest.approx(rc.SPEED_MULT_MAX)
    assert a.lookahead_m == pytest.approx(rc.LOOKAHEAD_MIN_M)
    assert a.lateral_offset_m == pytest.approx(rc.LATERAL_OFFSET_MAX_M)


def test_identity_residual_midpoint_roundtrip():
    res = rc.GuidanceResidual(
        speed_mult=(rc.SPEED_MULT_MIN + rc.SPEED_MULT_MAX) / 2,
        lookahead_m=(rc.LOOKAHEAD_MIN_M + rc.LOOKAHEAD_MAX_M) / 2,
        lateral_offset_m=0.0,
    )
    a = rc.residual_to_norm_action(res)
    assert np.allclose(a, [0.0, 0.0, 0.0], atol=1e-6)
    back = rc.action_to_residual(a)
    assert back.speed_mult == pytest.approx(res.speed_mult)
    assert back.lookahead_m == pytest.approx(res.lookahead_m)
    assert back.lateral_offset_m == pytest.approx(res.lateral_offset_m)


def test_action_dim_validation():
    with pytest.raises(ValueError):
        rc.action_to_residual([0.0, 0.0])


# ---------------------------------------------------------------- geometry
def _gate(x, y, z, w=2.72):
    return {"position_ned": (x, y, z), "width": w, "height": w}


def test_approach_dir_points_to_next_gate():
    gates = [_gate(-10, 0, 0), _gate(-20, 0, 0)]
    appr = rc.approach_dir(gates, 0)
    assert appr[0] == pytest.approx(-1.0, abs=1e-6)
    assert abs(appr[1]) < 1e-6


def test_gate_plane_progress_sign():
    gates = [_gate(-10, 0, 0), _gate(-20, 0, 0)]
    appr = rc.approach_dir(gates, 0)
    before = rc.gate_plane_progress((-5, 0, 0), gates[0], appr)  # not yet at gate
    after = rc.gate_plane_progress((-15, 0, 0), gates[0], appr)  # past gate
    assert before < 0 < after


def test_lateral_miss_distance_and_aperture():
    gates = [_gate(-10, 0, 0), _gate(-20, 0, 0)]
    appr = rc.approach_dir(gates, 0)
    # 1 m to the side of the through-axis
    assert rc.lateral_miss_distance((-10, 1.0, 0), gates[0], appr) == pytest.approx(
        1.0, abs=1e-6
    )
    assert rc.gate_aperture_radius(gates[0]) == pytest.approx(1.36)


def test_ned_vec_to_body_yaw():
    # facing north (yaw 0): a vector due east is to the right
    fwd, right, _ = rc.ned_vec_to_body(0.0, 1.0, 0.0, 0.0)
    assert fwd == pytest.approx(0.0, abs=1e-6)
    assert right == pytest.approx(1.0, abs=1e-6)
    # facing east (yaw pi/2): vector due east is straight ahead
    fwd, right, _ = rc.ned_vec_to_body(0.0, 1.0, 0.0, math.pi / 2)
    assert fwd == pytest.approx(1.0, abs=1e-6)
    assert right == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- observation
def test_observation_shape_and_finite():
    gates = [_gate(-10, 0, 0), _gate(-20, 0, 1), _gate(-30, 0, 2), _gate(-40, 0, 3)]
    obs = rc.build_observation(
        pos=(0, 0, 0),
        vel_ned=(1, 0, 0),
        roll=0.0,
        pitch=0.0,
        yaw=math.pi,  # facing -X (down-track)
        gates=gates,
        active_idx=0,
        residual=rc.identity_residual(),
        last_action=np.zeros(3),
    )
    assert obs.shape == (rc.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_observation_gate_ahead_is_forward():
    # Drone at origin facing -X; gate at -10 X is straight ahead -> positive body-forward.
    gates = [_gate(-10, 0, 0), _gate(-20, 0, 0), _gate(-30, 0, 0)]
    obs = rc.build_observation(
        (0, 0, 0),
        (0, 0, 0),
        0.0,
        0.0,
        math.pi,
        gates,
        0,
        rc.identity_residual(),
        np.zeros(3),
    )
    fwd = obs[14]  # first gate body-forward offset (scaled)
    assert fwd > 0


def test_observation_zero_pads_past_track():
    gates = [_gate(-10, 0, 0)]
    obs = rc.build_observation(
        (0, 0, 0),
        (0, 0, 0),
        0.0,
        0.0,
        math.pi,
        gates,
        0,
        rc.identity_residual(),
        np.zeros(3),
    )
    # gates 2 and 3 don't exist -> their blocks are zero
    assert np.allclose(obs[14 + rc._PER_GATE :], 0.0)


# ---------------------------------------------------------------- reward
def test_step_reward_progress_positive():
    r = rc.step_reward(
        prev_dist=10.0, cur_dist=8.0, gate_passed=False, action=[0, 0, 0]
    )
    assert r == pytest.approx(2.0 * rc.PROGRESS_W)


def test_step_reward_gate_bonus_and_action_penalty():
    r = rc.step_reward(5.0, 5.0, gate_passed=True, action=[1, 0, 0])
    assert r == pytest.approx(rc.GATE_BONUS - rc.ACTION_PEN)


def test_finish_reward_completion_gated_time():
    assert rc.finish_reward(20.0, time_weight=0.0) == pytest.approx(rc.FINISH_BONUS)
    faster = rc.finish_reward(10.0, time_weight=1.0)
    slower = rc.finish_reward(30.0, time_weight=1.0)
    assert faster > slower  # less time -> more reward when weighted


# ---------------------------------------------------------------- curriculum
def test_reverse_curriculum_start_gate_endpoints():
    assert (
        rc.reverse_curriculum_start_gate(0.0, 6, hardest_gate=2) == 2
    )  # start at hardest
    assert (
        rc.reverse_curriculum_start_gate(1.0, 6, hardest_gate=2) == 0
    )  # expand to start
    assert rc.reverse_curriculum_start_gate(0.5, 6, hardest_gate=2) == 1


def test_gate_normal_round1_is_minus_x():
    from simulator.config import ROUND1_GATES

    for g in ROUND1_GATES:
        n = rc.gate_normal(g)
        assert n[0] == pytest.approx(-1.0, abs=1e-3)
        assert abs(n[1]) < 1e-3
        assert abs(n[2]) < 1e-3


def test_gate_normal_sign_follows_travel():
    g = {"orientation_ned": (0.7071067, 0.0, 0.0, 0.7071067)}
    assert rc.gate_normal(g, travel_hint=(-1, 0, 0))[0] < 0  # points the way we travel
    assert rc.gate_normal(g, travel_hint=(1, 0, 0))[0] > 0  # flips to match hint


def test_is_flyaway():
    target = (-46.9, -2.5, 5.07)
    assert not rc.is_flyaway((-45.0, -2.0, 5.0), target)  # right at the gate
    assert not rc.is_flyaway((-20.0, 0.0, 0.0), target)  # one gate back, still sane
    assert rc.is_flyaway((-475.0, -750.0, 1845.0), target)  # km away -> runaway


def test_frame_looks_corrupted():
    assert not rc.frame_looks_corrupted(None)
    assert not rc.frame_looks_corrupted((0.0, 0.0, 0.0))  # on the pad
    assert not rc.frame_looks_corrupted((-160.0, -4.0, 26.0))  # last gate, sane
    assert rc.frame_looks_corrupted((-475.0, -750.0, 1845.0))  # corrupted


def test_time_weight_schedule():
    assert rc.time_weight_schedule(0.0) == 0.0
    assert rc.time_weight_schedule(0.5, reliability_frac=0.5) == 0.0
    assert rc.time_weight_schedule(
        1.0, reliability_frac=0.5, max_weight=1.0
    ) == pytest.approx(1.0)
    assert 0.0 < rc.time_weight_schedule(0.75, reliability_frac=0.5) < 1.0
