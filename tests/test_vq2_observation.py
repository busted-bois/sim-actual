"""Tests for the VQ2-legal observation.

The contract under test is a hard competition constraint, not a style choice:
under the VQ2 event block the simulator withholds ODOMETRY, ATTITUDE and
LOCAL_POSITION_NED even when they are explicitly requested, and the track
burst carries no gate poses (measured 2026-08-01: track_info=0 while
race_status streamed at 4.4 Hz). So the observation may be built ONLY from

  * YOLO+PnP gate estimates for the ONE gate the detector currently reports,
  * gyro (present, 114 Hz) and an AHRS gravity direction,
  * our own last action,
  * the gate index from race_status.

Anything derived from world position, world velocity or a gate map is
untestable in the qualifier and is therefore rejected here by construction.
"""

import inspect
import math
import unittest

import numpy as np

from rl.core import vq2_observation as vo


def _est(
    gate_pos_body=(10.0, 0.0, 0.0),
    normal_body=(-1.0, 0.0, 0.0),
    conf=0.9,
    n_visible=8,
    reproj_px=0.5,
    method="ippe-yolo8",
):
    """A vision estimate shaped exactly like simulator.gate_pnp._result_dict."""
    gb = np.asarray(gate_pos_body, dtype=float)
    horiz = float(np.hypot(gb[0], gb[1]))
    return {
        "gate_pos_body": gb,
        "normal_body": np.asarray(normal_body, dtype=float),
        "range_m": float(np.linalg.norm(gb)),
        "yaw_bearing": float(np.arctan2(gb[1], gb[0])),
        "pitch_bearing": float(np.arctan2(-gb[2], horiz)),
        "reproj_px": float(reproj_px),
        "n_visible": int(n_visible),
        "method": method,
        "conf": float(conf),
    }


LEVEL_GRAVITY = (0.0, 0.0, 1.0)
NO_ACTION = (0.0, 0.0, 0.0, 0.0)


class LayoutTests(unittest.TestCase):
    def test_slices_tile_the_vector_exactly(self):
        """Every index is claimed by exactly one field — no gaps, no overlap."""
        covered = []
        for name, sl in vo.OBS_LAYOUT.items():
            self.assertIsInstance(sl, slice, name)
            covered.extend(range(sl.start, sl.stop))
        self.assertEqual(
            sorted(covered),
            list(range(vo.OBS_DIM)),
            "OBS_LAYOUT must tile [0, OBS_DIM) exactly once",
        )

    def test_builder_cannot_accept_blocked_telemetry(self):
        """Structural guard: the signature must not offer a way to sneak in
        world pose or a gate map. This is the bug that made the previous
        pipeline undeployable."""
        params = set(inspect.signature(vo.GateFeatureTracker.update).parameters)
        forbidden = {
            "p",
            "pos",
            "position",
            "pos_ned",
            "v",
            "vel",
            "v_world",
            "velocity",
            "q",
            "quat",
            "gate_map",
            "gates",
            "odometry",
        }
        self.assertEqual(
            params & forbidden,
            set(),
            f"update() exposes blocked telemetry: {sorted(params & forbidden)}",
        )


class PerfectDetectionTests(unittest.TestCase):
    def setUp(self):
        self.tr = vo.GateFeatureTracker(n_gates=17)

    def test_gate_dead_ahead(self):
        obs = self.tr.update(
            0.0, _est((10.0, 0.0, 0.0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )
        d = obs[vo.OBS_LAYOUT["gate_dir_body"]]
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 1.0)
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["staleness"]][0], 0.0)

    def test_gate_to_the_right_and_above(self):
        # body frame is FRD: +y right, +z DOWN, so "above" is negative z.
        obs = self.tr.update(
            0.0, _est((10.0, 5.0, -2.0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )
        d = obs[vo.OBS_LAYOUT["gate_dir_body"]]
        self.assertGreater(d[1], 0.0, "gate on the right -> +y")
        self.assertLess(d[2], 0.0, "gate above -> -z in FRD")
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=6)

    def test_range_is_monotone_and_bounded(self):
        near = self.tr.update(
            0.0, _est((2.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )[vo.OBS_LAYOUT["log_range"]][0]
        tr2 = vo.GateFeatureTracker(n_gates=17)
        far = tr2.update(
            0.0, _est((30.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )[vo.OBS_LAYOUT["log_range"]][0]
        self.assertLess(near, far, "log_range must increase with distance")
        for v in (near, far):
            self.assertGreaterEqual(v, -3.0)
            self.assertLessEqual(v, 3.0)

    def test_apparent_size_is_independent_of_the_metric_assumption(self):
        """Angular size comes straight from geometry, so it must shrink with
        range even though absolute PnP scale rests on an unverified gate width."""
        a = self.tr.update(
            0.0, _est((5.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )[vo.OBS_LAYOUT["apparent_size"]][0]
        tr2 = vo.GateFeatureTracker(n_gates=17)
        b = tr2.update(0.0, _est((20.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)[
            vo.OBS_LAYOUT["apparent_size"]
        ][0]
        self.assertGreater(a, b)


class MeasuredAttitudeTests(unittest.TestCase):
    """Ties the gravity channel to real numbers off the live sim."""

    def test_probe_measured_start_attitude(self):
        # 2026-08-01 VQ2 probe, drone on the pad:
        #   accel=(-2.999,-0.002,-9.340), |a|=9.8097 -> 17.80 deg nose-down.
        ax, ay, az = -2.999, -0.002, -9.340
        n = math.sqrt(ax * ax + ay * ay + az * az)
        gravity_body = (-ax / n, -ay / n, -az / n)
        tr = vo.GateFeatureTracker(n_gates=17)
        obs = tr.update(0.0, None, (0, 0, 0), gravity_body, NO_ACTION, 0)
        g = obs[vo.OBS_LAYOUT["gravity_body"]]
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, places=5)
        tilt = math.degrees(math.atan2(math.hypot(g[0], g[1]), g[2]))
        self.assertAlmostEqual(tilt, 17.80, places=1)
        self.assertGreater(g[0], 0.0, "nose-down puts gravity forward in body x")


class DropoutTests(unittest.TestCase):
    """~18% of frames carry no detection and YOLO runs at 12-16 Hz against a
    30-50 Hz control loop, so dropout runs span several ticks. The observation
    must degrade gracefully rather than punch a hole in the policy input."""

    def setUp(self):
        self.tr = vo.GateFeatureTracker(n_gates=17)
        self.tr.update(
            0.0, _est((10.0, 1.0, 0.0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )

    def test_missing_detection_holds_last_direction_and_flags_it(self):
        held = self.tr.update(0.05, None, (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertAlmostEqual(held[vo.OBS_LAYOUT["detected"]][0], 0.0)
        self.assertGreater(held[vo.OBS_LAYOUT["staleness"]][0], 0.0)
        d = held[vo.OBS_LAYOUT["gate_dir_body"]]
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=6)

    def test_staleness_saturates_and_never_diverges(self):
        t = 0.0
        for _ in range(400):
            t += 0.02
            obs = self.tr.update(t, None, (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
            self.assertTrue(np.all(np.isfinite(obs)))
        self.assertLessEqual(obs[vo.OBS_LAYOUT["staleness"]][0], 1.0)

    def test_long_dropout_decays_confidence_to_zero(self):
        t = 0.0
        for _ in range(200):
            t += 0.02
            obs = self.tr.update(t, None, (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["conf"]][0], 0.0, places=3)

    def test_no_detection_ever_still_produces_a_finite_vector(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        obs = tr.update(0.0, None, (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 0.0)


class UntrustworthyDetectionTests(unittest.TestCase):
    """gate_pnp's edge-pair fallback reports reproj_px=0.0 with n_visible=2 and
    a FABRICATED plane normal, so a naive reproj filter ranks the least certain
    method as the best. The observation must not launder that into a confident
    normal."""

    def test_edge_pair_normal_is_suppressed(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        obs = tr.update(
            0.0,
            _est(
                normal_body=(-1.0, 0.0, 0.0),
                n_visible=2,
                reproj_px=0.0,
                method="edge-pair",
            ),
            (0, 0, 0),
            LEVEL_GRAVITY,
            NO_ACTION,
            0,
        )
        nrm = obs[vo.OBS_LAYOUT["gate_normal_body"]]
        np.testing.assert_allclose(nrm, [0.0, 0.0, 0.0], atol=1e-9)
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["normal_valid"]][0], 0.0)

    def test_full_keypoint_normal_is_kept(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        obs = tr.update(
            0.0,
            _est(n_visible=8, reproj_px=0.5),
            (0, 0, 0),
            LEVEL_GRAVITY,
            NO_ACTION,
            0,
        )
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["normal_valid"]][0], 1.0)
        self.assertAlmostEqual(
            float(np.linalg.norm(obs[vo.OBS_LAYOUT["gate_normal_body"]])),
            1.0,
            places=6,
        )

    def test_nan_estimate_is_rejected_not_propagated(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(0.0, _est((10.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        obs = tr.update(
            0.05, _est((float("nan"), 0.0, 0.0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 0.0)

    def test_gate_behind_us_is_rejected(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        obs = tr.update(
            0.0, _est((-5.0, 0.0, 0.0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0
        )
        self.assertAlmostEqual(obs[vo.OBS_LAYOUT["detected"]][0], 0.0)


class TemporalDerivativeTests(unittest.TestCase):
    """Velocity is not observable in VQ2 (measured estimator RMSE 28.9 m/s), so
    the policy gets the honest measured rates of the vision features instead of
    a fabricated velocity vector."""

    def test_closing_on_a_gate_gives_negative_range_rate(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(0.0, _est((10.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        obs = tr.update(0.1, _est((9.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertLess(obs[vo.OBS_LAYOUT["log_range_rate"]][0], 0.0)

    def test_receding_gate_gives_positive_range_rate(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(0.0, _est((9.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        obs = tr.update(0.1, _est((10.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertGreater(obs[vo.OBS_LAYOUT["log_range_rate"]][0], 0.0)

    def test_zero_dt_does_not_divide_by_zero(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(1.0, _est((10.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        obs = tr.update(1.0, _est((9.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_gate_switch_does_not_emit_a_huge_rate_spike(self):
        """A detector re-lock onto a different gate is a discontinuity, not
        motion. Scoring it as motion is exactly what produced the previous
        attempt's mean progress of -42 while it was flying forward at +26."""
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(0.0, _est((4.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        obs = tr.update(
            0.05, _est((28.0, 0, 0)), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 1
        )
        self.assertLessEqual(abs(obs[vo.OBS_LAYOUT["log_range_rate"]][0]), 1.0)


class ProgressTests(unittest.TestCase):
    def test_gate_index_is_normalized_over_the_course(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        first = tr.update(0.0, _est(), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        last = tr.update(1.0, _est(), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 16)
        self.assertAlmostEqual(first[vo.OBS_LAYOUT["gate_idx"]][0], 0.0)
        self.assertAlmostEqual(last[vo.OBS_LAYOUT["gate_idx"]][0], 1.0, places=2)

    def test_time_since_gate_resets_on_advance(self):
        tr = vo.GateFeatureTracker(n_gates=17)
        tr.update(0.0, _est(), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        stale = tr.update(5.0, _est(), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 0)
        self.assertGreater(stale[vo.OBS_LAYOUT["since_gate"]][0], 0.0)
        fresh = tr.update(5.1, _est(), (0, 0, 0), LEVEL_GRAVITY, NO_ACTION, 1)
        self.assertLess(
            fresh[vo.OBS_LAYOUT["since_gate"]][0],
            stale[vo.OBS_LAYOUT["since_gate"]][0],
        )


class BoundsTests(unittest.TestCase):
    def test_every_field_stays_in_range_under_adversarial_input(self):
        rng = np.random.default_rng(0)
        tr = vo.GateFeatureTracker(n_gates=17)
        t = 0.0
        for i in range(2000):
            t += float(rng.uniform(0.0, 0.2))
            if i % 5 == 0:
                est = None
            else:
                est = _est(
                    gate_pos_body=rng.uniform(-60, 60, 3),
                    normal_body=rng.uniform(-1, 1, 3),
                    conf=float(rng.uniform(0, 1)),
                    n_visible=int(rng.integers(0, 9)),
                    reproj_px=float(rng.uniform(0, 40)),
                )
            obs = tr.update(
                t,
                est,
                rng.uniform(-20, 20, 3),
                rng.uniform(-1, 1, 3),
                rng.uniform(-1, 1, 4),
                int(rng.integers(0, 17)),
            )
            self.assertEqual(obs.shape, (vo.OBS_DIM,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(obs)), f"non-finite at i={i}")
            self.assertTrue(
                np.all(np.abs(obs) <= vo.OBS_ABS_MAX + 1e-5),
                f"out of bounds at i={i}: {obs[np.abs(obs) > vo.OBS_ABS_MAX]}",
            )


if __name__ == "__main__":
    unittest.main()
