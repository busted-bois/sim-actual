"""Frozen action and observation contracts shared by training and deployment."""

import unittest

import numpy as np

from rl.core import spec
from rl.core.observation import build_observation


class SpecContractTests(unittest.TestCase):
    def test_dimensions_layout_and_constants(self) -> None:
        self.assertEqual(spec.ACTION_DIM, 4)
        self.assertEqual(spec.OBS_DIM, 24)
        self.assertEqual(spec.GRAVITY, 9.81)
        self.assertEqual(spec.GATE_SIZE_M, 2.72)
        self.assertEqual(
            [spec.MAX_ROLL_RATE, spec.MAX_PITCH_RATE, spec.MAX_YAW_RATE],
            [0.6, 0.6, 0.6],
        )
        self.assertEqual(
            list(spec.OBS_LAYOUT),
            [
                "to_gate_body",
                "dist_to_gate",
                "gate_normal_body",
                "vel_body",
                "ang_vel",
                "gravity_body",
                "yaw_align",
                "to_next_gate_body",
                "dist_to_next_gate",
                "last_action",
            ],
        )

        indices = [
            index
            for value in spec.OBS_LAYOUT.values()
            for index in range(value.start, value.stop)
        ]
        self.assertEqual(indices, list(range(spec.OBS_DIM)))

    def test_action_scaling_contract(self) -> None:
        rng = np.random.default_rng(0)
        actions = rng.uniform(-1.0, 1.0, size=(128, spec.ACTION_DIM))
        for action in actions:
            np.testing.assert_allclose(
                spec.unscale_action(spec.scale_action(action)), action, atol=1e-12
            )

        normalized = np.array([-1.25, -0.5, 0.25, 1.2])
        np.testing.assert_array_equal(
            spec.scale_action(normalized), np.array([-0.6, -0.3, 0.15, 1.0])
        )
        np.testing.assert_array_equal(
            spec.unscale_action(spec.scale_action(normalized)),
            np.array([-1.0, -0.5, 0.25, 1.0]),
        )

    def test_quaternion_rotation_is_proper_orthogonal(self) -> None:
        rotation = spec.quat_to_R(
            np.array([0.73786479, -0.21081851, 0.42163702, 0.48529624])
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_golden_observation(self) -> None:
        gate_map = [
            {"pos": [5.0, 1.0, -2.0], "quat": [1.0, 0.0, 0.0, 0.0]},
            {
                "pos": [11.0, -2.0, -3.5],
                "quat": [0.9800665778412416, 0.0, 0.0, 0.19866933079506122],
            },
        ]
        observation = build_observation(
            p=np.array([1.0, -0.5, -1.25]),
            v_world=np.array([2.5, -1.0, 0.4]),
            q=np.array([0.9659258262890683, 0.0, 0.0, 0.25881904510252074]),
            ang_vel=np.array([0.12, -0.18, 0.24]),
            gate_map=gate_map,
            gate_idx=0,
            last_action=np.array([0.3, -0.2, 0.1]),
        )
        expected = np.array(
            [
                0.9715871214866638,
                -0.16161109507083893,
                -0.17291712760925293,
                0.4337337911128998,
                0.8660253882408142,
                -0.5,
                0.0,
                0.1665063500404358,
                -0.21160253882408142,
                0.03999999910593033,
                0.20000000298023224,
                -0.30000001192092896,
                0.4000000059604645,
                0.0,
                0.0,
                1.0,
                -0.052466414868831635,
                0.7635988593101501,
                -0.6080636978149414,
                -0.21719877421855927,
                1.0359175205230713,
                0.30000001192092896,
                -0.20000000298023224,
                0.10000000149011612,
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(observation, expected)


if __name__ == "__main__":
    unittest.main()
