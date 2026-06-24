"""Benchmark: Jacobian vision blend vs main-branch P-control on nx/ny."""

from __future__ import annotations

import math
import unittest

from simulator import camera_model as cm

VISION_YAW_GAIN = math.radians(40)
VISION_VY_GAIN = 6.0


def main_branch_p(nx: float, ny: float) -> tuple[float, float]:
    return VISION_YAW_GAIN * nx, VISION_VY_GAIN * ny


class TestIbvsComparison(unittest.TestCase):
    """Documents why full IBVS pseudo-inverse underperforms P-control here."""

    def test_ibvs_pseudo_inverse_weakens_far_yaw(self):
        """At typical first-gate offset, IBVS blend cuts yaw ~35%+ vs main."""
        nx, ny, u, v, z, conf = 0.4, 0.1, 400.0, 185.0, 25.0, 0.55
        y_main, _ = main_branch_p(nx, ny)
        y_ibvs, _ = cm.blend_ibvs_pseudo(
            nx, ny, u, v, z, conf, VISION_YAW_GAIN, VISION_VY_GAIN, blend=0.6
        )
        self.assertLess(abs(y_ibvs), abs(y_main) * 0.75)

    def test_ibvs_pseudo_inverse_breaks_altitude(self):
        """IBVS altitude opposes P-control when gate is below center."""
        nx, ny, u, v, z, conf = 0.4, 0.1, 400.0, 185.0, 25.0, 0.55
        _, a_main = main_branch_p(nx, ny)
        _, a_ibvs = cm.blend_ibvs_pseudo(
            nx, ny, u, v, z, conf, VISION_YAW_GAIN, VISION_VY_GAIN, blend=0.6
        )
        self.assertGreater(a_main, 0.0)
        self.assertLess(a_ibvs, 0.0)

    def test_bearing_jacobian_yaw_only_altitude(self):
        """Jacobian blend adjusts yaw only; altitude stays on ny P-control."""
        nx, ny, u, v, conf = 0.4, 0.1, 448.0, 185.0, 0.55
        _, a_main = main_branch_p(nx, ny)
        _, a_j = cm.blend_jacobian_p(
            nx, ny, u, v, conf, VISION_YAW_GAIN, VISION_VY_GAIN, blend=0.5
        )
        self.assertAlmostEqual(a_main, a_j)

    def test_no_depth_matches_main(self):
        y_main, a_main = main_branch_p(0.3, -0.1)
        y_j, a_j = cm.blend_jacobian_p(
            0.3, -0.1, 380.0, 170.0, 0.0, VISION_YAW_GAIN, VISION_VY_GAIN
        )
        self.assertAlmostEqual(y_main, y_j)
        self.assertAlmostEqual(a_main, a_j)


if __name__ == "__main__":
    unittest.main()
