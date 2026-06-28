import math
import unittest

import numpy as np

from simulator import camera_model


class TestCameraModel(unittest.TestCase):
    def test_range_from_width(self):
        focal = 27.0 * 10.0 / 2.72
        r = camera_model.range_from_width(27.0, 2.72, focal_px=focal)
        self.assertAlmostEqual(r, 10.0, places=1)

    def test_interaction_matrix_shape(self):
        L = camera_model.interaction_matrix(320.0, 180.0, 8.0)
        self.assertEqual(L.shape, (2, 6))

    def test_ibvs_reduces_centered_error(self):
        twist = camera_model.ibvs_twist(340.0, 190.0, 10.0, gain=1.0)
        yaw, _, _ = camera_model.twist_to_body_rates(twist)
        self.assertLess(yaw, 0.0)
        self.assertGreater(np.linalg.norm(twist), 0.01)

    def test_blend_falls_back_without_depth(self):
        yaw, alt = camera_model.blend_jacobian_p(
            0.5, -0.2, 400.0, 200.0, 0.0, 0.7, 6.0
        )
        self.assertAlmostEqual(yaw, 0.35)
        self.assertAlmostEqual(alt, -1.2)

    def test_blend_uses_jacobian_with_confidence(self):
        yaw_p, alt_p = camera_model.blend_jacobian_p(
            0.3, 0.1, 350.0, 200.0, 0.0, 0.7, 6.0
        )
        yaw_j, alt_j = camera_model.blend_jacobian_p(
            0.3, 0.1, 350.0, 200.0, 0.8, 0.7, 6.0, blend=1.0
        )
        self.assertNotAlmostEqual(yaw_p, yaw_j, places=2)

    def test_pixel_bearing_at_center(self):
        b, e = camera_model.pixel_bearing(camera_model.CX, camera_model.CY)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(e, 0.0)

    def test_pixel_bearing_offset(self):
        b, _ = camera_model.pixel_bearing(camera_model.CX + 32.0, camera_model.CY)
        self.assertAlmostEqual(b, math.atan2(32.0, camera_model.FX), places=5)

    def test_adaptive_blend_stronger_at_large_offset(self):
        gain = 1.0
        nx_small, u_small = 0.05, camera_model.CX + 0.05 * (camera_model.IMG_W / 2.0)
        nx_large, u_large = 0.4, camera_model.CX + 0.4 * (camera_model.IMG_W / 2.0)
        y_p_small = gain * nx_small
        y_j_small, _ = camera_model.blend_jacobian_p(
            nx_small, 0.0, u_small, camera_model.CY, 0.8, gain, 6.0, blend=0.25
        )
        y_p_large = gain * nx_large
        y_j_large, _ = camera_model.blend_jacobian_p(
            nx_large, 0.0, u_large, camera_model.CY, 0.8, gain, 6.0, blend=0.25
        )
        self.assertGreater(abs(y_j_large - y_p_large), abs(y_j_small - y_p_small))

    def test_range_scales_yaw_when_close(self):
        nx, ny, u = 0.3, 0.1, 400.0
        y_far, _ = camera_model.blend_jacobian_p(
            nx, ny, u, 180.0, 0.0, 1.0, 6.0, range_m=20.0
        )
        y_close, _ = camera_model.blend_jacobian_p(
            nx, ny, u, 180.0, 0.0, 1.0, 6.0, range_m=3.0
        )
        self.assertLess(abs(y_close), abs(y_far))


if __name__ == "__main__":
    unittest.main()
