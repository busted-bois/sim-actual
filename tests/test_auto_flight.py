import unittest
from unittest.mock import patch

from simulator.auto_flight import auto_flight_enabled, is_ctrl_cancel


class AutoFlightTests(unittest.TestCase):
    def test_auto_flight_enabled(self):
        with patch.dict("os.environ", {"AUTO_FLIGHT": "1"}):
            self.assertTrue(auto_flight_enabled())
        with patch.dict("os.environ", {"AUTO_FLIGHT": "true"}):
            self.assertTrue(auto_flight_enabled())
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(auto_flight_enabled())

    def test_is_ctrl_cancel(self):
        self.assertTrue(is_ctrl_cancel(b"\x03"))  # Ctrl+C
        self.assertTrue(is_ctrl_cancel(b"\x01"))  # Ctrl+A
        self.assertTrue(is_ctrl_cancel(b"\x1a"))  # Ctrl+Z
        self.assertFalse(is_ctrl_cancel(b"a"))
        self.assertFalse(is_ctrl_cancel(b""))


if __name__ == "__main__":
    unittest.main()
