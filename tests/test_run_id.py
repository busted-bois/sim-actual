"""Unit tests for the per-process run id (simulator/run_id).

A video and its flight CSVs used to stamp themselves independently, minutes
apart on a slow start. These lock in the shared key that replaced that guess.
"""

import re
import unittest

from simulator.run_id import RUN_ID


class RunIdTests(unittest.TestCase):
    def test_shape_matches_the_existing_stamp_convention(self):
        # Unchanged from what display and gp_pilot produced before, so the push
        # scripts' `_\d{8}_\d{6}$` regex and the .gitignore globs still match.
        self.assertRegex(RUN_ID, r"^\d{8}_\d{6}$")

    def test_is_stable_across_imports(self):
        # Every writer in the process must agree, so this is minted once at
        # import and never recomputed. Re-importing hands back the same object.
        import importlib

        again = importlib.import_module("simulator.run_id")
        self.assertIs(again.RUN_ID, RUN_ID)

    def test_is_filename_safe(self):
        # It is pasted straight into filenames on Windows and POSIX.
        self.assertIsNone(re.search(r'[<>:"/\\|?*\s]', RUN_ID))


if __name__ == "__main__":
    unittest.main()
