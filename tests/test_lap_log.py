import json
import os
import tempfile
import unittest
from unittest.mock import patch

from simulator import lap_log


class LapLogTests(unittest.TestCase):
    def test_append_lap_writes_jsonl_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            laps = os.path.join(tmp, "auto_laps.jsonl")
            best = os.path.join(tmp, "best_lap.txt")
            with (
                patch.object(lap_log, "LAPS_JSONL_PATH", laps),
                patch.object(lap_log, "BEST_LAP_TXT_PATH", best),
            ):
                lap_log.append_lap(2, 45.3, 42.1)
                lap_log.append_lap(3, 50.0, 42.1)

            with open(laps, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["attempt"], 2)
            self.assertEqual(lines[0]["lap_s"], 45.3)
            self.assertEqual(lines[1]["best_lap_s"], 42.1)

            with open(best, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("best=42.1s", text)
            self.assertIn("attempt=3", text)


if __name__ == "__main__":
    unittest.main()
