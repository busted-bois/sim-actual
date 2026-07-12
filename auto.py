"""Entry point for make auto — sets AUTO_FLIGHT before main."""

import os
import runpy

os.environ["AUTO_FLIGHT"] = "1"
# Course speed is reduced to 1.5 m/s for reliable gate passage; spawn is ~15m
# before gate 0, so allow more time than the 15s default before retrying.
os.environ.setdefault("GATE1_TIMEOUT_S", "30")
os.environ.setdefault("GATE_PROGRESS_TIMEOUT_S", "25")
runpy.run_path("main.py", run_name="__main__")
