"""Entry point for make auto — sets AUTO_FLIGHT before main."""

import os
import runpy

os.environ["AUTO_FLIGHT"] = "1"
os.environ.setdefault("AUTO_FLIGHT_DEBUG", "1")
os.environ.setdefault("AUTO_VISION_PREVIEW", "1")
os.environ.setdefault("AUTO_GO_VISION", "1")
runpy.run_path("main.py", run_name="__main__")
