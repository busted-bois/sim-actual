"""Entry point for make auto — sets AUTO_FLIGHT before main."""

import os
import runpy

os.environ["AUTO_FLIGHT"] = "1"
runpy.run_path("main.py", run_name="__main__")
