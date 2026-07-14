"""Thin pointer — run AndurilGP's main.py unchanged from the git submodule.

Same entry shape as their `python main.py`. No overnight AUTO_FLIGHT, no
rewrite of their controller/vision. Requires submodule init:

    git submodule update --init --recursive
"""

from __future__ import annotations

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
_GP = os.path.join(_REPO, "vendor", "AndurilGP")
_MAIN = os.path.join(_GP, "main.py")


def main() -> None:
    if not os.path.isfile(_MAIN):
        print(
            "ERROR: vendor/AndurilGP missing.\n"
            "  git submodule update --init --recursive",
            flush=True,
        )
        sys.exit(1)
    os.chdir(_GP)
    if _GP not in sys.path:
        sys.path.insert(0, _GP)
    print(f"[auto-gp] running {_MAIN} (cwd={_GP})", flush=True)
    runpy.run_path(_MAIN, run_name="__main__")


if __name__ == "__main__":
    main()
