"""
Tests for the gate navigator state machine. Timing is driven explicitly via the
``now`` argument so the tests are deterministic.

    uv run python -m pytest tests/test_navigation.py
    uv run python tests/test_navigation.py        # standalone, no pytest
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import DEFAULT_CONFIG  # noqa: E402
from simulator.navigation import GateNavigator, Phase  # noqa: E402
from simulator.vision_processing import Detection, FrameAnalysis  # noqa: E402


def _cfg():
    return copy.deepcopy(DEFAULT_CONFIG)


def _frame(gate=None, path=None, t=0.0):
    return FrameAnalysis(
        frame_id=0,
        width=640,
        height=480,
        gate=gate or Detection(found=False),
        path=path or Detection(found=False),
        timestamp=t,
    )


def test_searches_when_nothing_seen():
    nav = GateNavigator(_cfg())
    cmd = nav.compute(_frame(), now=0.0)
    assert cmd.phase == Phase.SEARCHING
    assert cmd.yaw_rate != 0.0  # scanning
    assert not cmd.complete


def test_approaches_and_yaws_toward_offcenter_gate():
    nav = GateNavigator(_cfg())
    gate = Detection(found=True, cx_norm=0.5, cy_norm=0.0, area_frac=0.02)
    cmd = nav.compute(_frame(gate=gate), now=0.0)
    assert cmd.phase == Phase.APPROACHING
    assert cmd.yaw_rate > 0  # gate to the right -> turn right
    assert cmd.vx > 0


def test_passes_gate_when_big_and_centered():
    nav = GateNavigator(_cfg())
    gate = Detection(found=True, cx_norm=0.05, cy_norm=0.0, area_frac=0.2)
    cmd = nav.compute(_frame(gate=gate), now=0.0)
    assert cmd.phase == Phase.PASSING
    assert cmd.gates_passed == 1
    assert cmd.vx > 0


def test_does_not_pass_big_but_offcenter_gate():
    nav = GateNavigator(_cfg())
    gate = Detection(found=True, cx_norm=0.6, cy_norm=0.0, area_frac=0.2)
    cmd = nav.compute(_frame(gate=gate), now=0.0)
    assert cmd.phase == Phase.APPROACHING
    assert cmd.gates_passed == 0


def test_follows_path_when_no_gate():
    nav = GateNavigator(_cfg())
    path = Detection(found=True, cx_norm=-0.4, cy_norm=0.0, area_frac=0.05)
    cmd = nav.compute(_frame(path=path), now=0.0)
    assert cmd.phase == Phase.FOLLOWING_PATH
    assert cmd.yaw_rate < 0  # path to the left -> turn left


def test_end_of_course_after_losing_everything():
    nav = GateNavigator(_cfg())
    gate = Detection(found=True, cx_norm=0.0, cy_norm=0.0, area_frac=0.02)
    nav.compute(_frame(gate=gate), now=0.0)  # see a gate at t=0
    end_s = DEFAULT_CONFIG["safety"]["end_of_course_seconds"]
    cmd = nav.compute(_frame(), now=end_s + 0.5)  # nothing for > end_of_course_seconds
    assert cmd.complete
    assert cmd.phase == Phase.COMPLETE
    assert cmd.vx == 0.0 and cmd.yaw_rate == 0.0  # safe stop / hover


def test_no_premature_completion_before_first_detection():
    nav = GateNavigator(_cfg())
    nav.compute(_frame(), now=0.0)
    cmd = nav.compute(_frame(), now=999.0)  # long time, but we never saw anything
    assert not cmd.complete
    assert cmd.phase == Phase.SEARCHING


def test_max_run_seconds_hard_cap():
    cfg = _cfg()
    cfg["safety"]["max_run_seconds"] = 5
    nav = GateNavigator(cfg)
    gate = Detection(found=True, cx_norm=0.0, cy_norm=0.0, area_frac=0.02)
    nav.compute(_frame(gate=gate), now=0.0)
    cmd = nav.compute(
        _frame(gate=gate), now=6.0
    )  # still seeing a gate, but past the cap
    assert cmd.complete


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
