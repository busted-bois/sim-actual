import importlib

from aigp_pilot import course, raceconfig


def test_measured_config_is_default_round1_qualifier():
    cfg = raceconfig.MEASURED
    assert cfg.v_max == 3.0
    assert cfg.climb_max == 2.3
    assert cfg.alt_bias == 0.6
    assert len(course.GATES) == 6


def test_fly_imports():
    fly = importlib.import_module("fly")
    assert callable(fly.main)
