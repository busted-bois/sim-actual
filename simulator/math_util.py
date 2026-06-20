"""Shared math helpers used across navigation, flight control, and tracking."""

from __future__ import annotations

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(error):
    while error > math.pi:
        error -= 2.0 * math.pi
    while error < -math.pi:
        error += 2.0 * math.pi
    return error
