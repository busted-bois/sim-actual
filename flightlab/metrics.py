"""Flight metrics: std, peak-to-peak, overshoot, settle, PSD, zero-crossings."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(np.std(xs, ddof=0))


def mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.mean(xs))


def peak_to_peak(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.max(xs) - np.min(xs))


def overshoot_frac(series: Sequence[float], start: float, target: float) -> float:
    """Fractional overshoot past target relative to step size."""
    step = target - start
    if abs(step) < 1e-9:
        return 0.0
    if step > 0:
        peak = float(np.max(series))
        return max(0.0, (peak - target) / step)
    peak = float(np.min(series))
    return max(0.0, (target - peak) / abs(step))


def settle_time(
    t: Sequence[float],
    series: Sequence[float],
    target: float,
    band: float,
    after_t: float | None = None,
) -> float | None:
    """Seconds from after_t (or t[0]) until series stays in ±band of target."""
    if not t:
        return None
    t0 = after_t if after_t is not None else t[0]
    last_out = None
    for ti, yi in zip(t, series):
        if ti < t0:
            continue
        if abs(yi - target) > band:
            last_out = ti
    if last_out is None:
        # Never left band after t0 — settled immediately
        return 0.0
    # Find first time after last_out where we enter and stay
    # Simpler: time from t0 to when we last exited the band
    return float(last_out - t0)


def settle_time_enter(
    t: Sequence[float],
    series: Sequence[float],
    target: float,
    band: float,
    after_t: float,
    hold_s: float = 0.5,
) -> float | None:
    """Time from after_t until series enters ±band and stays for hold_s."""
    enter = None
    for ti, yi in zip(t, series):
        if ti < after_t:
            continue
        if abs(yi - target) <= band:
            if enter is None:
                enter = ti
            elif ti - enter >= hold_s:
                return float(enter - after_t)
        else:
            enter = None
    return None


def zero_crossings(xs: Sequence[float], center: float = 0.0) -> int:
    if len(xs) < 2:
        return 0
    count = 0
    prev = xs[0] - center
    for x in xs[1:]:
        cur = x - center
        if prev == 0:
            prev = cur
            continue
        if cur == 0:
            continue
        if (prev > 0) != (cur > 0):
            count += 1
        prev = cur
    return count


def psd_peak_hz(t: Sequence[float], xs: Sequence[float]) -> tuple[float, float]:
    """Return (peak_freq_hz, peak_power). Needs ≥16 samples."""
    if len(xs) < 16:
        return 0.0, 0.0
    t_arr = np.asarray(t, dtype=float)
    x_arr = np.asarray(xs, dtype=float) - np.mean(xs)
    dt = float(np.median(np.diff(t_arr)))
    if dt <= 0:
        return 0.0, 0.0
    n = len(x_arr)
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(x_arr)) ** 2
    # Ignore DC
    if len(spec) < 2:
        return 0.0, 0.0
    idx = int(np.argmax(spec[1:]) + 1)
    return float(freqs[idx]), float(spec[idx])


def secondary_bounce_ok(peaks: Sequence[float], ratio: float = 0.5) -> bool:
    """Each successive |peak| < ratio * previous |peak|."""
    if len(peaks) < 2:
        return True
    for a, b in zip(peaks, peaks[1:]):
        if abs(b) >= abs(a) * ratio - 1e-9:
            return False
    return True


def extract_error_peaks(
    t: Sequence[float], err: Sequence[float], after_t: float
) -> list[float]:
    """Local extrema of error after a step (for bounce check)."""
    peaks: list[float] = []
    xs = [(ti, ei) for ti, ei in zip(t, err) if ti >= after_t]
    if len(xs) < 3:
        return peaks
    for i in range(1, len(xs) - 1):
        _, e0 = xs[i - 1]
        _, e1 = xs[i]
        _, e2 = xs[i + 1]
        if (e1 >= e0 and e1 >= e2) or (e1 <= e0 and e1 <= e2):
            if abs(e1) > 0.05:  # ignore tiny noise peaks
                peaks.append(e1)
    return peaks


def alt_from_z(z: float) -> float:
    """NED z → altitude (up-positive)."""
    return -z


def deg(rad: float) -> float:
    return math.degrees(rad)
