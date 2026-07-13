"""Test metrics: signs, latency, step response, jitter, PSD, stability."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

SIGNS_PATH = os.path.join(os.path.dirname(__file__), "signs.json")


def signs_measured() -> bool:
    """True once B0 has written a measured signs.json."""
    return os.path.isfile(SIGNS_PATH)


def load_signs(*, vq2: bool = False) -> dict[str, float]:
    # Fallback when B0 hasn't measured yet: the live-verified convention from
    # manual_controls (2026-07-09, odometry attitude) — all +1. The old
    # {-1,-1,-1} default made every axis positive feedback (drone nosed over
    # from the ramp spawn and flew into the gate base). EKF/estimated attitude
    # is truth-convention too (see rl/fly2_course.py EST_SIGNS note), so vq2
    # uses the same fallback.
    defaults = {"roll": 1.0, "pitch": 1.0, "yaw": 1.0}
    if os.path.isfile(SIGNS_PATH):
        with open(SIGNS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {
            "roll": float(data.get("roll", defaults["roll"])),
            "pitch": float(data.get("pitch", defaults["pitch"])),
            "yaw": float(data.get("yaw", defaults["yaw"])),
        }
    print(
        "[metrics] WARNING: signs UNMEASURED (no signs.json) — using +1,+1,+1 "
        "fallback. Run B0 to measure before trusting closed-loop tests.",
        flush=True,
    )
    return dict(defaults)


def save_signs(signs: dict[str, float]) -> None:
    with open(SIGNS_PATH, "w", encoding="utf-8") as f:
        json.dump(signs, f, indent=2)
    print(f"[metrics] wrote {SIGNS_PATH}: {signs}", flush=True)


def identify_sign_from_pulse(
    angles_before: float, angles_after: float, cmd_rate: float
) -> float:
    delta = angles_after - angles_before
    if abs(delta) < math.radians(0.5):
        return 1.0
    return 1.0 if (delta * cmd_rate) > 0 else -1.0


def imu_tilt_sign_check(
    roll: float, pitch: float, g_body: tuple[float, float, float] | None
) -> dict[str, str]:
    """Cross-check IMU gravity tilt vs quaternion euler."""
    out: dict[str, str] = {"roll": "n/a", "pitch": "n/a"}
    if g_body is None:
        return out
    gx, gy, gz = g_body
    imu_roll = math.atan2(gy, -gz)
    imu_pitch = math.atan2(-gx, math.hypot(gy, gz))
    roll_match = math.copysign(1.0, roll) == math.copysign(1.0, imu_roll)
    pitch_match = math.copysign(1.0, pitch) == math.copysign(1.0, imu_pitch)
    out["roll"] = "match" if roll_match else "INVERTED"
    out["pitch"] = "match" if pitch_match else "INVERTED"
    return out


def measure_latency_ms(
    times: list[float], cmds: list[float], rates: list[float], threshold: float = 0.05
) -> float | None:
    """Latency from command step to rate response crossing threshold fraction."""
    if len(times) < 5:
        return None
    t0_idx = None
    for i, c in enumerate(cmds):
        if abs(c) > 0.1:
            t0_idx = i
            break
    if t0_idx is None:
        return None
    target = threshold * abs(cmds[t0_idx])
    for j in range(t0_idx, len(rates)):
        if abs(rates[j]) >= target:
            return (times[j] - times[t0_idx]) * 1000.0
    return None


@dataclass
class StepMetrics:
    rise_s: float = float("inf")
    overshoot_pct: float = float("inf")
    settle_s: float = float("inf")
    damped: bool = False
    peaks: list[float] = field(default_factory=list)


def analyze_step_response(
    times: list[float],
    angles_deg: list[float],
    target_deg: float,
    *,
    rise_frac: float = 0.9,
    settle_band_deg: float = 1.0,
    settle_hold_s: float = 0.2,
) -> StepMetrics:
    m = StepMetrics()
    if not times:
        return m
    t0 = times[0]
    y0 = angles_deg[0]
    tgt = target_deg
    span = tgt - y0
    if abs(span) < 0.1:
        return m

    # Rise time
    for t, y in zip(times, angles_deg, strict=False):
        if abs(y - y0) >= rise_frac * abs(span):
            m.rise_s = t - t0
            break

    # Overshoot
    if span > 0:
        peak = max(angles_deg)
        m.overshoot_pct = max(0.0, (peak - tgt) / abs(span) * 100.0)
    else:
        trough = min(angles_deg)
        m.overshoot_pct = max(0.0, (tgt - trough) / abs(span) * 100.0)

    # Peaks for damping check
    arr = np.array(angles_deg)
    peaks_idx, _ = signal.find_peaks(np.abs(arr - tgt))
    m.peaks = [float(arr[i]) for i in peaks_idx]
    if len(m.peaks) >= 2:
        m.damped = all(
            abs(m.peaks[i]) < 0.5 * abs(m.peaks[i - 1]) for i in range(1, len(m.peaks))
        )

    # Settle time
    hold_start: float | None = None
    for t, y in zip(times, angles_deg, strict=False):
        if abs(y - tgt) <= settle_band_deg:
            if hold_start is None:
                hold_start = t
            elif t - hold_start >= settle_hold_s:
                m.settle_s = hold_start - t0
                break
        else:
            hold_start = None

    return m


@dataclass
class JitterMetrics:
    roll_std_deg: float = 0.0
    pitch_std_deg: float = 0.0
    roll_ptp_deg: float = 0.0
    pitch_ptp_deg: float = 0.0
    rate_cmd_std: float = 0.0
    psd_peak_db: float = 0.0
    psd_pass: bool = False


def analyze_jitter(
    roll_deg: list[float],
    pitch_deg: list[float],
    rate_cmds: list[float],
    sample_hz: float,
) -> JitterMetrics:
    m = JitterMetrics()
    if len(roll_deg) < 10:
        return m
    r = np.array(roll_deg)
    p = np.array(pitch_deg)
    m.roll_std_deg = float(np.std(r))
    m.pitch_std_deg = float(np.std(p))
    m.roll_ptp_deg = float(np.ptp(r))
    m.pitch_ptp_deg = float(np.ptp(p))
    m.rate_cmd_std = float(np.std(rate_cmds)) if rate_cmds else 0.0

    # PSD on roll signal
    f, psd = signal.welch(r, fs=sample_hz, nperseg=min(256, len(r)))
    bg_mask = (f >= 0.1) & (f < 0.4)
    sig_mask = (f >= 0.5) & (f <= 10.0)
    if np.any(bg_mask) and np.any(sig_mask):
        bg = 10 * np.log10(np.median(psd[bg_mask]) + 1e-12)
        peak_sig = 10 * np.log10(np.max(psd[sig_mask]) + 1e-12)
        m.psd_peak_db = peak_sig - bg
        m.psd_pass = m.psd_peak_db <= 6.0
    else:
        m.psd_pass = True

    return m


def rate_tracking_error(cmd: float, measured: float) -> float:
    if abs(cmd) < 1e-6:
        return 0.0
    return abs(measured - cmd) / abs(cmd)
