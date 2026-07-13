"""Attitude inner-loop controllers."""

from __future__ import annotations

from flightlab.metrics import load_signs
from flightlab.safety import HOVER_THRUST
from flightlab.state import Cmd, State, Target

KP_Z = 0.025
KD_Z = 0.030
RATE_CLIP = 0.5
YAW_CLIP = 0.5
VQ2_RATE_CLIP = 0.30
VQ2_YAW_CLIP = 0.5


def _thrust(s: State, target: Target) -> float:
    if not s.alt_trusted and s.pose_source != "ekf":
        return HOVER_THRUST
    z, vz = s.pos_ned[2], s.vel_ned[2]
    return float(max(0.18, min(0.5, HOVER_THRUST + KP_Z * (z - target.z) + KD_Z * vz)))


class ProportionalController:
    name = "P"

    def __init__(
        self,
        k_att: float = 3.0,
        gain_scale: float = 1.0,
        rate_clip: float = RATE_CLIP,
        yaw_clip: float = YAW_CLIP,
        vq2: bool = False,
    ) -> None:
        self.k_att = k_att * gain_scale
        self.rate_clip = rate_clip
        self.yaw_clip = yaw_clip
        self.signs = load_signs(vq2=vq2)

    def reset(self, s: State) -> None:
        pass

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        sr, sp, sy = self.signs["roll"], self.signs["pitch"], self.signs["yaw"]
        rr = _clip(sr * self.k_att * (target.roll - s.roll), self.rate_clip)
        pr = _clip(sp * self.k_att * (target.pitch - s.pitch), self.rate_clip)
        yr = _clip(sy * self.k_att * _wrap(target.yaw - s.yaw), self.yaw_clip)
        return Cmd(rr, pr, yr, _thrust(s, target))


class PDController:
    name = "PD"

    def __init__(
        self,
        k_att: float = 3.0,
        k_d: float = 0.15,
        gain_scale: float = 1.0,
        rate_clip: float = RATE_CLIP,
        yaw_clip: float = YAW_CLIP,
        vq2: bool = False,
    ) -> None:
        self.k_att = k_att * gain_scale
        self.k_d = k_d
        self.rate_clip = rate_clip
        self.yaw_clip = yaw_clip
        self.signs = load_signs(vq2=vq2)

    def reset(self, s: State) -> None:
        pass

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        sr, sp, sy = self.signs["roll"], self.signs["pitch"], self.signs["yaw"]
        rr = _clip(
            sr * self.k_att * (target.roll - s.roll) - self.k_d * s.gyro[0],
            self.rate_clip,
        )
        pr = _clip(
            sp * self.k_att * (target.pitch - s.pitch) - self.k_d * s.gyro[1],
            self.rate_clip,
        )
        yr = _clip(
            sy * self.k_att * _wrap(target.yaw - s.yaw) - self.k_d * s.gyro[2],
            self.yaw_clip,
        )
        return Cmd(rr, pr, yr, _thrust(s, target))


class ShapedPDController:
    name = "PD_shaped"

    def __init__(
        self,
        k_att: float = 3.0,
        k_d: float = 0.15,
        tau_s: float = 0.05,
        slew_rate: float = 2.0,
        gain_scale: float = 1.0,
        rate_clip: float = RATE_CLIP,
        yaw_clip: float = YAW_CLIP,
        vq2: bool = False,
    ) -> None:
        self._pd = PDController(
            k_att, k_d, gain_scale, rate_clip=rate_clip, yaw_clip=yaw_clip, vq2=vq2
        )
        self.tau = tau_s
        self.slew = slew_rate
        self._filt = [0.0, 0.0, 0.0]
        self._last_cmd = [0.0, 0.0, 0.0]

    def reset(self, s: State) -> None:
        self._pd.reset(s)
        self._filt = [0.0, 0.0, 0.0]
        self._last_cmd = [0.0, 0.0, 0.0]

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        raw = self._pd.update(s, target, dt)
        alpha = dt / (self.tau + dt) if dt > 0 else 1.0
        out = []
        for i, v in enumerate((raw.roll_rate, raw.pitch_rate, raw.yaw_rate)):
            self._filt[i] += alpha * (v - self._filt[i])
            delta = _clip(self._filt[i] - self._last_cmd[i], self.slew * dt)
            self._last_cmd[i] += delta
            out.append(self._last_cmd[i])
        return Cmd(out[0], out[1], out[2], raw.thrust)


class ScheduledPDController:
    name = "PD_scheduled"

    def __init__(
        self,
        k_att: float = 3.0,
        k_d: float = 0.15,
        level_deg: float = 5.0,
        gain_scale: float = 1.0,
        rate_clip: float = RATE_CLIP,
        yaw_clip: float = YAW_CLIP,
        vq2: bool = False,
    ) -> None:
        import math

        self.k_att = k_att * gain_scale
        self.k_d = k_d
        self.rate_clip = rate_clip
        self.yaw_clip = yaw_clip
        self.level_rad = math.radians(level_deg)
        self.signs = load_signs(vq2=vq2)

    def reset(self, s: State) -> None:
        pass

    def update(self, s: State, target: Target, dt: float) -> Cmd:
        tilt = max(abs(s.roll), abs(s.pitch))
        scale = 0.5 + 0.5 * min(1.0, tilt / self.level_rad)
        k = self.k_att * scale
        kd = self.k_d * (2.0 - scale)
        sr, sp, sy = self.signs["roll"], self.signs["pitch"], self.signs["yaw"]
        rr = _clip(sr * k * (target.roll - s.roll) - kd * s.gyro[0], self.rate_clip)
        pr = _clip(sp * k * (target.pitch - s.pitch) - kd * s.gyro[1], self.rate_clip)
        yr = _clip(sy * k * _wrap(target.yaw - s.yaw) - kd * s.gyro[2], self.yaw_clip)
        return Cmd(rr, pr, yr, _thrust(s, target))


CONTROLLERS: dict[str, type] = {
    "P": ProportionalController,
    "PD": PDController,
    "PD_shaped": ShapedPDController,
    "PD_scheduled": ScheduledPDController,
}


def make_controller(name: str, *, vq2: bool = False, **kwargs):
    cls = CONTROLLERS.get(name)
    if cls is None:
        raise ValueError(f"unknown method {name!r}; choose from {list(CONTROLLERS)}")
    if vq2:
        kwargs.setdefault("k_att", 0.6)
        kwargs.setdefault("k_d", 0.12)
        kwargs.setdefault("rate_clip", VQ2_RATE_CLIP)
        kwargs.setdefault("yaw_clip", VQ2_YAW_CLIP)
        kwargs.setdefault("vq2", True)
    return cls(**kwargs)


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _wrap(a: float) -> float:
    import math

    return (a + math.pi) % (2 * math.pi) - math.pi
