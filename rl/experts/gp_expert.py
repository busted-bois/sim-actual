"""AndurilGP guidance as an offline expert for GateRacingEnv.

Wraps simulator.gp_pilot.compute_guidance with synthetic 30 Hz "vision" built
from privileged env state (gate position/normal in body FRD), so the GP
racing behavior can generate (obs, action) demonstrations for BC pretraining
without the live sim. Returns normalized [-1,1]^4 actions (spec.unscale_action).

    uv run -m rl.gp_expert --selftest
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from rl.core import spec
from rl.environment.env import DECISION_HZ
from simulator.gp_pilot import _fresh_hold_state, compute_guidance
from simulator.gp_vision import VisionVelocityTracker

VISION_HZ = 30.0  # live FPV camera rate the guidance D-terms assume

# compute_guidance emits deg-of-error as deg/s of rate (their live sim's
# quat-encoding quirk absorbed the gain). Through the env's first-order rate
# plant that is a ~1 s attitude time constant — far too slow to track the
# bank/pitch targets. Measured sweep: x6 clears stages 0-1 fully.
ATT_RATE_GAIN = 6.0


def _euler_deg(q: np.ndarray) -> tuple[float, float]:
    """Roll/pitch (deg) from quaternion (w,x,y,z), aerospace convention."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


class GPExpert:
    """Stateful GP guidance expert. Call act() once per env decision step."""

    def __init__(self):
        self.dt = 1.0 / DECISION_HZ
        self.tracker = VisionVelocityTracker()
        self.reset()

    def reset(self) -> None:
        self.hold = _fresh_hold_state()
        self.tracker.reset()
        self._t = 0.0
        self._gate_idx: int | None = None
        self._vision: dict | None = None
        self._frame_id = -1

    def _synthetic_vision(
        self, p: np.ndarray, R: np.ndarray, gate: dict
    ) -> dict | None:
        """Privileged gate pose as an AndurilGP vision_gate_estimate, 30 Hz."""
        frame_id = int(self._t * VISION_HZ)
        if frame_id == self._frame_id:
            return self._vision  # camera hasn't produced a new frame yet
        self._frame_id = frame_id

        gc = np.asarray(gate["pos"], dtype=np.float64)
        b = R.T @ (gc - p)  # body FRD
        if b[0] <= 0.1:  # gate behind/beside: no detection (matches guidance guard)
            self._vision = None
            return None
        n_world = spec.quat_to_R(np.asarray(gate["quat"], dtype=np.float64)) @ np.array(
            [1.0, 0.0, 0.0]
        )
        self._vision = {
            "frame_id": frame_id,
            "body_x_m": float(b[0]),
            "body_y_m": float(b[1]),
            "body_z_m": float(b[2]),
            "pnp_ok": True,
            "pnp_rvec": None,
            "normal_body": R.T @ n_world,
            "u_px": None,
            "v_px": None,
        }
        return self._vision

    def act(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        gate_map: list,
        gate_idx: int,
    ) -> np.ndarray:
        """Normalized [-1,1]^4 action toward gate_map[gate_idx]."""
        if gate_idx != self._gate_idx:
            # Gate switch = fresh acquisition: drop stale D-term history.
            self.hold = _fresh_hold_state()
            self.tracker.reset()
            self._vision = None
            self._frame_id = -1
            self._gate_idx = gate_idx
        self._t += self.dt

        q = np.asarray(q, dtype=np.float64)
        R = spec.quat_to_R(q)
        roll_deg, pitch_deg = _euler_deg(q)
        v = np.asarray(v, dtype=np.float64)
        vY = float((R.T @ v)[1])  # body-right
        vD = float(v[2])  # NED down

        gate_idx = int(np.clip(gate_idx, 0, len(gate_map) - 1))
        gate = gate_map[gate_idx]
        gate_quat = None
        if isinstance(gate, dict) and "quat" in gate:
            try:
                gate_quat = np.asarray(gate["quat"], dtype=np.float64)
            except (TypeError, ValueError):
                gate_quat = None
        vision = self._synthetic_vision(np.asarray(p, dtype=np.float64), R, gate)
        vision_vel = self.tracker.update(vision)

        roll_cmd_deg, pitch_cmd_deg, yaw_cmd_deg, thrust, _dbg = compute_guidance(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            quat=q,
            vY=vY,
            vD=vD,
            vision=vision,
            vision_vel=vision_vel,
            state=self.hold,
            hover_thrust=spec.HOVER_THRUST,  # env plant hovers at spec value
            gate_quat=gate_quat,
        )
        # The live pilot ships these degree commands on the attitude-quat
        # wire; the internal env is a rate plant, so interpret deg -> rad/s
        # here (the exact conversion the old rate-port applied).
        roll_r = math.radians(roll_cmd_deg)
        pitch_r = math.radians(pitch_cmd_deg)
        yaw_r = math.radians(yaw_cmd_deg)
        # compute_guidance emits the live sim's inverted roll/yaw signs
        # (gp_pilot KR = KY = -1, per measured signs.json). The internal env
        # integrates proper NED/FRD rates, so flip those two axes back.
        g = ATT_RATE_GAIN
        return spec.unscale_action(
            np.array([-roll_r * g, pitch_r * g, -yaw_r * g, thrust])
        )


def _selftest():
    from rl.environment.env import GateRacingEnv

    expert = GPExpert()
    passes = trials = 0
    for trial in range(6):
        env = GateRacingEnv(stage=0, seed=200 + trial)
        obs, _ = env.reset()
        expert.reset()
        trials += 1
        term = trunc = False
        while not (term or trunc):
            a = expert.act(env.p, env.v, env.q, env.gate_map, env.gate_idx)
            assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)
            obs, r, term, trunc, info = env.step(a)
            if info.get("gate_passed"):
                passes += 1
                break
    print(f"[selftest] GP expert passed {passes}/{trials} stage-0 gates")
    assert passes >= 5, "GP expert should clear nearly all stage-0 gates"
    print("[selftest] OK — GP guidance flies the internal env")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
