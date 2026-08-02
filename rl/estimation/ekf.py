"""Module 5 — Error-State EKF (loosely-coupled INS + vision).

Nominal state : position p(3), velocity v(3), orientation q(4, body->world).
Error state   : [dp(3), dv(3), dtheta(3)]  (9-D), body-frame attitude error.

Predict  : IMU strapdown (accel + gyro), world NED, gravity = +Z(down).
           Powered-flight predict_commanded uses thrust model:
           a_world = R @ f_thrust_body - drag_kd * v + g_world
Update 1 : vision world-position from PnP (Module 4).
Update 2 : attitude anchor from the sim's reported quaternion.

    uv run -m rl.estimation.ekf --selftest
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from rl.core.spec import quat_to_R

GRAVITY = 9.81
GRAVITY_SIGN = 1.0
G_WORLD = np.array([0.0, 0.0, GRAVITY_SIGN * GRAVITY])

C_THRUST = 36.0
DRAG_KD = 0.6
GYRO_SPIKE = 15.0


def quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0, 0, 0])


def quat_from_smallangle(dtheta):
    half = 0.5 * dtheta
    return quat_norm(np.array([1.0, half[0], half[1], half[2]]))


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (q @ q)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _symmetrize(P):
    return 0.5 * (P + P.T)


def _validate_vec(name, v, shape):
    v = np.asarray(v, float)
    if v.shape != (shape,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be finite ({shape},), got shape={v.shape}")
    return v


class ESKF:
    def __init__(
        self,
        p0=None,
        v0=None,
        q0=None,
        sigma_accel=0.3,
        sigma_gyro=0.02,
        sigma_pos=0.15,
        sigma_att=0.03,
        p0_std=1.0,
        c_thrust: float = C_THRUST,
        drag_kd: float = DRAG_KD,
        gravity_sign: float = GRAVITY_SIGN,
    ):
        self.p = np.zeros(3) if p0 is None else _validate_vec("p0", p0, 3).copy()
        self.v = np.zeros(3) if v0 is None else _validate_vec("v0", v0, 3).copy()
        self.q = (
            np.array([1.0, 0, 0, 0])
            if q0 is None
            else quat_norm(_validate_vec("q0", q0, 4))
        )
        if not (math.isfinite(p0_std) and p0_std > 0):
            raise ValueError(f"p0_std must be positive finite, got {p0_std}")
        self.P = np.eye(9) * (p0_std**2)
        self.sa, self.sg = sigma_accel, sigma_gyro
        self.s_pos, self.s_att = sigma_pos, sigma_att
        if gravity_sign not in (1.0, -1.0):
            raise ValueError(f"gravity_sign must be ±1.0, got {gravity_sign}")
        self.gravity_sign = gravity_sign
        self.g_world = np.array([0.0, 0.0, self.gravity_sign * GRAVITY])
        if not (math.isfinite(c_thrust) and c_thrust > 0):
            raise ValueError(f"c_thrust must be positive finite, got {c_thrust}")
        self.c_thrust = c_thrust
        if not (math.isfinite(drag_kd) and drag_kd >= 0):
            raise ValueError(f"drag_kd must be nonnegative finite, got {drag_kd}")
        self.drag_kd = drag_kd
        self._coast_count = 0
        self._recovery_count = 0

    @property
    def coast_count(self) -> int:
        return self._coast_count

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    def healthy(self) -> bool:
        return bool(
            np.isfinite(self.p).all()
            and np.isfinite(self.v).all()
            and np.isfinite(self.q).all()
            and np.isfinite(self.P).all()
        )

    def predict(self, accel_body, gyro_body, dt):
        if dt <= 0 or dt > 0.5:
            return
        accel_body = np.asarray(accel_body, float)
        gyro_body = np.asarray(gyro_body, float)
        if not (np.isfinite(accel_body).all() and np.isfinite(gyro_body).all()):
            return
        R = quat_to_R(self.q)
        a_world = R @ accel_body + self.g_world

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(gyro_body * dt)))

        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 6:9] = -R @ skew(accel_body)
        A[6:9, 6:9] = -skew(gyro_body)
        F = np.eye(9) + A * dt

        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = (self.sa**2) * dt * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    def predict_commanded(self, thrust: float, gyro_body, dt):
        if dt <= 0 or dt > 0.5:
            return
        if not isinstance(thrust, (int, float, np.integer, np.floating)):
            return
        thrust = float(thrust)
        if not (math.isfinite(thrust) and thrust >= 0):
            return
        if thrust < 0.05:
            self.predict_zupt(gyro_body, dt)
            return
        try:
            gyro_body = np.asarray(gyro_body, float)
        except (TypeError, ValueError):
            return
        if gyro_body.shape != (3,) or not np.isfinite(gyro_body).all():
            return
        if float(np.linalg.norm(gyro_body)) > GYRO_SPIKE:
            return
        R = quat_to_R(self.q)
        f_thrust_body = np.array(
            [0.0, 0.0, -self.gravity_sign * self.c_thrust * thrust]
        )
        a_world = R @ f_thrust_body - self.drag_kd * self.v + self.g_world

        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(gyro_body * dt)))

        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 3:6] = -self.drag_kd * np.eye(3)
        A[3:6, 6:9] = -R @ skew(f_thrust_body)
        A[6:9, 6:9] = -skew(gyro_body)
        F = np.eye(9) + A * dt

        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = (self.sa**2) * dt * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        self.P = _symmetrize(F @ self.P @ F.T + Q)

    def predict_zupt(self, gyro_body, dt):
        if dt <= 0 or dt > 0.5:
            return
        try:
            gyro_body = np.asarray(gyro_body, float)
        except (TypeError, ValueError):
            return
        if gyro_body.shape != (3,) or not np.isfinite(gyro_body).all():
            return
        if float(np.linalg.norm(gyro_body)) > GYRO_SPIKE:
            return
        self.v = np.zeros(3)
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(gyro_body * dt)))

        A = np.zeros((9, 9))
        A[6:9, 6:9] = -skew(gyro_body)
        F = np.eye(9) + A * dt
        Q = np.zeros((9, 9))
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        self.P = _symmetrize(F @ self.P @ F.T + Q)

    def coast(self, steps: int, dt: float):
        if dt <= 0 or dt > 0.5 or steps < 1:
            return
        if not np.isfinite(self.P).all():
            self._coast_count += steps
            return
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = (self.sa**2) * dt * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        for _ in range(steps):
            self.P = _symmetrize(F @ self.P @ F.T + Q)
        self._coast_count += steps

    def reset(self, p0=None, v0=None, q0=None, p0_std: float | None = None):
        p = np.zeros(3) if p0 is None else _validate_vec("p0", p0, 3).copy()
        v = np.zeros(3) if v0 is None else _validate_vec("v0", v0, 3).copy()
        q = (
            np.array([1.0, 0, 0, 0])
            if q0 is None
            else quat_norm(_validate_vec("q0", q0, 4))
        )
        std = p0_std if p0_std is not None else 1.0
        if not (math.isfinite(std) and std > 0):
            raise ValueError(f"p0_std must be positive finite, got {std}")
        self.p = p
        self.v = v
        self.q = q
        self.P = np.eye(9) * (std**2)
        self._coast_count = 0
        self._recovery_count += 1

    def notify_collision(self):
        self.v = np.zeros(3)
        self.P[0:3, 0:3] += np.eye(3) * 25.0
        self.P[3:6, 3:6] += np.eye(3) * 9.0
        self.P = _symmetrize(self.P)

    def innovation_gate(self, floor: float = 1.0) -> float:
        sigma_p = math.sqrt(max(1e-9, float(np.trace(self.P[0:3, 0:3])) / 3))
        return max(floor, 3.0 * sigma_p)

    def reset_on_detection(
        self, *, drone_pos=None, pnp_gate_body=None, gate_world_pos=None, q=None
    ):
        has_pos = drone_pos is not None
        has_pnp = pnp_gate_body is not None and gate_world_pos is not None
        if has_pos and has_pnp:
            raise ValueError(
                "Pass drone_pos OR (pnp_gate_body, gate_world_pos), not both"
            )
        if not has_pos and not has_pnp:
            raise ValueError("Pass drone_pos OR (pnp_gate_body, gate_world_pos)")
        q_use = self.q.copy()
        if q is not None:
            q_use = quat_norm(_validate_vec("q", q, 4))
        if has_pos:
            self.p = _validate_vec("drone_pos", drone_pos, 3).copy()
        else:
            gb = _validate_vec("pnp_gate_body", pnp_gate_body, 3)
            gw = _validate_vec("gate_world_pos", gate_world_pos, 3)
            R_wb = quat_to_R(q_use)
            self.p = (gw - R_wb @ gb).copy()
        self.v = np.zeros(3)
        self.q = q_use
        self.P = np.eye(9)
        self._coast_count = 0
        self._recovery_count += 1

    def _update(self, H, r, Rm):
        r = np.asarray(r, float)
        if not np.isfinite(r).all():
            return
        S = H @ self.P @ H.T + Rm
        try:
            Kk = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        if not np.isfinite(Kk).all():
            return
        dx = Kk @ r
        self._inject(dx)
        I_KH = np.eye(9) - Kk @ H
        self.P = _symmetrize(I_KH @ self.P @ I_KH.T + Kk @ Rm @ Kk.T)

    def _inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(dx[6:9])))
        self._coast_count = 0

    def update_position(self, p_meas, sigma=None):
        s = self.s_pos if sigma is None else sigma
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        r = np.asarray(p_meas, float) - self.p
        self._update(H, r, (s**2) * np.eye(3))

    def update_attitude(self, q_meas, sigma=None):
        s = self.s_att if sigma is None else sigma
        dq = quat_mult(quat_inv(self.q), quat_norm(np.asarray(q_meas, float)))
        if dq[0] < 0:
            dq = -dq
        dtheta = 2.0 * dq[1:4]
        H = np.zeros((3, 9))
        H[:, 6:9] = np.eye(3)
        self._update(H, dtheta, (s**2) * np.eye(3))

    def state(self):
        return {
            "p": self.p.copy(),
            "v": self.v.copy(),
            "q": self.q.copy(),
            "P_trace": float(np.trace(self.P)),
            "coast_count": self._coast_count,
            "recovery_count": self._recovery_count,
        }


def _selftest():
    rng = np.random.default_rng(3)
    dt = 1 / 100.0
    T = 8.0
    n = int(T / dt)
    q_true = np.array([1.0, 0, 0, 0])
    R = quat_to_R(q_true)

    def true_pv(t):
        p = np.array(
            [2 * np.sin(0.5 * t), 1.5 * np.sin(0.3 * t + 1), -3 + 0.5 * np.sin(0.4 * t)]
        )
        v = np.array(
            [
                2 * 0.5 * np.cos(0.5 * t),
                1.5 * 0.3 * np.cos(0.3 * t + 1),
                0.5 * 0.4 * np.cos(0.4 * t),
            ]
        )
        a = np.array(
            [
                -2 * 0.25 * np.sin(0.5 * t),
                -1.5 * 0.09 * np.sin(0.3 * t + 1),
                -0.5 * 0.16 * np.sin(0.4 * t),
            ]
        )
        return p, v, a

    p0, v0, _ = true_pv(0)
    ekf = ESKF(p0=p0 + np.array([1.5, -1.0, 0.8]), v0=v0 + 0.5, q0=q_true)

    errs = []
    for k in range(n):
        t = k * dt
        _, _, a = true_pv(t)
        accel_body = R.T @ (a - G_WORLD) + rng.normal(0, 0.05, 3)
        gyro_body = rng.normal(0, 0.005, 3)
        ekf.predict(accel_body, gyro_body, dt)
        if k % 10 == 0:
            p_true, _, _ = true_pv(t)
            ekf.update_position(p_true + rng.normal(0, 0.1, 3))
            ekf.update_attitude(q_true)
        p_true, v_true, _ = true_pv(t)
        errs.append([np.linalg.norm(ekf.p - p_true), np.linalg.norm(ekf.v - v_true)])

    errs = np.array(errs)
    final_p = errs[-100:, 0].mean()
    final_v = errs[-100:, 1].mean()
    print(
        f"[selftest] init pos err -> converged "
        f"pos={final_p:.3f}m vel={final_v:.3f}m/s (P_trace={ekf.state()['P_trace']:.3f})"
    )
    assert final_p < 0.2
    assert final_v < 0.4

    print("[selftest] testing predict_commanded (hover, +1)...")
    ekf2 = ESKF(p0=np.zeros(3), v0=np.zeros(3), q0=q_true, sigma_accel=2.0)
    hover_t = GRAVITY / C_THRUST
    for k in range(500):
        ekf2.predict_commanded(hover_t, rng.normal(0, 0.002, 3), dt)
    drift_pos = np.linalg.norm(ekf2.p)
    assert ekf2.healthy()
    assert drift_pos < 0.1, f"NED hover drift={drift_pos:.4f}m"
    print(f"[selftest] NED hover drift={drift_pos:.6f}m OK")

    print("[selftest] testing predict_commanded (hover, -1)...")
    ekf2n = ESKF(
        p0=np.zeros(3), v0=np.zeros(3), q0=q_true, sigma_accel=2.0, gravity_sign=-1.0
    )
    for k in range(500):
        ekf2n.predict_commanded(hover_t, rng.normal(0, 0.002, 3), dt)
    drift_neg = np.linalg.norm(ekf2n.p)
    assert ekf2n.healthy()
    assert drift_neg < 0.1, f"flipped hover drift={drift_neg:.4f}m"
    print(f"[selftest] flipped hover drift={drift_neg:.6f}m OK")

    ratio = abs(drift_pos - drift_neg) / max(drift_pos, 1e-12)
    print(f"[selftest] sign drift ratio={ratio:.4f}")
    assert ratio < 0.1, f"sign drift ratio {ratio:.4f} >= 0.1"

    print("[selftest] testing zero thrust -> ZUPT...")
    ekf_zt = ESKF(p0=np.zeros(3), v0=np.array([3, 0, 0]), q0=q_true, sigma_accel=2.0)
    for k in range(200):
        ekf_zt.predict_commanded(0.0, rng.normal(0, 0.002, 3), dt)
    assert np.max(np.abs(ekf_zt.v)) < 1e-15
    print("[selftest] zero-thrust ZUPT OK")

    print("[selftest] testing coast...")
    ekf3 = ESKF(p0=np.array([1, 2, 3]), v0=np.array([0.1, 0, -0.1]), q0=q_true)
    tr0 = ekf3.state()["P_trace"]
    ekf3.coast(10, dt)
    assert ekf3.coast_count == 10
    assert ekf3.state()["P_trace"] > tr0
    print(f"[selftest] coast trace {tr0:.3f} -> {ekf3.state()['P_trace']:.3f} OK")

    print("[selftest] testing reset with v0...")
    ekf4 = ESKF()
    ekf4.reset(p0=np.zeros(3), v0=np.array([1, 2, 3]))
    assert np.max(np.abs(ekf4.v - [1, 2, 3])) < 1e-15
    print("[selftest] reset v0 OK")

    print("[selftest] testing notify_collision...")
    ekf5 = ESKF(p0=np.zeros(3), v0=np.array([5, 0, 0]), q0=q_true)
    ekf5.notify_collision()
    assert np.allclose(ekf5.v, 0)
    assert ekf5.state()["P_trace"] > 9.0
    print("[selftest] notify_collision OK")

    print("[selftest] testing covariance symmetry...")
    ekf7 = ESKF(q0=q_true)
    for _ in range(200):
        ekf7.predict_commanded(GRAVITY / C_THRUST, rng.normal(0, 0.005, 3), dt)
    ekf7.update_position(rng.normal(0, 0.2, 3))
    assert np.max(np.abs(ekf7.P - ekf7.P.T)) < 1e-12
    print("[selftest] covariance symmetry OK")

    print("[selftest] testing parameter validation...")
    for bad in [0.5, 2.0]:
        try:
            ESKF(gravity_sign=bad)
            assert False
        except ValueError:
            pass
    for bad in [-1.0, float("nan")]:
        try:
            ESKF(c_thrust=bad)
            assert False
        except ValueError:
            pass
    for bad in [-0.1, float("inf")]:
        try:
            ESKF(drag_kd=bad)
            assert False
        except ValueError:
            pass
    for bad in [0.0, -1.0]:
        try:
            ESKF(p0_std=bad)
            assert False
        except ValueError:
            pass
    print("[selftest] parameter validation OK")

    print("[selftest] ALL OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
