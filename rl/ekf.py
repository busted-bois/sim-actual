"""Module 5 — Error-State EKF (loosely-coupled INS + vision).

Nominal state : position p(3), velocity v(3), orientation q(4, body->world).
Error state   : [dp(3), dv(3), dtheta(3)]  (9-D), body-frame attitude error.

Predict  : IMU strapdown (accel + gyro), world NED, gravity = +Z(down).
Update 1 : vision world-position from PnP (Module 4).
Update 2 : attitude anchor from the sim's reported quaternion.

IMU convention (validated by the synthetic self-test): the accelerometer
reports specific force f = R^T (a_world - g). Prediction inverts it as
a_world = R f + g. If the live sim uses the opposite gravity sign, flip
GRAVITY_SIGN — a one-line change.

    uv run -m rl.ekf --selftest
"""

from __future__ import annotations

import argparse

import numpy as np

from rl.spec import quat_to_R

GRAVITY = 9.81
GRAVITY_SIGN = 1.0  # +1: g points +Z (down) in NED
G_WORLD = np.array([0.0, 0.0, GRAVITY_SIGN * GRAVITY])


# ---- quaternion helpers (w,x,y,z) ------------------------------------------
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
    ):
        self.p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, float).copy()
        self.q = (
            np.array([1.0, 0, 0, 0]) if q0 is None else quat_norm(np.asarray(q0, float))
        )
        self.P = np.eye(9) * (p0_std**2)
        self.sa, self.sg = sigma_accel, sigma_gyro
        self.s_pos, self.s_att = sigma_pos, sigma_att

    # ---- prediction --------------------------------------------------------
    def predict(self, accel_body, gyro_body, dt):
        if dt <= 0 or dt > 0.5:
            return
        accel_body = np.asarray(accel_body, float)
        gyro_body = np.asarray(gyro_body, float)
        R = quat_to_R(self.q)
        a_world = R @ accel_body + G_WORLD

        # Nominal integration.
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(gyro_body * dt)))

        # Error-state transition F = I + A dt.
        A = np.zeros((9, 9))
        A[0:3, 3:6] = np.eye(3)
        A[3:6, 6:9] = -R @ skew(accel_body)
        A[6:9, 6:9] = -skew(gyro_body)
        F = np.eye(9) + A * dt

        # Process noise (accel on velocity, gyro on attitude).
        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = (self.sa**2) * dt * dt * np.eye(3)
        Q[6:9, 6:9] = (self.sg**2) * dt * dt * np.eye(3)
        self.P = F @ self.P @ F.T + Q

    # ---- generic update ----------------------------------------------------
    def _update(self, H, r, Rm):
        S = H @ self.P @ H.T + Rm
        Kk = self.P @ H.T @ np.linalg.inv(S)
        dx = Kk @ r
        self._inject(dx)
        I_KH = np.eye(9) - Kk @ H
        self.P = I_KH @ self.P @ I_KH.T + Kk @ Rm @ Kk.T  # Joseph form

    def _inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = quat_norm(quat_mult(self.q, quat_from_smallangle(dx[6:9])))

    def update_position(self, p_meas, sigma=None):
        s = self.s_pos if sigma is None else sigma
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        r = np.asarray(p_meas, float) - self.p
        self._update(H, r, (s**2) * np.eye(3))

    def update_position_gated(
        self, p_meas, sigma=None, chi2_thresh: float = 11.34
    ) -> bool:
        """Position update with Mahalanobis outlier rejection (3-dof, ~99%)."""
        s = self.s_pos if sigma is None else sigma
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        r = np.asarray(p_meas, float) - self.p
        Rm = (s**2) * np.eye(3)
        S = H @ self.P @ H.T + Rm
        mahal = float(r.T @ np.linalg.inv(S) @ r)
        if mahal > chi2_thresh:
            return False
        self._update(H, r, Rm)
        return True

    def update_attitude(self, q_meas, sigma=None):
        s = self.s_att if sigma is None else sigma
        dq = quat_mult(quat_inv(self.q), quat_norm(np.asarray(q_meas, float)))
        if dq[0] < 0:
            dq = -dq
        dtheta = 2.0 * dq[1:4]  # small-angle attitude error (body frame)
        H = np.zeros((3, 9))
        H[:, 6:9] = np.eye(3)
        self._update(H, dtheta, (s**2) * np.eye(3))

    def state(self):
        return {
            "p": self.p.copy(),
            "v": self.v.copy(),
            "q": self.q.copy(),
            "P_trace": float(np.trace(self.P)),
        }


# ---------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(3)
    dt = 1 / 100.0
    T = 8.0
    n = int(T / dt)
    # True trajectory: gentle 3D sinusoid, constant level attitude.
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
    # Start with wrong position/velocity to show convergence.
    ekf = ESKF(p0=p0 + np.array([1.5, -1.0, 0.8]), v0=v0 + 0.5, q0=q_true)

    errs = []
    for k in range(n):
        t = k * dt
        _, _, a = true_pv(t)
        accel_body = R.T @ (a - G_WORLD) + rng.normal(
            0, 0.05, 3
        )  # specific force + noise
        gyro_body = rng.normal(0, 0.005, 3)
        ekf.predict(accel_body, gyro_body, dt)
        if k % 10 == 0:  # 10 Hz vision
            p_true, _, _ = true_pv(t)
            ekf.update_position(p_true + rng.normal(0, 0.1, 3))
            ekf.update_attitude(q_true)
        p_true, v_true, _ = true_pv(t)
        errs.append([np.linalg.norm(ekf.p - p_true), np.linalg.norm(ekf.v - v_true)])

    errs = np.array(errs)
    final_p = errs[-100:, 0].mean()
    final_v = errs[-100:, 1].mean()
    init_p = errs[0, 0]
    print(
        f"[selftest] init pos err={init_p:.2f}m -> converged "
        f"pos={final_p:.3f}m vel={final_v:.3f}m/s (P_trace={ekf.state()['P_trace']:.3f})"
    )
    assert final_p < 0.2, "position should converge"
    assert final_v < 0.4, "velocity should converge"
    print("[selftest] OK — EKF fuses IMU predict + vision/attitude update")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    _selftest()
