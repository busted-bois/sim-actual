"""Python front-end for the PX4 ecl/EKF2 state estimator (vendored + built as a
native DLL under ekf2/, driven via ctypes).

This fuses HIGHRES_IMU (accel+gyro) with a body-FRD VISION VELOCITY measurement
into a clean, bias-corrected velocity — the thing raw IMU dead-reckoning can't
give us vision-only (it diverges to tens of m/s). Configured for IMU + external-
vision VELOCITY only; GPS/baro/mag/flow are compiled out.

Frames: world = NED, body = FRD. Feed accel as SPECIFIC FORCE (m/s^2, gravity
NOT removed), gyro in rad/s, both body-FRD. Read `velocity_body` (forward,
right, down) once `valid` (tilt aligned + EV velocity fusing).

    ekf = EclEkf()
    ekf.push_imu(ax, ay, az, gx, gy, gz, dt, t_us)      # every IMU sample
    ekf.push_vision_velocity(vx, vy, vz, var, t_us)     # each vision fix (body-FRD)
    ekf.update()                                        # every tick
    vb = ekf.velocity_body()                            # (fwd, right, down) m/s
"""

from __future__ import annotations

import ctypes
import os

_DLL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "ekf2", "Release", "ekf2core.dll"),
    os.path.join(os.path.dirname(__file__), "..", "ekf2", "ekf2core.dll"),
]


def _find_dll() -> str:
    for c in _DLL_CANDIDATES:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "ekf2core.dll not found — build it: cmake -S ekf2 -B ekf2/build "
        "-G 'Visual Studio 17 2022' -A x64 && cmake --build ekf2/build --config Release"
    )


class EclEkf:
    """Thin ctypes wrapper over the ecl/EKF2 C API (ekf2/wrapper/ekf_c_api.cpp)."""

    _lib = None  # class-level cached CDLL

    def __init__(self):
        if EclEkf._lib is None:
            lib = ctypes.CDLL(_find_dll())
            lib.ekf2_create.restype = ctypes.c_void_p
            lib.ekf2_destroy.argtypes = [ctypes.c_void_p]
            lib.ekf2_push_imu.argtypes = [ctypes.c_void_p] + [ctypes.c_double] * 7 + [ctypes.c_uint64]
            lib.ekf2_push_ext_vision_vel.argtypes = [ctypes.c_void_p] + [ctypes.c_double] * 6 + [ctypes.c_uint64]
            lib.ekf2_update.argtypes = [ctypes.c_void_p]
            lib.ekf2_update.restype = ctypes.c_int
            lib.ekf2_set_at_rest.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.ekf2_set_in_air.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.ekf2_get_state.argtypes = [
                ctypes.c_void_p, ctypes.c_double * 3, ctypes.c_double * 3,
                ctypes.c_double * 4, ctypes.POINTER(ctypes.c_int),
            ]
            lib.ekf2_get_status.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            EclEkf._lib = lib
        self._h = EclEkf._lib.ekf2_create()
        self._pos = (ctypes.c_double * 3)()
        self._vel = (ctypes.c_double * 3)()
        self._quat = (ctypes.c_double * 4)()
        self._valid = ctypes.c_int(0)

    def push_imu(self, ax, ay, az, gx, gy, gz, dt, t_us):
        """Body-FRD accel (specific force, m/s^2) + gyro (rad/s), dt (s), t_us (microseconds)."""
        EclEkf._lib.ekf2_push_imu(
            self._h, float(ax), float(ay), float(az),
            float(gx), float(gy), float(gz), float(dt), int(t_us),
        )

    def push_vision_velocity(self, vx, vy, vz, var, t_us):
        """Body-FRD velocity measurement (m/s). `var` is either a scalar isotropic
        variance or a (var_x, var_y, var_z) triple — hand an axis a huge variance
        to make the filter ignore that component ((m/s)^2)."""
        if isinstance(var, (tuple, list)):
            vxr, vyr, vzr = (float(v) for v in var)
        else:
            vxr = vyr = vzr = float(var)
        EclEkf._lib.ekf2_push_ext_vision_vel(
            self._h, float(vx), float(vy), float(vz), vxr, vyr, vzr, int(t_us),
        )

    def update(self) -> bool:
        return bool(EclEkf._lib.ekf2_update(self._h))

    def set_at_rest(self, at_rest: bool) -> None:
        """Assert the vehicle is stationary -> enables the Zero-Velocity Update,
        the fast tilt aligner. Set during the pre-flight hold; clear once moving."""
        EclEkf._lib.ekf2_set_at_rest(self._h, 1 if at_rest else 0)

    def set_in_air(self, in_air: bool) -> None:
        EclEkf._lib.ekf2_set_in_air(self._h, 1 if in_air else 0)

    def _refresh(self):
        EclEkf._lib.ekf2_get_state(
            self._h, self._pos, self._vel, self._quat, ctypes.byref(self._valid)
        )

    def velocity_ned(self):
        self._refresh()
        return (self._vel[0], self._vel[1], self._vel[2])

    def velocity_body(self):
        """Fused velocity rotated into body FRD: (forward, right, down) m/s."""
        self._refresh()
        vn, ve, vd = self._vel[0], self._vel[1], self._vel[2]
        w, x, y, z = self._quat[0], self._quat[1], self._quat[2], self._quat[3]
        # v_body = R_nb^T * v_ned  (R_nb from q_nb, body->NED). Rows of R^T:
        r00 = 1 - 2 * (y * y + z * z); r01 = 2 * (x * y + w * z); r02 = 2 * (x * z - w * y)
        r10 = 2 * (x * y - w * z); r11 = 1 - 2 * (x * x + z * z); r12 = 2 * (y * z + w * x)
        r20 = 2 * (x * z + w * y); r21 = 2 * (y * z - w * x); r22 = 1 - 2 * (x * x + y * y)
        fwd = r00 * vn + r01 * ve + r02 * vd
        right = r10 * vn + r11 * ve + r12 * vd
        down = r20 * vn + r21 * ve + r22 * vd
        return (fwd, right, down)

    def quaternion(self):
        self._refresh()
        return (self._quat[0], self._quat[1], self._quat[2], self._quat[3])

    def valid(self) -> bool:
        self._refresh()
        return bool(self._valid.value)

    def status(self):
        """(tilt_align, yaw_align, ev_vel) booleans."""
        ti = ctypes.c_int(0); ya = ctypes.c_int(0); ev = ctypes.c_int(0)
        EclEkf._lib.ekf2_get_status(self._h, ctypes.byref(ti), ctypes.byref(ya), ctypes.byref(ev))
        return (bool(ti.value), bool(ya.value), bool(ev.value))

    def close(self):
        if getattr(self, "_h", None):
            EclEkf._lib.ekf2_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Smoke: stationary align, then vision says fwd 1.5 m/s -> fused tracks it.
    ekf = EclEkf()
    t = 0.0
    dt = 0.004
    for i in range(2500):
        t += dt
        tus = int(t * 1e6)
        ekf.push_imu(0.0, 0.0, -9.81, 0.0, 0.0, 0.0, dt, tus)
        if i % 8 == 0:
            ekf.push_vision_velocity(1.5 if i >= 1000 else 0.0, 0.0, 0.0, 0.05, tus)
        ekf.update()
    vb = ekf.velocity_body()
    print(f"[ecl_ekf selftest] status={ekf.status()} velocity_body={tuple(round(x,3) for x in vb)} (target fwd 1.5)")
    assert abs(vb[0] - 1.5) < 0.2, vb
    print("[ecl_ekf selftest] OK")
