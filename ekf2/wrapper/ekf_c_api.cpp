// extern "C" facade over PX4's ecl/EKF2 (class Ekf) so it can be driven from
// Python via ctypes. Configured to fuse ONLY IMU + external-vision VELOCITY
// (body-FRD) — no GPS/baro/mag/flow — which is what the vision-only drone has.
// Frames: world = NED, body = FRD. Quaternion is q_nb (body->NED), w-first.
#include "ekf.h"              // class Ekf (global ns); pulls common.h (namespace estimator)
#include <matrix/math.hpp>
#include <cstdint>

using estimator::imuSample;
using estimator::extVisionSample;
using estimator::EvCtrl;
using estimator::VelocityFrame;

struct Ekf2Handle {
	Ekf ekf;
};

extern "C" {

__declspec(dllexport) Ekf2Handle *ekf2_create(void)
{
	Ekf2Handle *h = new Ekf2Handle();
	// GPS/baro/mag/flow are COMPILED OUT (their CONFIG_EKF2_* flags are undefined),
	// so their params don't exist and need no disabling. Only EV is active.
	auto *p = h->ekf.getParamHandle();
	p->ekf2_ev_ctrl   = static_cast<int32_t>(EvCtrl::VEL);   // fuse EV velocity
	p->ekf2_evv_noise = 0.1f;
	p->ekf2_ev_qmin   = 0;
	p->ekf2_ev_delay  = 0.0f;   // caller timestamps already account for latency
	// The PX4 module glue (EKF2.cpp, which we don't build) normally toggles the
	// per-sensor runtime enable each cycle from ekf2_sens_en. Enable EV here.
	h->ekf.getFusionControlHandle()->ev.enabled = true;
	return h;
}

__declspec(dllexport) void ekf2_destroy(Ekf2Handle *h) { delete h; }

// Raw body-FRD IMU: accel = specific force (m/s^2, gravity NOT removed),
// gyro (rad/s), dt (s), t_us (microseconds). Wrapper integrates to deltas.
__declspec(dllexport) void ekf2_push_imu(Ekf2Handle *h,
		double ax, double ay, double az,
		double gx, double gy, double gz,
		double dt, uint64_t t_us)
{
	const float fdt = static_cast<float>(dt);
	imuSample s{};
	s.time_us = t_us;
	s.delta_ang = matrix::Vector3f((float)gx, (float)gy, (float)gz) * fdt;
	s.delta_vel = matrix::Vector3f((float)ax, (float)ay, (float)az) * fdt;
	s.delta_ang_dt = fdt;
	s.delta_vel_dt = fdt;
	h->ekf.setIMUData(s);
}

// External-vision VELOCITY measurement in body-FRD (m/s), isotropic variance.
__declspec(dllexport) void ekf2_push_ext_vision_vel(Ekf2Handle *h,
		double vx, double vy, double vz, double vel_var, uint64_t t_us)
{
	extVisionSample ev{};
	ev.time_us = t_us;
	ev.vel = matrix::Vector3f((float)vx, (float)vy, (float)vz);
	ev.vel_frame = VelocityFrame::BODY_FRAME_FRD;
	const float v = static_cast<float>(vel_var);
	ev.velocity_var = matrix::Vector3f(v, v, v);
	ev.quat = matrix::Quatf();   // identity; unused for VEL-only fusion
	ev.quality = 100;
	ev.reset_counter = 0;
	h->ekf.setExtVisionData(ev);
}

// Run one filter step; returns 1 if the filter produced an update this call.
__declspec(dllexport) int ekf2_update(Ekf2Handle *h) { return h->ekf.update() ? 1 : 0; }

// out: pos[3]=NED position (m), vel_ned[3]=NED velocity (m/s), quat[4]=(w,x,y,z)
// body->NED. valid=1 once tilt alignment is complete (velocity meaningful).
__declspec(dllexport) void ekf2_get_state(Ekf2Handle *h,
		double pos[3], double vel_ned[3], double quat[4], int *valid)
{
	const matrix::Vector3f p = h->ekf.getPosition();
	const matrix::Vector3f vv = h->ekf.getVelocity();
	const matrix::Quatf q = h->ekf.getQuaternion();
	for (int i = 0; i < 3; ++i) { pos[i] = p(i); vel_ned[i] = vv(i); }
	for (int i = 0; i < 4; ++i) { quat[i] = q(i); }
	if (valid) { *valid = h->ekf.attitude_valid() ? 1 : 0; }
}

// Diagnostics: which alignment/aiding flags are set.
__declspec(dllexport) void ekf2_get_status(Ekf2Handle *h,
		int *tilt_align, int *yaw_align, int *ev_vel)
{
	const auto &f = h->ekf.control_status_flags();
	if (tilt_align) { *tilt_align = f.tilt_align; }
	if (yaw_align) { *yaw_align = f.yaw_align; }
	if (ev_vel) { *ev_vel = f.ev_vel; }
}

}  // extern "C"
