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

// External-vision VELOCITY measurement in body-FRD (m/s), PER-AXIS variance so a
// caller can down-weight an axis it distrusts (e.g. forward PnP velocity) by
// handing it a huge variance instead of fusing it.
__declspec(dllexport) void ekf2_push_ext_vision_vel(Ekf2Handle *h,
		double vx, double vy, double vz,
		double var_x, double var_y, double var_z, uint64_t t_us)
{
	extVisionSample ev{};
	ev.time_us = t_us;
	ev.vel = matrix::Vector3f((float)vx, (float)vy, (float)vz);
	ev.vel_frame = VelocityFrame::BODY_FRAME_FRD;
	ev.velocity_var = matrix::Vector3f((float)var_x, (float)var_y, (float)var_z);
	ev.quat = matrix::Quatf();   // identity; unused for VEL-only fusion
	ev.quality = 100;
	ev.reset_counter = 0;
	h->ekf.setExtVisionData(ev);
}

// Run one filter step; returns 1 if the filter produced an update this call.
__declspec(dllexport) int ekf2_update(Ekf2Handle *h) { return h->ekf.update() ? 1 : 0; }

// The excluded module glue (EKF2.cpp) normally sets these from the land detector.
// vehicle_at_rest gates the Zero-Velocity Update, which is the FAST tilt aligner
// (tight obs_var while unaligned) — set it during the stationary pre-flight hold
// so tilt aligns in ~1s instead of ~8s. Clear it (and set in_air) once moving.
__declspec(dllexport) void ekf2_set_at_rest(Ekf2Handle *h, int at_rest) {
	h->ekf.set_vehicle_at_rest(at_rest != 0);
}
__declspec(dllexport) void ekf2_set_in_air(Ekf2Handle *h, int in_air) {
	h->ekf.set_in_air_status(in_air != 0);
}

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

// Estimator covariance (state variances) — how uncertain the filter thinks it
// is. vel_var[3]=NED velocity variance (m/s)^2, pos_var[3]=NED position variance
// (m)^2, gyro_bias_var[3] (rad/s)^2, accel_bias_var[3] (m/s^2)^2.
__declspec(dllexport) void ekf2_get_variance(Ekf2Handle *h,
		double vel_var[3], double pos_var[3],
		double gyro_bias_var[3], double accel_bias_var[3])
{
	const matrix::Vector3f vv = h->ekf.getVelocityVariance();
	const matrix::Vector3f pv = h->ekf.getPositionVariance();
	const matrix::Vector3f gv = h->ekf.getGyroBiasVariance();
	const matrix::Vector3f av = h->ekf.getAccelBiasVariance();
	for (int i = 0; i < 3; ++i) {
		vel_var[i] = vv(i); pos_var[i] = pv(i);
		gyro_bias_var[i] = gv(i); accel_bias_var[i] = av(i);
	}
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
