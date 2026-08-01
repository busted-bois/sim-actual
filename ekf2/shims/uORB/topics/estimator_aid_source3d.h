// Hand-written POD shim for the codegen'd uORB struct (from msg/EstimatorAidSource3d.msg).
#pragma once
#include <cstdint>

struct estimator_aid_source3d_s {
	uint64_t timestamp;
	uint64_t timestamp_sample;
	uint8_t estimator_instance;
	uint32_t device_id;
	uint64_t time_last_fuse;
	double observation[3];
	float observation_variance[3];
	float innovation[3];
	float innovation_filtered[3];
	float innovation_variance[3];
	float test_ratio[3];
	float test_ratio_filtered[3];
	bool innovation_rejected;
	bool fused;
};
