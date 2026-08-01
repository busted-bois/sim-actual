// Hand-written POD shim for the codegen'd uORB struct (from msg/EstimatorAidSource2d.msg).
#pragma once
#include <cstdint>

struct estimator_aid_source2d_s {
	uint64_t timestamp;
	uint64_t timestamp_sample;
	uint8_t estimator_instance;
	uint32_t device_id;
	uint64_t time_last_fuse;
	double observation[2];
	float observation_variance[2];
	float innovation[2];
	float innovation_filtered[2];
	float innovation_variance[2];
	float test_ratio[2];
	float test_ratio_filtered[2];
	bool innovation_rejected;
	bool fused;
};
