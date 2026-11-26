// Hand-written POD shim for the codegen'd uORB struct (from msg/EstimatorAidSource1d.msg).
#pragma once
#include <cstdint>

struct estimator_aid_source1d_s {
	uint64_t timestamp;
	uint64_t timestamp_sample;
	uint8_t estimator_instance;
	uint32_t device_id;
	uint64_t time_last_fuse;
	float observation;
	float observation_variance;
	float innovation;
	float innovation_filtered;
	float innovation_variance;
	float test_ratio;
	float test_ratio_filtered;
	bool innovation_rejected;
	bool fused;
};
