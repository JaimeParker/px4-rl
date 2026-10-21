// Batched CPU counterpart of the CUDA fleet: same operations, host pointers, optional OpenMP.
#pragma once

#include <cstdint>
#include <vector>

#include "px4rl/c_api.h"
#include "px4rl/vehicle.hpp"

namespace px4rl {

class CpuFleet {
public:
	/// Constructs n vehicles with px4rl_params_default(). Throws std::invalid_argument if n < 1.
	explicit CpuFleet(int n);

	int size() const { return static_cast<int>(vehicles_.size()); }

	/// mask: host uint8[n] or nullptr (= all). n_params must be 1 (broadcast) or n, otherwise
	/// std::invalid_argument is thrown, as it is for parameters that fail validateParams().
	void reset(const uint8_t *mask, const px4rl_params *params, int n_params);

	/// Same arguments as px4rl_cuda_position_step, on host pointers; see c_api.h.
	void positionStep(const float *position, const float *velocity, const float *heading, const float *acceleration_z,
			  const float *trajectory_sp, const uint8_t *in_air, float dt, uint64_t time_us, float *q_sp_out,
			  float *thrust_sp_body_out, float *yawspeed_out, float *hover_thrust_out, uint8_t *outcome_out);

	/// q_sp == nullptr uses the setpoint stored by positionStep; see px4rl_cuda_attitude_step.
	void attitudeStep(const float *q, const float *q_sp, const float *yawspeed_ff, const float *thrust_sp_body,
			  float *rates_sp_out, float *thrust_body_out);

	void rateStep(const float *rates_sp, const float *thrust_body, const float *rates, const float *angular_accel,
		      float dt, const uint8_t *landed, float *motors_out, float *actuator_out, float *torque_sp_out);

private:
	std::vector<Vehicle> vehicles_;
};

} // namespace px4rl
