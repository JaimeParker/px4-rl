// Row-major SoA buffer glue between the C API and Vehicle. Shared by CpuFleet (loop over i)
// and the CUDA kernels (one thread per i), so there is a single copy of the load/store code.
#pragma once

#include <cstdint>

#include "px4rl/vehicle.hpp"

namespace px4rl {

struct AttitudeStepBuffers {
	const float *q;               // n,4
	const float *q_sp;            // n,4
	const float *yawspeed_ff;     // n or nullptr
	const float *thrust_sp_body;  // n,3
	float *rates_sp_out;          // n,3
	float *thrust_body_out;       // n,3
};

struct RateStepBuffers {
	const float *rates_sp;        // n,3
	const float *thrust_body;     // n,3
	const float *rates;           // n,3
	const float *angular_accel;   // n,3
	float dt;
	const uint8_t *landed;        // n or nullptr
	float *motors_out;            // n,4
	float *actuator_out;          // n,4 or nullptr
	float *torque_sp_out;         // n,3 or nullptr
};

PX4RL_HD inline void attitudeStepAt(Vehicle &vehicle, int i, const AttitudeStepBuffers &b)
{
	const float *q = b.q + 4 * i;
	const float *q_sp = b.q_sp + 4 * i;
	const float yawspeed_ff = b.yawspeed_ff != nullptr ? b.yawspeed_ff[i] : 0.f;

	matrix::Vector3f rates_sp;
	matrix::Vector3f thrust_body;
	vehicle.attitudeStep(matrix::Quatf(q[0], q[1], q[2], q[3]), matrix::Quatf(q_sp[0], q_sp[1], q_sp[2], q_sp[3]),
			     yawspeed_ff, detail::load(b.thrust_sp_body + 3 * i), rates_sp, thrust_body);
	detail::store(b.rates_sp_out + 3 * i, rates_sp);
	detail::store(b.thrust_body_out + 3 * i, thrust_body);
}

PX4RL_HD inline void rateStepAt(Vehicle &vehicle, int i, const RateStepBuffers &b)
{
	RateStepResult result;
	vehicle.rateStep(detail::load(b.rates_sp + 3 * i), detail::load(b.thrust_body + 3 * i),
			 detail::load(b.rates + 3 * i), detail::load(b.angular_accel + 3 * i), b.dt,
			 b.landed != nullptr && b.landed[i] != 0, result);

	for (int k = 0; k < 4; ++k) {
		b.motors_out[4 * i + k] = result.motors[k];
	}

	if (b.actuator_out != nullptr) {
		for (int k = 0; k < 4; ++k) {
			b.actuator_out[4 * i + k] = result.actuator[k];
		}
	}

	if (b.torque_sp_out != nullptr) {
		detail::store(b.torque_sp_out + 3 * i, result.torque_sp);
	}
}

} // namespace px4rl
