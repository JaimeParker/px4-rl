/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ActuatorOutputWrapper.cpp
 *
 * Implementation of the actuator output mapping stage (Phase 4).
 * Applies thrust model factor inverse, then maps to PWM or DShot values.
 *
 * PWM path:  motor[0,1] → thrust curve inverse → [-1,1] → interpolate → [PWM_MIN, PWM_MAX]
 * DShot path: motor[0,1] → thrust curve inverse → [DSHOT_MIN, DSHOT_MAX] with min fraction
 */

#include "ActuatorOutputWrapper.hpp"

#include <mathlib/math/Functions.hpp>
#include <cmath>

namespace px4_stack {

PX4RL_HD void ActuatorOutputWrapper::update(const ActuatorOutputInput &input, ActuatorOutputOutput &output)
{
	const int n = input.num_motors;
	output.num_motors = n;

	// If not armed, output disarmed values immediately
	if (!input.armed) {
		const uint16_t disarmed_val = (_params.mode == OutputMode::PWM)
					      ? _params.pwm_disarmed
					      : _params.dshot_disarmed;

		for (int i = 0; i < n; ++i) {
			output.values[i] = disarmed_val;
		}

		return;
	}

	// Copy motor commands to working array
	float motors[4];

	for (int i = 0; i < n; ++i) {
		motors[i] = input.motors[i];
	}

	// Apply thrust model factor inverse (linearize motor response)
	applyThrustCurveInverse(motors, n, _params.thrust_model_factor);

	// Map to output protocol
	switch (_params.mode) {
	case OutputMode::PWM:
		mapPWM(motors, n, output);
		break;

	case OutputMode::DShot:
		mapDShot(motors, n, output);
		break;
	}
}

PX4RL_HD void ActuatorOutputWrapper::mapPWM(const float *motors, int num_motors, ActuatorOutputOutput &output)
{
	// PX4 convention for non-reversible motors:
	//   1. Remap [0, 1] → [-1, 1]:  value = motor * 2 - 1
	//   2. Interpolate [-1, 1]  → [PWM_MIN, PWM_MAX]
	//
	// Extracted from MixingOutput::output_limit_calc_single() in
	// src/lib/mixer_module/mixer_module.cpp (ON state).

	const float pwm_min = static_cast<float>(_params.pwm_min);
	const float pwm_max = static_cast<float>(_params.pwm_max);

	for (int i = 0; i < num_motors; ++i) {
		// Remap [0,1] → [-1,1]
		const float value = motors[i] * 2.f - 1.f;

		// Linear interpolation with clamping
		const float pwm_f = math::interpolate(value, -1.f, 1.f, pwm_min, pwm_max);

		// Round to integer, constrain to uint16 range
		const long rounded = lroundf(pwm_f);
		output.values[i] = static_cast<uint16_t>(math::constrain(rounded, 0L, static_cast<long>(UINT16_MAX)));
	}
}

PX4RL_HD void ActuatorOutputWrapper::mapDShot(const float *motors, int num_motors, ActuatorOutputOutput &output)
{
	// DShot mapping for non-reversible motors:
	//   1. Compute effective min from min_fraction: min_output = max * dshot_min_fraction
	//   2. Scale motor [0,1] into [min_output, max]
	//   3. Clamp to [DSHOT_MIN, DSHOT_MAX]
	//
	// Extracted from DShot.cpp motor_output_channel() logic.

	const float max_output = static_cast<float>(_params.dshot_max_throttle);
	const float min_fraction_output = max_output * _params.dshot_min_fraction;

	// Effective minimum is at least dshot_min_throttle
	const float effective_min = fmaxf(static_cast<float>(_params.dshot_min_throttle), min_fraction_output);

	for (int i = 0; i < num_motors; ++i) {
		const float motor = motors[i];

		// Scale [0,1] → [effective_min, max_output]
		float dshot_f = effective_min + motor * (max_output - effective_min);

		// Clamp to valid DShot range
		dshot_f = fmaxf(static_cast<float>(_params.dshot_min_throttle), fminf(dshot_f, max_output));

		output.values[i] = static_cast<uint16_t>(lroundf(dshot_f));
	}
}

} // namespace px4_stack
