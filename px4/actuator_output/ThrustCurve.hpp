/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ThrustCurve.hpp
 *
 * Thrust model factor (THR_MDL_FAC) inverse curve.
 * Extracted from PX4's FunctionMotors::updateValues() — standalone, no PX4 dependencies.
 *
 * The motor thrust model is: rel_thrust = f * x² + (1-f) * x
 * where f = thrust_model_factor, x = normalized motor signal [0,1].
 *
 * Given a desired thrust (control value), we solve for x (signal) via
 * the quadratic inverse:
 *   x = -(1-f)/(2f) + sqrt( (1-f)²/(4f²) + control/f )
 *
 * When f = 0, the model is linear: x = control (no transformation needed).
 */

#pragma once

#include <cfloat>
#include <cmath>

namespace px4_stack {

/**
 * Apply the thrust model factor inverse to an array of motor commands.
 * Converts desired thrust [0,1] → linearized motor signal [0,1].
 *
 * Non-reversible motors only: negative values are clamped to 0.
 *
 * This is a verbatim extraction of the core math from
 * PX4's FunctionMotors::updateValues() (src/lib/mixer_module/functions/FunctionMotors.hpp).
 *
 * @param motors          Array of motor commands [0,1], modified in-place
 * @param num_motors      Number of motors in the array
 * @param thrust_factor   THR_MDL_FAC parameter [0,1], 0 = linear passthrough
 */
PX4RL_HD inline void applyThrustCurveInverse(float *motors, int num_motors, float thrust_factor)
{
	if (thrust_factor > FLT_EPSILON && thrust_factor <= 1.f) {
		// Thrust model: rel_thrust = a * x^2 + b * x
		const float a = thrust_factor;
		const float b = (1.f - thrust_factor);

		// Precompute quadratic formula terms (ax^2 + bx - control = 0)
		const float tmp1 = b / (2.f * a);
		const float tmp2 = b * b / (4.f * a * a);

		for (int i = 0; i < num_motors; ++i) {
			const float control = motors[i];

			if (std::isfinite(control) && control > FLT_EPSILON) {
				// Positive root of quadratic: x = -b/(2a) + sqrt(b²/(4a²) + c/a)
				motors[i] = -tmp1 + sqrtf(tmp2 + (control / a));
			}
			// control <= 0 or NaN: leave as-is (will be clamped below)
		}
	}

	// Non-reversible: clamp negatives to 0
	for (int i = 0; i < num_motors; ++i) {
		if (!std::isfinite(motors[i]) || motors[i] < 0.f) {
			motors[i] = 0.f;
		}

		if (motors[i] > 1.f) {
			motors[i] = 1.f;
		}
	}
}

/**
 * Apply the thrust model forward curve (for testing round-trips).
 * Computes: rel_thrust = f * x² + (1-f) * x
 *
 * @param x               Motor signal [0,1]
 * @param thrust_factor   THR_MDL_FAC parameter [0,1]
 * @return Resulting thrust [0,1]
 */
PX4RL_HD inline float thrustCurveForward(float x, float thrust_factor)
{
	return thrust_factor * x * x + (1.f - thrust_factor) * x;
}

} // namespace px4_stack
