/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ActuatorOutputTypes.hpp
 *
 * Parameter, input and output data structures for the actuator output stage.
 * Converts normalized motor commands [0,1] to PWM or DShot values.
 */

#pragma once

#include <cstdint>

namespace px4_stack {

/**
 * Output mode selection.
 */
enum class OutputMode {
	PWM,    ///< Standard PWM output [PWM_MIN, PWM_MAX] µs
	DShot   ///< DShot digital protocol [1, 1999]
};

/**
 * Actuator output configuration parameters.
 * Replaces PX4 THR_MDL_FAC, PWM_MAIN_*, and DSHOT_* parameters.
 */
struct ActuatorOutputParams {
	/// Thrust model factor [0, 1]. 0 = linear, 0.3 = typical racer.
	/// Maps: rel_thrust = f * x² + (1-f) * x. We apply the inverse.
	float thrust_model_factor{0.0f};

	/// Output mode: PWM or DShot.
	OutputMode mode{OutputMode::PWM};

	// --- PWM mode parameters ---

	/// PWM minimum value [µs] (motor idle speed when armed).
	uint16_t pwm_min{1000};

	/// PWM maximum value [µs] (full throttle).
	uint16_t pwm_max{2000};

	/// PWM disarmed value [µs] (motors stopped).
	uint16_t pwm_disarmed{900};

	// --- DShot mode parameters ---

	/// DShot minimum throttle command (motors spinning at idle).
	uint16_t dshot_min_throttle{1};

	/// DShot maximum throttle command (full throttle).
	uint16_t dshot_max_throttle{1999};

	/// DShot disarmed value (motors stopped).
	uint16_t dshot_disarmed{0};

	/// DShot minimum throttle fraction [0, 1]. Ensures motors don't stall.
	/// Effective min = max * dshot_min_fraction.
	float dshot_min_fraction{0.0f};
};

/**
 * Input to the actuator output stage.
 * Motor commands come from the control allocator (Phase 3).
 */
struct ActuatorOutputInput {
	/// Normalized motor commands [0, 1] from the control allocator.
	float motors[4] = {0.f, 0.f, 0.f, 0.f};

	/// Number of motors.
	int num_motors{4};

	/// Whether the vehicle is armed. If false, outputs disarmed values.
	bool armed{true};
};

/**
 * Output from the actuator output stage.
 * Final integer values ready for the hardware ESC protocol.
 */
struct ActuatorOutputOutput {
	/// Output values: PWM [µs] or DShot command per motor.
	uint16_t values[4] = {0, 0, 0, 0};

	/// Number of motors.
	int num_motors{4};
};

} // namespace px4_stack
