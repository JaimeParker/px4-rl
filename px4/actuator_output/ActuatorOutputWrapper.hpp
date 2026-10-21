/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ActuatorOutputWrapper.hpp
 *
 * Clean C++ API for the actuator output stage (Phase 4).
 * Converts normalized motor commands [0,1] from the control allocator
 * to final PWM or DShot integer values.
 *
 * Data flow:
 *   motors[0..3] ∈ [0,1]
 *     → THR_MDL_FAC inverse (linearize motor response)
 *     → PWM: remap [0,1] → [-1,1] → interpolate to [PWM_MIN, PWM_MAX]
 *     → DShot: remap [0,1] → [DSHOT_MIN, DSHOT_MAX] with min fraction
 */

#pragma once

#include "ActuatorOutputTypes.hpp"
#include "ThrustCurve.hpp"

namespace px4_stack {

class ActuatorOutputWrapper {
public:
	ActuatorOutputWrapper() = default;
	~ActuatorOutputWrapper() = default;

	/**
	 * Configure output parameters (thrust model, PWM/DShot ranges, etc.).
	 */
	PX4RL_HD void setParams(const ActuatorOutputParams &params) { _params = params; }

	/**
	 * Get current parameters.
	 */
	PX4RL_HD ActuatorOutputParams getParams() const { return _params; }

	/**
	 * Run one output mapping step.
	 * Applies thrust curve inverse, then maps to PWM or DShot values.
	 *
	 * @param input  Motor commands [0,1] + armed flag
	 * @param output Integer output values (PWM µs or DShot command)
	 */
	PX4RL_HD void update(const ActuatorOutputInput &input, ActuatorOutputOutput &output);

private:
	PX4RL_HD void mapPWM(const float *motors, int num_motors, ActuatorOutputOutput &output);
	PX4RL_HD void mapDShot(const float *motors, int num_motors, ActuatorOutputOutput &output);

	ActuatorOutputParams _params;
};

} // namespace px4_stack
