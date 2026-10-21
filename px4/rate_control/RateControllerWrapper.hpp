/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/**
 * @file RateControllerWrapper.hpp
 *
 * Standalone wrapper for PX4 rate controller.
 * Provides a clean interface for use in RL environments.
 *
 * Replicates the gain-scaling and yaw LPF post-processing from
 * PX4's MulticopterRateControl module without any middleware.
 *
 * IMPORTANT: The angular_accel input (D-term) must be provided by the
 * simulation environment. In PX4, this comes from the gyro pipeline.
 * For RL use, compute it as finite-difference of angular velocity
 * or from the physics engine's analytic derivative.
 */

#pragma once

#include "RateControl.hpp"
#include "RateControlTypes.hpp"
#include <mathlib/math/filter/AlphaFilter.hpp>

namespace px4_stack {

class RateControllerWrapper {
public:
	/**
	 * Constructor — initializes with default PX4 gains
	 */
	PX4RL_HD RateControllerWrapper();
	~RateControllerWrapper() = default;

	/**
	 * Configure all rate controller parameters.
	 * Applies K-scaling (ideal→parallel form conversion) internally.
	 * @param params Configuration struct with PID gains, limits, and filter settings
	 */
	PX4RL_HD void setParams(const RateControlParams &params);

	/**
	 * Get current configuration
	 */
	PX4RL_HD RateControlParams getParams() const { return _params; }

	/**
	 * Run one rate control cycle.
	 * @param input  Rate controller inputs (rate_sp, rates_actual, angular_accel, dt, landed)
	 * @param output Rate controller outputs (torque_sp, thrust_body passthrough)
	 */
	PX4RL_HD void update(const RateControlInput &input, RateControlOutput &output);

	/**
	 * Set saturation feedback from control allocator (Phase 3 hook).
	 * Used for integrator anti-windup.
	 */
	PX4RL_HD void setSaturationStatus(const matrix::Vector3<bool> &saturation_positive,
				 const matrix::Vector3<bool> &saturation_negative);

	/**
	 * Reset integrator to zero on all axes
	 */
	PX4RL_HD void resetIntegral();

	/**
	 * Get integrator states for debugging/logging
	 */
	PX4RL_HD void getStatus(RateCtrlStatus &status);

private:
	PX4RL_HD void applyParams();

	RateControl _rate_control;         ///< Core PX4 rate control algorithm
	AlphaFilter<float> _yaw_lpf;       ///< Low-pass filter on yaw torque output
	RateControlParams _params;         ///< Current configuration
};

} // namespace px4_stack
