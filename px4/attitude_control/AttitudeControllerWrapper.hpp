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
 * @file AttitudeControllerWrapper.hpp
 *
 * Standalone wrapper for PX4 attitude controller
 * Provides a clean interface for use in RL environments
 */

#pragma once

#include "AttitudeControl.hpp"
#include "AttitudeControlTypes.hpp"

namespace px4_stack {

/**
 * @class AttitudeControllerWrapper
 *
 * Wrapper class providing a simplified interface to the PX4 attitude controller
 * for autonomous mode operation (no manual control, no VTOL, no EKF reset handling)
 */
class AttitudeControllerWrapper {
public:
	/**
	 * Constructor
	 * Initializes the controller with default PX4 gains
	 */
	PX4RL_HD AttitudeControllerWrapper();

	/**
	 * Destructor
	 */
	~AttitudeControllerWrapper() = default;

	/**
	 * Configure controller gains and rate limits
	 * @param config Configuration structure with gains and limits
	 */
	PX4RL_HD void setGains(const AttitudeControlConfig &config);

	/**
	 * Get current configuration
	 * @return Current configuration
	 */
	PX4RL_HD AttitudeControlConfig getGains() const { return _config; }

	/**
	 * Update the attitude controller
	 * @param state Current vehicle attitude state
	 * @param setpoint Desired attitude setpoint from position controller
	 * @return Rate setpoint for the rate controller
	 */
	PX4RL_HD RateSetpoint update(const AttitudeState &state, const AttitudeSetpoint &setpoint);

	/**
	 * Reset the controller (currently no internal state, placeholder for future)
	 */
	PX4RL_HD void reset();

private:
	/**
	 * Apply configuration to the underlying AttitudeControl instance
	 */
	PX4RL_HD void applyConfiguration();

	AttitudeControl _attitude_control;      ///< Core PX4 attitude control algorithm
	AttitudeControlConfig _config;          ///< Current configuration
};

} // namespace px4_stack
