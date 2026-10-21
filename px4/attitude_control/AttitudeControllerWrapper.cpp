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
 * @file AttitudeControllerWrapper.cpp
 */

#include "AttitudeControllerWrapper.hpp"

namespace px4_stack {

PX4RL_HD AttitudeControllerWrapper::AttitudeControllerWrapper()
{
	// Initialize with default PX4 gains
	applyConfiguration();
}

PX4RL_HD void AttitudeControllerWrapper::setGains(const AttitudeControlConfig &config)
{
	_config = config;
	applyConfiguration();
}

PX4RL_HD void AttitudeControllerWrapper::applyConfiguration()
{
	// Set proportional gains
	matrix::Vector3f proportional_gain(
		_config.roll_p,
		_config.pitch_p,
		_config.yaw_p
	);

	_attitude_control.setProportionalGain(proportional_gain, _config.yaw_weight);

	// Set rate limits
	matrix::Vector3f rate_limit(
		_config.max_roll_rate,
		_config.max_pitch_rate,
		_config.max_yaw_rate
	);

	_attitude_control.setRateLimit(rate_limit);
}

PX4RL_HD RateSetpoint AttitudeControllerWrapper::update(const AttitudeState &state,
                                                const AttitudeSetpoint &setpoint)
{
	// Set the attitude setpoint in the controller
	_attitude_control.setAttitudeSetpoint(setpoint.q_d, setpoint.yaw_sp_move_rate);

	// Run the control algorithm
	matrix::Vector3f rates_sp = _attitude_control.update(state.q);

	// Prepare output
	RateSetpoint output;
	output.timestamp = state.timestamp;
	output.rates = rates_sp;
	output.thrust_body = setpoint.thrust_body;  // Pass through thrust

	return output;
}

PX4RL_HD void AttitudeControllerWrapper::reset()
{
	// Currently no internal state to reset
	// The core AttitudeControl class only stores setpoint which is updated each cycle
	// Placeholder for future extensions (e.g., filters, integrators)
}

} // namespace px4_stack
