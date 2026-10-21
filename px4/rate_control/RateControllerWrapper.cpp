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
 * @file RateControllerWrapper.cpp
 *
 * Replicates the gain-scaling and yaw LPF post-processing from
 * PX4's MulticopterRateControl::Run() / parameters_updated() methods.
 */

#include "RateControllerWrapper.hpp"

namespace px4_stack {

PX4RL_HD RateControllerWrapper::RateControllerWrapper()
{
	applyParams();
}

PX4RL_HD void RateControllerWrapper::setParams(const RateControlParams &params)
{
	_params = params;
	applyParams();
}

PX4RL_HD void RateControllerWrapper::applyParams()
{
	// Apply K-scaling (ideal form → parallel form conversion)
	// Matches PX4 MulticopterRateControl::parameters_updated()
	const matrix::Vector3f effective_p = _params.gain_k.emult(_params.gain_p);
	const matrix::Vector3f effective_i = _params.gain_k.emult(_params.gain_i);
	const matrix::Vector3f effective_d = _params.gain_k.emult(_params.gain_d);

	_rate_control.setPidGains(effective_p, effective_i, effective_d);
	_rate_control.setIntegratorLimit(_params.integrator_limit);
	_rate_control.setFeedForwardGain(_params.gain_ff);

	// Configure yaw torque low-pass filter
	_yaw_lpf.setCutoffFreq(_params.yaw_torque_lpf_cutoff_freq);
	_yaw_lpf.reset(0.f);
}

PX4RL_HD void RateControllerWrapper::update(const RateControlInput &input, RateControlOutput &output)
{
	// Run the core PID rate controller
	matrix::Vector3f torque = _rate_control.update(
		input.rates_actual,
		input.rates_sp,
		input.angular_accel,
		input.dt,
		input.landed
	);

	// Apply yaw torque low-pass filter (matches MulticopterRateControl::Run())
	if (input.dt > 0.f) {
		torque(2) = _yaw_lpf.update(torque(2), input.dt);
	}

	// Fill output
	output.torque_sp = torque;
	output.thrust_body = input.thrust_body;
}

PX4RL_HD void RateControllerWrapper::setSaturationStatus(const matrix::Vector3<bool> &saturation_positive,
						const matrix::Vector3<bool> &saturation_negative)
{
	_rate_control.setSaturationStatus(saturation_positive, saturation_negative);
}

PX4RL_HD void RateControllerWrapper::resetIntegral()
{
	_rate_control.resetIntegral();
	_yaw_lpf.reset(0.f);
}

PX4RL_HD void RateControllerWrapper::getStatus(RateCtrlStatus &status)
{
	_rate_control.getRateControlStatus(status);
}

} // namespace px4_stack
