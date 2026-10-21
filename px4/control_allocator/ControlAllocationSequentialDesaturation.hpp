/****************************************************************************
 *
 *   Copyright (c) 2019 PX4 Development Team. All rights reserved.
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
 * @file ControlAllocationSequentialDesaturation.hpp
 *
 * Control Allocation Algorithm which sequentially modifies control demands in order to
 * eliminate the saturation of the actuator setpoint vector.
 * Adapted for standalone extraction — ModuleParams dependency removed.
 *
 * @author Roman Bapst <bapstroman@gmail.com>
 */

#pragma once

#include "ControlAllocationPseudoInverse.hpp"

class ControlAllocationSequentialDesaturation: public ControlAllocationPseudoInverse
{
public:

	ControlAllocationSequentialDesaturation() = default;
	virtual ~ControlAllocationSequentialDesaturation() = default;

	PX4RL_HD void allocate() override;

	/**
	 * Set the airmode. Replaces PX4 MC_AIRMODE parameter.
	 * 0 = disabled, 1 = roll/pitch airmode (default), 2 = roll/pitch/yaw airmode
	 */
	PX4RL_HD void setAirmode(int airmode) { _airmode = airmode; }

	/**
	 * Get the current airmode setting.
	 */
	PX4RL_HD int getAirmode() const { return _airmode; }

	// This is the minimum actuator yaw granted when the controller is saturated.
	static constexpr float MINIMUM_YAW_MARGIN{0.15f};

private:

	PX4RL_HD void desaturateActuators(ActuatorVector &actuator_sp, const ActuatorVector &desaturation_vector,
				 bool increase_only = false);

	PX4RL_HD float computeDesaturationGain(const ActuatorVector &desaturation_vector, const ActuatorVector &actuator_sp);

	PX4RL_HD void mixAirmodeRP();
	PX4RL_HD void mixAirmodeRPY();
	PX4RL_HD void mixAirmodeDisabled();
	PX4RL_HD void mixYaw();

	int _airmode{1}; ///< 0=disabled, 1=RP airmode (default), 2=RPY airmode
};
