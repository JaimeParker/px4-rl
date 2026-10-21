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
 * @file ControlAllocation.hpp
 *
 * Interface for Control Allocation Algorithms
 *
 * @author Julien Lecoeur <julien.lecoeur@gmail.com>
 */

#pragma once

#include <matrix/matrix/math.hpp>

#include "ActuatorEffectiveness.hpp"

class ControlAllocation
{
public:
	PX4RL_HD ControlAllocation();
	virtual ~ControlAllocation() = default;

	static constexpr uint8_t NUM_ACTUATORS = ActuatorEffectiveness::NUM_ACTUATORS;
	static constexpr uint8_t NUM_AXES = ActuatorEffectiveness::NUM_AXES;

	typedef matrix::Vector<float, NUM_ACTUATORS> ActuatorVector;

	enum ControlAxis {
		ROLL = 0,
		PITCH,
		YAW,
		THRUST_X,
		THRUST_Y,
		THRUST_Z
	};

	/**
	 * Allocate control setpoint to actuators
	 */
	PX4RL_HD virtual void allocate() = 0;

	/**
	 * Set actuator failure flag
	 */
	PX4RL_HD void setHadActuatorFailure(bool failure) { _had_actuator_failure = failure; }

	/**
	 * Set the control effectiveness matrix
	 */
	PX4RL_HD virtual void setEffectivenessMatrix(const matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS> &effectiveness,
					    const ActuatorVector &actuator_trim, const ActuatorVector &linearization_point, int num_actuators,
					    bool update_normalization_scale);

	/**
	 * Get the allocated actuator vector
	 */
	PX4RL_HD const matrix::Vector<float, NUM_ACTUATORS> &getActuatorSetpoint() const { return _actuator_sp; }

	/**
	 * Set the desired control vector
	 */
	PX4RL_HD void setControlSetpoint(const matrix::Vector<float, NUM_AXES> &control) { _control_sp = control; }

	/**
	 * Get the desired control vector
	 */
	PX4RL_HD const matrix::Vector<float, NUM_AXES> &getControlSetpoint() const { return _control_sp; }

	/**
	 * Get the allocated control vector
	 */
	PX4RL_HD matrix::Vector<float, NUM_AXES> getAllocatedControl() const
	{ return (_effectiveness * (_actuator_sp - _actuator_trim)).emult(_control_allocation_scale); }

	/**
	 * Get the control effectiveness matrix
	 */
	PX4RL_HD const matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS> &getEffectivenessMatrix() const { return _effectiveness; }

	/**
	 * Set the minimum actuator values
	 */
	PX4RL_HD void setActuatorMin(const matrix::Vector<float, NUM_ACTUATORS> &actuator_min) { _actuator_min = actuator_min; }

	/**
	 * Get the minimum actuator values
	 */
	PX4RL_HD const matrix::Vector<float, NUM_ACTUATORS> &getActuatorMin() const { return _actuator_min; }

	/**
	 * Set the maximum actuator values
	 */
	PX4RL_HD void setActuatorMax(const matrix::Vector<float, NUM_ACTUATORS> &actuator_max) { _actuator_max = actuator_max; }

	/**
	 * Get the maximum actuator values
	 */
	PX4RL_HD const matrix::Vector<float, NUM_ACTUATORS> &getActuatorMax() const { return _actuator_max; }

	/**
	 * Set the current actuator setpoint
	 */
	PX4RL_HD void setActuatorSetpoint(const matrix::Vector<float, NUM_ACTUATORS> &actuator_sp);

	PX4RL_HD void setSlewRateLimit(const matrix::Vector<float, NUM_ACTUATORS> &slew_rate_limit)
	{ _actuator_slew_rate_limit = slew_rate_limit; }

	/**
	 * Apply slew rate to current actuator setpoint
	 */
	PX4RL_HD void applySlewRateLimit(float dt);

	/**
	 * Clip the actuator setpoint between minimum and maximum values.
	 */
	PX4RL_HD void clipActuatorSetpoint(matrix::Vector<float, NUM_ACTUATORS> &actuator) const;

	PX4RL_HD void clipActuatorSetpoint() { clipActuatorSetpoint(_actuator_sp); }

	/**
	 * Normalize the actuator setpoint between minimum and maximum values.
	 */
	PX4RL_HD matrix::Vector<float, NUM_ACTUATORS> normalizeActuatorSetpoint(const matrix::Vector<float, NUM_ACTUATORS> &actuator)
	const;

	PX4RL_HD virtual void updateParameters() {}

	PX4RL_HD int numConfiguredActuators() const { return _num_actuators; }

	PX4RL_HD void setNormalizeRPY(bool normalize_rpy) { _normalize_rpy = normalize_rpy; }

protected:
	matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS> _effectiveness;
	matrix::Vector<float, NUM_AXES> _control_allocation_scale;
	matrix::Vector<float, NUM_ACTUATORS> _actuator_trim;
	matrix::Vector<float, NUM_ACTUATORS> _actuator_min;
	matrix::Vector<float, NUM_ACTUATORS> _actuator_max;
	matrix::Vector<float, NUM_ACTUATORS> _actuator_slew_rate_limit;
	matrix::Vector<float, NUM_ACTUATORS> _prev_actuator_sp;
	matrix::Vector<float, NUM_ACTUATORS> _actuator_sp;
	matrix::Vector<float, NUM_AXES> _control_sp;
	matrix::Vector<float, NUM_AXES> _control_trim;
	int _num_actuators{0};
	bool _normalize_rpy{false};
	bool _had_actuator_failure{false};
};
