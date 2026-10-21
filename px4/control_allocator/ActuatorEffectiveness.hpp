/****************************************************************************
 *
 *   Copyright (c) 2020 PX4 Development Team. All rights reserved.
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
 * @file ActuatorEffectiveness.hpp
 *
 * Interface for Actuator Effectiveness
 * Adapted for standalone extraction — uORB dependencies removed.
 *
 * @author Julien Lecoeur <julien.lecoeur@gmail.com>
 */

#pragma once

#include <cstdint>
#include <cstdio>

#include <matrix/matrix/math.hpp>

enum class AllocationMethod {
	NONE = -1,
	PSEUDO_INVERSE = 0,
	SEQUENTIAL_DESATURATION = 1,
	AUTO = 2,
};

enum class ActuatorType {
	MOTORS = 0,
	SERVOS,

	COUNT
};

enum class EffectivenessUpdateReason {
	NO_EXTERNAL_UPDATE = 0,
	CONFIGURATION_UPDATE = 1,
	MOTOR_ACTIVATION_UPDATE = 2,
};


class ActuatorEffectiveness
{
public:
	ActuatorEffectiveness() = default;
	virtual ~ActuatorEffectiveness() = default;

	static constexpr int NUM_ACTUATORS = 16;
	static constexpr int NUM_AXES = 6;

	enum ControlAxis {
		ROLL = 0,
		PITCH,
		YAW,
		THRUST_X,
		THRUST_Y,
		THRUST_Z
	};

	static constexpr int MAX_NUM_MATRICES = 2;

	using EffectivenessMatrix = matrix::Matrix<float, NUM_AXES, NUM_ACTUATORS>;
	using ActuatorVector = matrix::Vector<float, NUM_ACTUATORS>;

	enum class FlightPhase {
		HOVER_FLIGHT = 0,
		FORWARD_FLIGHT = 1,
		TRANSITION_HF_TO_FF = 2,
		TRANSITION_FF_TO_HF = 3
	};

	struct Configuration {
		/**
		 * Add an actuator to the selected matrix, returning the index, or -1 on error
		 */
		PX4RL_HD int addActuator(ActuatorType type, const matrix::Vector3f &torque, const matrix::Vector3f &thrust);

		/**
		 * Call this after manually adding N actuators to the selected matrix
		 */
		PX4RL_HD void actuatorsAdded(ActuatorType type, int count);

		PX4RL_HD int totalNumActuators() const;

		/// Configured effectiveness matrix
		EffectivenessMatrix effectiveness_matrices[MAX_NUM_MATRICES];

		int num_actuators_matrix[MAX_NUM_MATRICES] {};
		ActuatorVector trim[MAX_NUM_MATRICES];

		ActuatorVector linearization_point[MAX_NUM_MATRICES];

		int selected_matrix {0};

		uint8_t matrix_selection_indexes[NUM_ACTUATORS * MAX_NUM_MATRICES] {};
		int num_actuators[(int)ActuatorType::COUNT] {};
	};

	PX4RL_HD virtual void setFlightPhase(const FlightPhase &flight_phase)
	{
		_flight_phase = flight_phase;
	}

	PX4RL_HD virtual int numMatrices() const { return 1; }

	PX4RL_HD virtual void getDesiredAllocationMethod(AllocationMethod allocation_method_out[MAX_NUM_MATRICES]) const
	{
		for (int i = 0; i < MAX_NUM_MATRICES; ++i) {
			allocation_method_out[i] = AllocationMethod::PSEUDO_INVERSE;
		}
	}

	PX4RL_HD virtual void getNormalizeRPY(bool normalize[MAX_NUM_MATRICES]) const
	{
		for (int i = 0; i < MAX_NUM_MATRICES; ++i) {
			normalize[i] = false;
		}
	}

	PX4RL_HD virtual bool getEffectivenessMatrix(Configuration &configuration, EffectivenessUpdateReason external_update) { return false;}

	PX4RL_HD const FlightPhase &getFlightPhase() const
	{
		return _flight_phase;
	}

	PX4RL_HD virtual const char *name() const = 0;

	PX4RL_HD virtual void allocateAuxilaryControls(const float dt, int matrix_index, ActuatorVector &actuator_sp) {}

	PX4RL_HD virtual void updateSetpoint(const matrix::Vector<float, NUM_AXES> &control_sp,
				    int matrix_index, ActuatorVector &actuator_sp, const matrix::Vector<float, NUM_ACTUATORS> &actuator_min,
				    const matrix::Vector<float, NUM_ACTUATORS> &actuator_max) {}

	PX4RL_HD virtual uint32_t getStoppedMotors() const { return _stopped_motors_mask; }

	PX4RL_HD virtual void stopMaskedMotorsWithZeroThrust(uint32_t stoppable_motors_mask, ActuatorVector &actuator_sp);

protected:
	FlightPhase _flight_phase{FlightPhase::HOVER_FLIGHT};
	uint32_t _stopped_motors_mask{0};
};
