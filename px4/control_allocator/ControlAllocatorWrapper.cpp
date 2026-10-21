/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ControlAllocatorWrapper.cpp
 *
 * Clean C++ API for the PX4 control allocator.
 */

#include "ControlAllocatorWrapper.hpp"

#include <cfloat>
#include <cmath>

namespace px4_stack {

PX4RL_HD ControlAllocatorWrapper::ControlAllocatorWrapper()
{
	applyParams();
}

PX4RL_HD void ControlAllocatorWrapper::setParams(const ControlAllocatorParams &params)
{
	_params = params;
	applyParams();
}

PX4RL_HD void ControlAllocatorWrapper::applyParams()
{
	// Build effectiveness matrix from geometry
	ActuatorEffectiveness::EffectivenessMatrix effectiveness{};
	_num_motors = computeEffectivenessMatrix(_params.geometry, effectiveness, 0);

	// Set up actuator bounds (16-element vectors, only first _num_motors entries matter)
	ControlAllocation::ActuatorVector actuator_min;
	ControlAllocation::ActuatorVector actuator_max;
	ControlAllocation::ActuatorVector actuator_trim;
	ControlAllocation::ActuatorVector linearization_point;
	ControlAllocation::ActuatorVector slew_rate;

	actuator_min.setAll(0.f);
	actuator_max.setAll(1.f);
	actuator_trim.setAll(0.f);
	linearization_point.setAll(0.f);
	slew_rate.setAll(0.f);

	for (int i = 0; i < _num_motors && i < 4; i++) {
		actuator_min(i) = _params.actuator_min[i];
		actuator_max(i) = _params.actuator_max[i];
		slew_rate(i) = _params.slew_rate_limit[i];
	}

	// Configure the allocation
	_allocation.setEffectivenessMatrix(effectiveness, actuator_trim, linearization_point,
					   _num_motors, true);
	_allocation.setActuatorMin(actuator_min);
	_allocation.setActuatorMax(actuator_max);
	_allocation.setNormalizeRPY(_params.normalize_rpy);
	_allocation.setAirmode(_params.airmode);
	_allocation.setSlewRateLimit(slew_rate);
}

PX4RL_HD void ControlAllocatorWrapper::update(const ControlAllocatorInput &input, ControlAllocatorOutput &output)
{
	// Build the 6D control vector: [roll_torque, pitch_torque, yaw_torque, thrust_x, thrust_y, thrust_z]
	matrix::Vector<float, 6> control_sp;
	control_sp(ControlAllocation::ROLL) = input.torque_sp(0);
	control_sp(ControlAllocation::PITCH) = input.torque_sp(1);
	control_sp(ControlAllocation::YAW) = input.torque_sp(2);
	control_sp(ControlAllocation::THRUST_X) = input.thrust_sp(0);
	control_sp(ControlAllocation::THRUST_Y) = input.thrust_sp(1);
	control_sp(ControlAllocation::THRUST_Z) = input.thrust_sp(2);

	// Set control setpoint and allocate
	_allocation.setControlSetpoint(control_sp);
	_allocation.allocate();

	// Apply slew rate limit
	if (input.dt > 0.f) {
		_allocation.applySlewRateLimit(input.dt);
	}

	// Clip to actuator bounds
	_allocation.clipActuatorSetpoint();

	// Extract motor outputs
	const auto &actuator_sp = _allocation.getActuatorSetpoint();
	output.num_motors = _num_motors;

	for (int i = 0; i < _num_motors && i < 4; i++) {
		output.motors[i] = actuator_sp(i);
	}

	// Compute unallocated control for saturation feedback
	// allocated_control = effectiveness * (actuator_sp - actuator_trim) * scale
	matrix::Vector<float, 6> allocated_control = _allocation.getAllocatedControl();
	matrix::Vector<float, 6> unallocated = control_sp - allocated_control;

	// Extract torque-axis unallocated control
	output.unallocated_torque(0) = unallocated(ControlAllocation::ROLL);
	output.unallocated_torque(1) = unallocated(ControlAllocation::PITCH);
	output.unallocated_torque(2) = unallocated(ControlAllocation::YAW);

	// Extract thrust-axis unallocated control (consumed by the hover thrust estimator)
	output.unallocated_thrust(0) = unallocated(ControlAllocation::THRUST_X);
	output.unallocated_thrust(1) = unallocated(ControlAllocation::THRUST_Y);
	output.unallocated_thrust(2) = unallocated(ControlAllocation::THRUST_Z);

	// Compute saturation flags: if unallocated torque exceeds threshold, axis is saturated
	// This matches PX4's MulticopterRateControl logic for feeding back to the rate controller
	bool torque_setpoint_achieved = (fabsf(output.unallocated_torque(0)) < FLT_EPSILON &&
					 fabsf(output.unallocated_torque(1)) < FLT_EPSILON &&
					 fabsf(output.unallocated_torque(2)) < FLT_EPSILON);

	for (int i = 0; i < 3; i++) {
		if (!torque_setpoint_achieved) {
			output.saturation_positive(i) = (output.unallocated_torque(i) > FLT_EPSILON);
			output.saturation_negative(i) = (output.unallocated_torque(i) < -FLT_EPSILON);
		} else {
			output.saturation_positive(i) = false;
			output.saturation_negative(i) = false;
		}
	}
}

} // namespace px4_stack
