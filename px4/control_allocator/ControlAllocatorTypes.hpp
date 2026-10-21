/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ControlAllocatorTypes.hpp
 *
 * Parameter, input and output data structures for the standalone control allocator.
 * Follows the project pattern: plain C++ structs replace uORB and PX4 params.
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include "RotorGeometry.hpp"

namespace px4_stack {

/**
 * Control allocator configuration parameters.
 * Replaces PX4 CA_* and MC_AIRMODE parameters.
 */
struct ControlAllocatorParams {
	/// Rotor geometry. Default: Iris quadrotor X.
	Geometry geometry = createQuadXIris();

	/// Airmode: 0=disabled, 1=RP airmode (default), 2=RPY airmode.
	int airmode{1};

	/// Actuator minimum values per motor (default 0.0 for non-reversible motors).
	float actuator_min[4] = {0.f, 0.f, 0.f, 0.f};

	/// Actuator maximum values per motor (default 1.0).
	float actuator_max[4] = {1.f, 1.f, 1.f, 1.f};

	/// Slew rate limit per motor (0 = unlimited). Time in seconds from 0 to full.
	float slew_rate_limit[4] = {0.f, 0.f, 0.f, 0.f};

	/// Whether to normalize the RPY columns of the mix matrix.
	bool normalize_rpy{true};
};

/**
 * Input to the control allocator.
 * torque_sp comes from the rate controller (Phase 2).
 * thrust_sp comes from the rate controller (passthrough from attitude controller).
 */
struct ControlAllocatorInput {
	/// Normalized torque setpoint [-1, 1] for [roll, pitch, yaw].
	matrix::Vector3f torque_sp{0.f, 0.f, 0.f};

	/// Thrust setpoint. For standard quad: only z-component used (negative = upward in NED).
	matrix::Vector3f thrust_sp{0.f, 0.f, 0.f};

	/// Time step [s] from the RL simulation environment.
	float dt{0.004f};
};

/**
 * Output from the control allocator.
 */
struct ControlAllocatorOutput {
	/// Normalized motor commands [0, 1] for motors 0-3.
	float motors[4] = {0.f, 0.f, 0.f, 0.f};

	/// Number of motors.
	int num_motors{4};

	/// Per-axis saturation flags for rate controller anti-windup.
	/// True if the allocator couldn't fully allocate the demanded torque on that axis.
	matrix::Vector3<bool> saturation_positive;
	matrix::Vector3<bool> saturation_negative;

	/// Unallocated torque: control_sp - allocated_control (torque axes only).
	matrix::Vector3f unallocated_torque{0.f, 0.f, 0.f};

	/// Unallocated thrust: control_sp - allocated_control (thrust axes x, y, z; body FRD, normalized).
	matrix::Vector3f unallocated_thrust{0.f, 0.f, 0.f};
};

} // namespace px4_stack
