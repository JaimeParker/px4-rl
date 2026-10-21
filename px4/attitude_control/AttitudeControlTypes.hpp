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
 * @file AttitudeControlTypes.hpp
 *
 * Data structures for attitude controller extraction
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <cstdint>

namespace px4_stack {

/**
 * Configuration parameters for attitude controller
 */
struct AttitudeControlConfig {
	float roll_p{6.5f};           ///< Roll proportional gain [default: 6.5]
	float pitch_p{6.5f};          ///< Pitch proportional gain [default: 6.5]
	float yaw_p{2.8f};            ///< Yaw proportional gain [default: 2.8]
	float yaw_weight{0.4f};       ///< Yaw weight [0-1, default: 0.4]
	float max_roll_rate{3.84f};   ///< Max roll rate [rad/s, default: 220°/s = 3.84 rad/s]
	float max_pitch_rate{3.84f};  ///< Max pitch rate [rad/s, default: 220°/s = 3.84 rad/s]
	float max_yaw_rate{3.49f};    ///< Max yaw rate [rad/s, default: 200°/s = 3.49 rad/s]
};

/**
 * Attitude state input (current vehicle attitude)
 */
struct AttitudeState {
	uint64_t timestamp{0};        ///< Time since system start (microseconds)
	matrix::Quatf q;              ///< Current attitude quaternion (Hamilton: w,x,y,z)
	                              ///< Rotation from body FRD to earth NED frame

	PX4RL_HD AttitudeState() : q(1.f, 0.f, 0.f, 0.f) {}
};

/**
 * Attitude setpoint input (desired attitude from position controller)
 */
struct AttitudeSetpoint {
	uint64_t timestamp{0};           ///< Time since system start (microseconds)
	matrix::Quatf q_d;               ///< Desired attitude quaternion (Hamilton: w,x,y,z)
	float yaw_sp_move_rate{0.f};     ///< Feedforward yaw rate [rad/s]
	matrix::Vector3f thrust_body;    ///< Normalized thrust command in body FRD frame [-1,1]

	PX4RL_HD AttitudeSetpoint() : q_d(1.f, 0.f, 0.f, 0.f), thrust_body(0.f, 0.f, 0.f) {}
};

/**
 * Rate setpoint output (body angular rate commands to rate controller)
 */
struct RateSetpoint {
	uint64_t timestamp{0};           ///< Time since system start (microseconds)
	matrix::Vector3f rates;          ///< Body angular rates [rad/s] (roll, pitch, yaw)
	matrix::Vector3f thrust_body;    ///< Normalized thrust command in body FRD frame [-1,1]

	PX4RL_HD RateSetpoint() : rates(0.f, 0.f, 0.f), thrust_body(0.f, 0.f, 0.f) {}
};

} // namespace px4_stack
