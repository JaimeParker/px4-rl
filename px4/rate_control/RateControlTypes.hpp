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
 * @file RateControlTypes.hpp
 *
 * Data structures for rate controller extraction.
 * Replaces uORB types and PX4 parameter system with plain C++ structs.
 *
 * Default values sourced from PX4 mc_rate_control_params.c
 */

#pragma once

#include <matrix/matrix/math.hpp>
#include <cstdint>

namespace px4_stack {

/**
 * Replacement for uORB rate_ctrl_status_s
 * Used to expose integrator states for debugging/logging.
 */
struct RateCtrlStatus {
	float rollspeed_integ{0.f};
	float pitchspeed_integ{0.f};
	float yawspeed_integ{0.f};
};

/**
 * Configuration parameters for rate controller.
 * Replaces PX4 params: MC_ROLLRATE_*, MC_PITCHRATE_*, MC_YAWRATE_*, MC_*_INT_LIM
 *
 * PX4 uses an "ideal form" PID: effective_gain = K * gain
 * With K=1.0 (default), gains are used directly (parallel form).
 */
struct RateControlParams {
	// Per-axis PID gains (parallel form, before K scaling)
	matrix::Vector3f gain_p{0.15f, 0.15f, 0.2f};    ///< Proportional gains [roll, pitch, yaw]
	matrix::Vector3f gain_i{0.2f, 0.2f, 0.1f};      ///< Integral gains
	matrix::Vector3f gain_d{0.003f, 0.003f, 0.0f};   ///< Derivative gains
	matrix::Vector3f gain_ff{0.0f, 0.0f, 0.0f};      ///< Feed-forward gains

	// Global gain multiplier per axis (ideal form PID: effective = K * gain)
	matrix::Vector3f gain_k{1.0f, 1.0f, 1.0f};      ///< MC_ROLLRATE_K, MC_PITCHRATE_K, MC_YAWRATE_K

	// Integrator limits (absolute value per axis)
	matrix::Vector3f integrator_limit{0.30f, 0.30f, 0.30f}; ///< MC_RR_INT_LIM, MC_PR_INT_LIM, MC_YR_INT_LIM

	// Yaw torque output low-pass filter cutoff frequency [Hz]
	// Applied after the PID computation, only on the yaw axis.
	float yaw_torque_lpf_cutoff_freq{2.0f};          ///< MC_YAW_TQ_CUTOFF [Hz]
};

/**
 * Rate controller input
 */
struct RateControlInput {
	matrix::Vector3f rates_sp;       ///< Desired body angular rates [rad/s] (from attitude controller)
	matrix::Vector3f rates_actual;   ///< Current body angular rates [rad/s] (from sim gyro)
	matrix::Vector3f angular_accel;  ///< Body angular acceleration [rad/s²] (from sim or finite-diff of gyro)
	matrix::Vector3f thrust_body;    ///< Normalized thrust in body FRD frame [-1,1] (passthrough)
	float dt{0.f};                   ///< Time step [s] provided by the RL simulation
	bool landed{false};              ///< If true, integrator is frozen

	PX4RL_HD RateControlInput()
		: rates_sp(0.f, 0.f, 0.f)
		, rates_actual(0.f, 0.f, 0.f)
		, angular_accel(0.f, 0.f, 0.f)
		, thrust_body(0.f, 0.f, 0.f) {}
};

/**
 * Rate controller output
 */
struct RateControlOutput {
	matrix::Vector3f torque_sp;      ///< Normalized torque setpoint [-1,1] (roll, pitch, yaw)
	matrix::Vector3f thrust_body;    ///< Normalized thrust passthrough in body FRD frame [-1,1]

	PX4RL_HD RateControlOutput()
		: torque_sp(0.f, 0.f, 0.f)
		, thrust_body(0.f, 0.f, 0.f) {}
};

} // namespace px4_stack
