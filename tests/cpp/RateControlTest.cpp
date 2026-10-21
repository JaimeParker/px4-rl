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
 * @file RateControlTest.cpp
 *
 * GTest-based unit tests for extracted rate controller.
 * Tests cover the core RateControl class and the RateControllerWrapper.
 */

#include <gtest/gtest.h>
#include <cmath>

#include <RateControl.hpp>
#include <RateControllerWrapper.hpp>
#include <AttitudeControllerWrapper.hpp>

using namespace matrix;
using namespace px4_stack;

// ============================================================================
// Core RateControl class tests
// ============================================================================

TEST(RateControlCoreTest, AllZeroCase)
{
	// Matches PX4's original rate_control_test.cpp
	RateControl rate_control;
	Vector3f torque = rate_control.update(Vector3f(), Vector3f(), Vector3f(), 0.f, false);
	EXPECT_EQ(torque, Vector3f());
}

TEST(RateControlCoreTest, ProportionalResponse)
{
	// P-only: torque = P * (rate_sp - rate)
	RateControl rc;
	const Vector3f P(0.15f, 0.15f, 0.2f);
	rc.setPidGains(P, Vector3f(), Vector3f()); // I=0, D=0
	rc.setIntegratorLimit(Vector3f(0.3f, 0.3f, 0.3f));

	const Vector3f rate(0.f, 0.f, 0.f);
	const Vector3f rate_sp(1.0f, 0.5f, -0.3f);
	const Vector3f angular_accel(0.f, 0.f, 0.f);
	const float dt = 0.004f;

	Vector3f torque = rc.update(rate, rate_sp, angular_accel, dt, false);

	// Expected: P * rate_error = P * rate_sp (since rate=0)
	Vector3f expected = P.emult(rate_sp);
	EXPECT_NEAR(torque(0), expected(0), 1e-6f);
	EXPECT_NEAR(torque(1), expected(1), 1e-6f);
	EXPECT_NEAR(torque(2), expected(2), 1e-6f);
}

TEST(RateControlCoreTest, DerivativeDamping)
{
	// D-term subtracts: torque -= D * angular_accel
	RateControl rc;
	const Vector3f D(0.003f, 0.003f, 0.001f);
	rc.setPidGains(Vector3f(), Vector3f(), D); // P=0, I=0
	rc.setIntegratorLimit(Vector3f(0.3f, 0.3f, 0.3f));

	const Vector3f angular_accel(10.f, -5.f, 20.f);
	Vector3f torque = rc.update(Vector3f(), Vector3f(), angular_accel, 0.004f, false);

	// Expected: -D * angular_accel (P term is zero, I is zero, FF is zero)
	EXPECT_NEAR(torque(0), -D(0) * angular_accel(0), 1e-6f);
	EXPECT_NEAR(torque(1), -D(1) * angular_accel(1), 1e-6f);
	EXPECT_NEAR(torque(2), -D(2) * angular_accel(2), 1e-6f);
}

TEST(RateControlCoreTest, FeedForward)
{
	// FF adds: torque += FF * rate_sp
	RateControl rc;
	const Vector3f FF(0.1f, 0.1f, 0.05f);
	rc.setPidGains(Vector3f(), Vector3f(), Vector3f()); // P=0, I=0, D=0
	rc.setFeedForwardGain(FF);
	rc.setIntegratorLimit(Vector3f(0.3f, 0.3f, 0.3f));

	const Vector3f rate_sp(2.0f, -1.0f, 0.5f);
	Vector3f torque = rc.update(Vector3f(), rate_sp, Vector3f(), 0.004f, false);

	// Expected: FF * rate_sp
	EXPECT_NEAR(torque(0), FF(0) * rate_sp(0), 1e-6f);
	EXPECT_NEAR(torque(1), FF(1) * rate_sp(1), 1e-6f);
	EXPECT_NEAR(torque(2), FF(2) * rate_sp(2), 1e-6f);
}

TEST(RateControlCoreTest, IntegratorAccumulation)
{
	// Constant rate error over N steps → integrator builds up
	RateControl rc;
	const Vector3f I(0.2f, 0.2f, 0.1f);
	rc.setPidGains(Vector3f(), I, Vector3f()); // P=0, D=0
	rc.setIntegratorLimit(Vector3f(1.0f, 1.0f, 1.0f)); // high limit to avoid clamping

	const Vector3f rate_sp(1.0f, 0.f, 0.f); // 1 rad/s roll error
	const float dt = 0.004f;
	const int N = 10;

	// Run N steps
	Vector3f torque;
	for (int i = 0; i < N; i++) {
		torque = rc.update(Vector3f(), rate_sp, Vector3f(), dt, false);
	}

	// Check integrator state via status
	RateCtrlStatus status;
	rc.getRateControlStatus(status);

	// Expected integrator (small error, i_factor slightly < 1 due to nonlinear reduction):
	// i_factor = max(0, 1 - (error/radians(400))²) ≈ 1 - (1.0/6.98)² ≈ 0.9795
	// integral ≈ i_factor * I_gain * rate_error * dt * N
	float i_factor = 1.0f / math::radians(400.f);
	i_factor = 1.f - i_factor * i_factor; // ≈ 0.9795
	float expected_integ = i_factor * I(0) * 1.0f * dt * N;
	EXPECT_NEAR(status.rollspeed_integ, expected_integ, 1e-5f);
	EXPECT_NEAR(status.pitchspeed_integ, 0.f, 1e-6f);
	EXPECT_NEAR(status.yawspeed_integ, 0.f, 1e-6f);
}

TEST(RateControlCoreTest, IntegratorClamping)
{
	// Integrator must not exceed the limit
	RateControl rc;
	const float int_lim = 0.05f;
	rc.setPidGains(Vector3f(), Vector3f(0.2f, 0.2f, 0.1f), Vector3f());
	rc.setIntegratorLimit(Vector3f(int_lim, int_lim, int_lim));

	const Vector3f rate_sp(1.0f, 1.0f, 1.0f);
	const float dt = 0.004f;

	// Run many steps to saturate integrator
	for (int i = 0; i < 1000; i++) {
		rc.update(Vector3f(), rate_sp, Vector3f(), dt, false);
	}

	RateCtrlStatus status;
	rc.getRateControlStatus(status);

	EXPECT_LE(std::abs(status.rollspeed_integ), int_lim + 1e-6f);
	EXPECT_LE(std::abs(status.pitchspeed_integ), int_lim + 1e-6f);
	EXPECT_LE(std::abs(status.yawspeed_integ), int_lim + 1e-6f);
}

TEST(RateControlCoreTest, IntegratorAntiWindupPositiveSaturation)
{
	// When positive saturation is reported, integrator should not increase for positive error
	RateControl rc;
	rc.setPidGains(Vector3f(), Vector3f(0.2f, 0.2f, 0.1f), Vector3f());
	rc.setIntegratorLimit(Vector3f(1.0f, 1.0f, 1.0f));

	const Vector3f rate_sp(1.0f, 0.f, 0.f); // positive roll error
	const float dt = 0.004f;

	// First accumulate some positive integrator
	for (int i = 0; i < 50; i++) {
		rc.update(Vector3f(), rate_sp, Vector3f(), dt, false);
	}

	RateCtrlStatus status_before;
	rc.getRateControlStatus(status_before);
	EXPECT_GT(status_before.rollspeed_integ, 0.f);

	// Now set positive saturation on roll axis
	rc.setPositiveSaturationFlag(0, true);

	float integ_before = status_before.rollspeed_integ;

	// Run more steps with positive error — integrator should not increase
	for (int i = 0; i < 50; i++) {
		rc.update(Vector3f(), rate_sp, Vector3f(), dt, false);
	}

	RateCtrlStatus status_after;
	rc.getRateControlStatus(status_after);

	// Integrator should not have increased (may have decreased slightly)
	EXPECT_LE(status_after.rollspeed_integ, integ_before + 1e-6f);
}

TEST(RateControlCoreTest, IntegratorFrozenWhenLanded)
{
	RateControl rc;
	rc.setPidGains(Vector3f(), Vector3f(0.2f, 0.2f, 0.1f), Vector3f());
	rc.setIntegratorLimit(Vector3f(1.0f, 1.0f, 1.0f));

	const Vector3f rate_sp(1.0f, 1.0f, 1.0f);
	const float dt = 0.004f;

	// Run with landed=true
	for (int i = 0; i < 100; i++) {
		rc.update(Vector3f(), rate_sp, Vector3f(), dt, true);
	}

	RateCtrlStatus status;
	rc.getRateControlStatus(status);

	EXPECT_FLOAT_EQ(status.rollspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.pitchspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.yawspeed_integ, 0.f);
}

TEST(RateControlCoreTest, ResetIntegral)
{
	RateControl rc;
	rc.setPidGains(Vector3f(), Vector3f(0.2f, 0.2f, 0.1f), Vector3f());
	rc.setIntegratorLimit(Vector3f(1.0f, 1.0f, 1.0f));

	// Accumulate some integrator
	for (int i = 0; i < 50; i++) {
		rc.update(Vector3f(), Vector3f(1.f, 1.f, 1.f), Vector3f(), 0.004f, false);
	}

	// Reset all
	rc.resetIntegral();
	RateCtrlStatus status;
	rc.getRateControlStatus(status);
	EXPECT_FLOAT_EQ(status.rollspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.pitchspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.yawspeed_integ, 0.f);

	// Accumulate again, reset single axis
	for (int i = 0; i < 50; i++) {
		rc.update(Vector3f(), Vector3f(1.f, 1.f, 1.f), Vector3f(), 0.004f, false);
	}
	rc.resetIntegral(0); // reset roll only
	rc.getRateControlStatus(status);
	EXPECT_FLOAT_EQ(status.rollspeed_integ, 0.f);
	EXPECT_GT(status.pitchspeed_integ, 0.f);
}

TEST(RateControlCoreTest, CombinedPIDFF)
{
	// Full PID+FF: torque = P*error + I_integral - D*accel + FF*rate_sp
	RateControl rc;
	const Vector3f P(0.15f, 0.15f, 0.2f);
	const Vector3f I(0.0f, 0.0f, 0.0f); // zero I for predictable single-step test
	const Vector3f D(0.003f, 0.003f, 0.0f);
	const Vector3f FF(0.05f, 0.05f, 0.02f);

	rc.setPidGains(P, I, D);
	rc.setFeedForwardGain(FF);
	rc.setIntegratorLimit(Vector3f(0.3f, 0.3f, 0.3f));

	const Vector3f rate(0.5f, -0.2f, 0.1f);
	const Vector3f rate_sp(1.5f, 0.3f, -0.5f);
	const Vector3f angular_accel(2.0f, -1.0f, 0.5f);
	const float dt = 0.004f;

	Vector3f torque = rc.update(rate, rate_sp, angular_accel, dt, false);

	Vector3f rate_error = rate_sp - rate;
	Vector3f expected = P.emult(rate_error) - D.emult(angular_accel) + FF.emult(rate_sp);
	// integrator is zero on first call

	EXPECT_NEAR(torque(0), expected(0), 1e-5f);
	EXPECT_NEAR(torque(1), expected(1), 1e-5f);
	EXPECT_NEAR(torque(2), expected(2), 1e-5f);
}

// ============================================================================
// RateControllerWrapper tests
// ============================================================================

TEST(RateControlWrapperTest, DefaultGainsProduceOutput)
{
	RateControllerWrapper wrapper;

	RateControlInput input;
	input.rates_sp = Vector3f(1.0f, 0.f, 0.f);
	input.rates_actual = Vector3f(0.f, 0.f, 0.f);
	input.angular_accel = Vector3f(0.f, 0.f, 0.f);
	input.thrust_body = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;
	input.landed = false;

	RateControlOutput output;
	wrapper.update(input, output);

	// With default P=0.15 and 1 rad/s error, expect ~0.15 roll torque
	EXPECT_NEAR(output.torque_sp(0), 0.15f, 0.01f);
	EXPECT_NEAR(output.torque_sp(1), 0.f, 1e-6f);
	// Thrust passthrough
	EXPECT_FLOAT_EQ(output.thrust_body(2), -0.5f);
}

TEST(RateControlWrapperTest, KScalingApplied)
{
	// Verify that K multiplier is applied to gains
	RateControllerWrapper wrapper;

	RateControlParams params;
	params.gain_p = Vector3f(0.15f, 0.15f, 0.2f);
	params.gain_i = Vector3f(0.f, 0.f, 0.f); // disable I for predictable test
	params.gain_d = Vector3f(0.f, 0.f, 0.f);
	params.gain_k = Vector3f(2.0f, 1.0f, 1.0f); // 2x on roll
	wrapper.setParams(params);

	RateControlInput input;
	input.rates_sp = Vector3f(1.0f, 1.0f, 0.f);
	input.rates_actual = Vector3f(0.f, 0.f, 0.f);
	input.dt = 0.004f;

	RateControlOutput output;
	wrapper.update(input, output);

	// Roll: K=2 * P=0.15 * error=1.0 = 0.30
	// Pitch: K=1 * P=0.15 * error=1.0 = 0.15
	EXPECT_NEAR(output.torque_sp(0), 0.30f, 1e-5f);
	EXPECT_NEAR(output.torque_sp(1), 0.15f, 1e-5f);
}

TEST(RateControlWrapperTest, YawLPFSmooths)
{
	// Yaw LPF should smooth a step input over multiple calls
	RateControllerWrapper wrapper;

	RateControlParams params;
	params.gain_p = Vector3f(0.f, 0.f, 0.2f);
	params.gain_i = Vector3f(0.f, 0.f, 0.f);
	params.gain_d = Vector3f(0.f, 0.f, 0.f);
	params.yaw_torque_lpf_cutoff_freq = 2.0f; // 2 Hz
	wrapper.setParams(params);

	RateControlInput input;
	input.rates_sp = Vector3f(0.f, 0.f, 1.0f); // step input
	input.dt = 0.004f; // 250 Hz

	// First output should be smaller than steady-state (filtered)
	RateControlOutput output1;
	wrapper.update(input, output1);
	float first_yaw = output1.torque_sp(2);

	// Run many more steps
	RateControlOutput output;
	for (int i = 0; i < 500; i++) {
		wrapper.update(input, output);
	}
	float converged_yaw = output.torque_sp(2);

	// The unfiltered P-only value would be 0.2 * 1.0 = 0.2
	// First filtered sample should be less than converged
	EXPECT_LT(std::abs(first_yaw), std::abs(converged_yaw) + 1e-6f);

	// Converged value should be close to the P-only value
	EXPECT_NEAR(converged_yaw, 0.2f, 0.01f);
}

TEST(RateControlWrapperTest, ResetIntegralAndStatus)
{
	RateControllerWrapper wrapper;

	RateControlInput input;
	input.rates_sp = Vector3f(1.0f, 1.0f, 1.0f);
	input.dt = 0.004f;

	RateControlOutput output;
	for (int i = 0; i < 100; i++) {
		wrapper.update(input, output);
	}

	RateCtrlStatus status;
	wrapper.getStatus(status);
	EXPECT_GT(status.rollspeed_integ, 0.f);

	wrapper.resetIntegral();
	wrapper.getStatus(status);
	EXPECT_FLOAT_EQ(status.rollspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.pitchspeed_integ, 0.f);
	EXPECT_FLOAT_EQ(status.yawspeed_integ, 0.f);
}

// ============================================================================
// End-to-end pipeline test: Attitude → Rate → Torque
// ============================================================================

TEST(PipelineTest, AttitudeToRateToTorque)
{
	// Full pipeline: attitude error → rate setpoint → torque output
	AttitudeControllerWrapper att_ctrl;
	RateControllerWrapper rate_ctrl;

	// 15 degree roll error
	const float roll_angle = 15.0f * M_PI / 180.0f;

	AttitudeState state;
	state.q = Quatf(1.f, 0.f, 0.f, 0.f); // level

	AttitudeSetpoint att_sp;
	att_sp.q_d = Quatf(cosf(roll_angle / 2.f), sinf(roll_angle / 2.f), 0.f, 0.f);
	att_sp.thrust_body = Vector3f(0.f, 0.f, -0.5f);

	// Step 1: Attitude controller produces rate setpoint
	RateSetpoint rate_sp = att_ctrl.update(state, att_sp);

	EXPECT_GT(rate_sp.rates(0), 0.f); // positive roll rate to correct error
	EXPECT_NEAR(rate_sp.rates(1), 0.f, 0.01f); // no pitch
	EXPECT_NEAR(rate_sp.rates(2), 0.f, 0.01f); // no yaw

	// Step 2: Feed rate setpoint into rate controller
	RateControlInput rc_input;
	rc_input.rates_sp = rate_sp.rates;
	rc_input.rates_actual = Vector3f(0.f, 0.f, 0.f); // currently not rotating
	rc_input.angular_accel = Vector3f(0.f, 0.f, 0.f);
	rc_input.thrust_body = rate_sp.thrust_body;
	rc_input.dt = 0.004f;
	rc_input.landed = false;

	RateControlOutput rc_output;
	rate_ctrl.update(rc_input, rc_output);

	// Torque should be positive on roll to correct the error
	EXPECT_GT(rc_output.torque_sp(0), 0.f);
	EXPECT_NEAR(rc_output.torque_sp(1), 0.f, 0.01f);

	// Thrust passthrough
	EXPECT_FLOAT_EQ(rc_output.thrust_body(2), -0.5f);

	// Verify reasonable magnitude:
	// rate_sp.rates(0) ≈ 1.70 (from attitude test), P=0.15 → torque ≈ 0.255
	EXPECT_GT(rc_output.torque_sp(0), 0.1f);
	EXPECT_LT(rc_output.torque_sp(0), 1.0f);
}

TEST(PipelineTest, MultiStepConvergence)
{
	// Simulate several steps of the attitude→rate pipeline
	// Rate error should decrease over time as rate approaches setpoint
	AttitudeControllerWrapper att_ctrl;
	RateControllerWrapper rate_ctrl;

	const float roll_angle = 30.0f * M_PI / 180.0f;
	const float dt = 0.004f;

	AttitudeState state;
	state.q = Quatf(1.f, 0.f, 0.f, 0.f);

	AttitudeSetpoint att_sp;
	att_sp.q_d = Quatf(cosf(roll_angle / 2.f), sinf(roll_angle / 2.f), 0.f, 0.f);
	att_sp.thrust_body = Vector3f(0.f, 0.f, -0.5f);

	RateSetpoint rate_sp = att_ctrl.update(state, att_sp);

	// Simulate rate controller with a "virtual" angular rate that follows the torque
	// (simplified: assume angular rate increases proportionally to torque * dt * some_inertia_inv)
	Vector3f current_rate(0.f, 0.f, 0.f);
	Vector3f prev_rate(0.f, 0.f, 0.f);
	const float inertia_inv = 50.f; // 1/J simplified

	for (int i = 0; i < 50; i++) {
		RateControlInput input;
		input.rates_sp = rate_sp.rates;
		input.rates_actual = current_rate;
		input.angular_accel = (current_rate - prev_rate) / dt;
		input.thrust_body = att_sp.thrust_body;
		input.dt = dt;
		input.landed = false;

		RateControlOutput output;
		rate_ctrl.update(input, output);

		// Simple Euler integration of angular velocity
		prev_rate = current_rate;
		current_rate = current_rate + output.torque_sp * inertia_inv * dt;
	}

	// After 50 steps, the rate should have approached the setpoint
	float rate_error = (rate_sp.rates - current_rate).norm();
	EXPECT_LT(rate_error, rate_sp.rates.norm()); // error should be less than initial setpoint magnitude
}
