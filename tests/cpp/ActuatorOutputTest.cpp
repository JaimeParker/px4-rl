/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ActuatorOutputTest.cpp
 *
 * GTest-based unit tests for the extracted actuator output stage (Phase 4).
 * Tests cover thrust curve inverse, PWM mapping, DShot mapping,
 * disarmed output, edge cases, and the full 4-phase pipeline.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cfloat>

#include <ActuatorOutputWrapper.hpp>
#include <ActuatorOutputTypes.hpp>
#include <ThrustCurve.hpp>

// Include all pipeline stages for end-to-end test
#include <ControlAllocatorWrapper.hpp>
#include <ControlAllocatorTypes.hpp>
#include <RateControllerWrapper.hpp>
#include <RateControlTypes.hpp>
#include <AttitudeControllerWrapper.hpp>
#include <AttitudeControlTypes.hpp>

using namespace matrix;
using namespace px4_stack;

// ============================================================================
// Thrust Curve Inverse Tests
// ============================================================================

TEST(ThrustCurveTest, LinearPassthrough)
{
	// When thrust_model_factor = 0, no transformation should occur
	float motors[4] = {0.0f, 0.25f, 0.5f, 1.0f};
	applyThrustCurveInverse(motors, 4, 0.0f);

	EXPECT_FLOAT_EQ(motors[0], 0.0f);
	EXPECT_FLOAT_EQ(motors[1], 0.25f);
	EXPECT_FLOAT_EQ(motors[2], 0.5f);
	EXPECT_FLOAT_EQ(motors[3], 1.0f);
}

TEST(ThrustCurveTest, ForwardInverseRoundTrip_F03)
{
	// Forward then inverse should recover the original signal
	const float f = 0.3f;
	const float signals[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};

	for (float x : signals) {
		float thrust = thrustCurveForward(x, f);
		float recovered[1] = {thrust};
		applyThrustCurveInverse(recovered, 1, f);
		EXPECT_NEAR(recovered[0], x, 1e-5f) << "Round-trip failed for x=" << x << " f=" << f;
	}
}

TEST(ThrustCurveTest, ForwardInverseRoundTrip_F10)
{
	// Edge case: f=1.0 means rel_thrust = x², inverse = sqrt(control)
	const float f = 1.0f;
	const float signals[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 1.0f};

	for (float x : signals) {
		float thrust = thrustCurveForward(x, f);
		float recovered[1] = {thrust};
		applyThrustCurveInverse(recovered, 1, f);
		EXPECT_NEAR(recovered[0], x, 1e-5f) << "Round-trip failed for x=" << x << " f=" << f;
	}
}

TEST(ThrustCurveTest, InverseIncreasesSignal)
{
	// For f > 0, the inverse should output values >= input
	// because the motor needs more signal to produce the desired thrust
	const float f = 0.3f;
	float motors[1] = {0.5f};
	applyThrustCurveInverse(motors, 1, f);
	EXPECT_GT(motors[0], 0.5f) << "Inverse should increase signal for f>0";
	EXPECT_LE(motors[0], 1.0f);
}

TEST(ThrustCurveTest, NegativesClamped)
{
	// Non-reversible: negatives should be clamped to 0
	float motors[2] = {-0.5f, -1.0f};
	applyThrustCurveInverse(motors, 2, 0.3f);
	EXPECT_FLOAT_EQ(motors[0], 0.0f);
	EXPECT_FLOAT_EQ(motors[1], 0.0f);
}

TEST(ThrustCurveTest, NaNHandled)
{
	// NaN inputs should be clamped to 0
	float motors[1] = {NAN};
	applyThrustCurveInverse(motors, 1, 0.3f);
	EXPECT_FLOAT_EQ(motors[0], 0.0f);
}

TEST(ThrustCurveTest, OverOneClamped)
{
	// Values above 1 should be clamped to 1
	float motors[1] = {1.5f};
	applyThrustCurveInverse(motors, 1, 0.0f);
	EXPECT_FLOAT_EQ(motors[0], 1.0f);
}

TEST(ThrustCurveTest, ForwardModelBoundaries)
{
	// Forward model at x=0 → 0, x=1 → 1 for any valid f
	for (float f : {0.0f, 0.3f, 0.5f, 1.0f}) {
		EXPECT_FLOAT_EQ(thrustCurveForward(0.0f, f), 0.0f) << "f=" << f;
		EXPECT_FLOAT_EQ(thrustCurveForward(1.0f, f), 1.0f) << "f=" << f;
	}
}

// ============================================================================
// PWM Mapping Tests
// ============================================================================

class PWMOutputTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		ActuatorOutputParams params;
		params.mode = OutputMode::PWM;
		params.thrust_model_factor = 0.0f; // linear for easier verification
		params.pwm_min = 1000;
		params.pwm_max = 2000;
		params.pwm_disarmed = 900;
		wrapper.setParams(params);
	}

	ActuatorOutputWrapper wrapper;
	ActuatorOutputInput input;
	ActuatorOutputOutput output;
};

TEST_F(PWMOutputTest, MotorZeroGivesPWMMin)
{
	// motor=0 → remap to -1 → interpolate to PWM_MIN=1000
	input.motors[0] = 0.0f;
	input.motors[1] = 0.0f;
	input.motors[2] = 0.0f;
	input.motors[3] = 0.0f;
	input.armed = true;

	wrapper.update(input, output);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(output.values[i], 1000) << "Motor " << i;
	}
}

TEST_F(PWMOutputTest, MotorOneGivesPWMMax)
{
	// motor=1 → remap to +1 → interpolate to PWM_MAX=2000
	input.motors[0] = 1.0f;
	input.motors[1] = 1.0f;
	input.motors[2] = 1.0f;
	input.motors[3] = 1.0f;
	input.armed = true;

	wrapper.update(input, output);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(output.values[i], 2000) << "Motor " << i;
	}
}

TEST_F(PWMOutputTest, MotorHalfGivesMidpoint)
{
	// motor=0.5 → remap to 0 → interpolate to (1000+2000)/2 = 1500
	input.motors[0] = 0.5f;
	input.motors[1] = 0.5f;
	input.motors[2] = 0.5f;
	input.motors[3] = 0.5f;
	input.armed = true;

	wrapper.update(input, output);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(output.values[i], 1500) << "Motor " << i;
	}
}

TEST_F(PWMOutputTest, MotorQuarterGives1250)
{
	// motor=0.25 → remap to -0.5 → interpolate to 1250
	input.motors[0] = 0.25f;
	input.armed = true;

	wrapper.update(input, output);
	EXPECT_EQ(output.values[0], 1250);
}

TEST_F(PWMOutputTest, DisarmedOutputsDisarmedValue)
{
	input.motors[0] = 0.5f;
	input.motors[1] = 0.8f;
	input.motors[2] = 0.3f;
	input.motors[3] = 1.0f;
	input.armed = false;

	wrapper.update(input, output);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(output.values[i], 900) << "Motor " << i << " should be disarmed (900)";
	}
}

TEST_F(PWMOutputTest, CustomPWMRange)
{
	// Test with different PWM range: 1100-1900
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.0f;
	params.pwm_min = 1100;
	params.pwm_max = 1900;
	params.pwm_disarmed = 950;
	wrapper.setParams(params);

	input.motors[0] = 0.0f;
	input.motors[1] = 0.5f;
	input.motors[2] = 1.0f;
	input.armed = true;

	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1100);
	EXPECT_EQ(output.values[1], 1500);
	EXPECT_EQ(output.values[2], 1900);
}

// ============================================================================
// PWM with Thrust Curve Tests
// ============================================================================

TEST(PWMThrustCurveTest, NonlinearThrustCurveIncreasesPWM)
{
	// With thrust_model_factor > 0, the same input thrust should produce
	// a higher PWM than linear, because the inverse maps to a higher motor signal
	ActuatorOutputWrapper wrapper_linear;
	ActuatorOutputWrapper wrapper_nonlinear;

	ActuatorOutputParams p_lin;
	p_lin.mode = OutputMode::PWM;
	p_lin.thrust_model_factor = 0.0f;
	p_lin.pwm_min = 1000;
	p_lin.pwm_max = 2000;

	ActuatorOutputParams p_nl = p_lin;
	p_nl.thrust_model_factor = 0.3f;

	wrapper_linear.setParams(p_lin);
	wrapper_nonlinear.setParams(p_nl);

	ActuatorOutputInput input;
	input.motors[0] = 0.5f;
	input.armed = true;

	ActuatorOutputOutput out_lin, out_nl;
	wrapper_linear.update(input, out_lin);
	wrapper_nonlinear.update(input, out_nl);

	// Nonlinear should give higher PWM for mid-range thrust
	EXPECT_GT(out_nl.values[0], out_lin.values[0])
		<< "Nonlinear curve should produce higher PWM for 50% thrust command";
}

TEST(PWMThrustCurveTest, FullThrustUnaffectedByCurve)
{
	// At motor=1.0, thrust curve inverse should still output 1.0 → PWM_MAX
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.35f; // Kopis2 value
	params.pwm_min = 1000;
	params.pwm_max = 2000;
	wrapper.setParams(params);

	ActuatorOutputInput input;
	input.motors[0] = 1.0f;
	input.armed = true;

	ActuatorOutputOutput output;
	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 2000);
}

TEST(PWMThrustCurveTest, ZeroThrustUnaffectedByCurve)
{
	// At motor=0.0, thrust curve inverse should still output 0.0 → PWM_MIN
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.35f;
	params.pwm_min = 1000;
	params.pwm_max = 2000;
	wrapper.setParams(params);

	ActuatorOutputInput input;
	input.motors[0] = 0.0f;
	input.armed = true;

	ActuatorOutputOutput output;
	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1000);
}

// ============================================================================
// DShot Mapping Tests
// ============================================================================

class DShotOutputTest : public ::testing::Test {
protected:
	void SetUp() override
	{
		ActuatorOutputParams params;
		params.mode = OutputMode::DShot;
		params.thrust_model_factor = 0.0f; // linear
		params.dshot_min_throttle = 1;
		params.dshot_max_throttle = 1999;
		params.dshot_disarmed = 0;
		params.dshot_min_fraction = 0.0f;
		wrapper.setParams(params);
	}

	ActuatorOutputWrapper wrapper;
	ActuatorOutputInput input;
	ActuatorOutputOutput output;
};

TEST_F(DShotOutputTest, MotorZeroGivesDShotMin)
{
	input.motors[0] = 0.0f;
	input.armed = true;

	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1); // dshot_min_throttle
}

TEST_F(DShotOutputTest, MotorOneGivesDShotMax)
{
	input.motors[0] = 1.0f;
	input.armed = true;

	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1999);
}

TEST_F(DShotOutputTest, DisarmedOutputsDShotDisarmed)
{
	input.motors[0] = 0.5f;
	input.motors[1] = 1.0f;
	input.armed = false;

	wrapper.update(input, output);

	for (int i = 0; i < 4; ++i) {
		EXPECT_EQ(output.values[i], 0) << "Motor " << i << " should be DShot disarmed (0)";
	}
}

TEST_F(DShotOutputTest, MinFractionRaisesFloor)
{
	// With dshot_min_fraction = 0.1, effective min = 1999 * 0.1 ≈ 200
	ActuatorOutputParams params;
	params.mode = OutputMode::DShot;
	params.thrust_model_factor = 0.0f;
	params.dshot_min_throttle = 1;
	params.dshot_max_throttle = 1999;
	params.dshot_disarmed = 0;
	params.dshot_min_fraction = 0.1f;
	wrapper.setParams(params);

	input.motors[0] = 0.0f;
	input.armed = true;

	wrapper.update(input, output);

	// effective_min = max(1, 1999 * 0.1) = max(1, 199.9) = 199.9 → 200
	EXPECT_GE(output.values[0], 199);
	EXPECT_LE(output.values[0], 201);
}

TEST_F(DShotOutputTest, MidpointScaling)
{
	// motor=0.5, no min fraction → (1 + 1999) / 2 = 1000
	input.motors[0] = 0.5f;
	input.armed = true;

	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1000);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EdgeCaseTest, NaNMotorInput)
{
	// NaN motor input should be safely handled (clamped to 0 by thrust curve)
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.0f;
	params.pwm_min = 1000;
	params.pwm_max = 2000;
	wrapper.setParams(params);

	ActuatorOutputInput input;
	input.motors[0] = NAN;
	input.armed = true;

	ActuatorOutputOutput output;
	wrapper.update(input, output);

	// NaN → clamped to 0 → motor=0 → PWM_MIN
	EXPECT_EQ(output.values[0], 1000);
}

TEST(EdgeCaseTest, NegativeMotorInput)
{
	// Negative motor input should be clamped to 0 → PWM_MIN
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.0f;
	params.pwm_min = 1000;
	params.pwm_max = 2000;
	wrapper.setParams(params);

	ActuatorOutputInput input;
	input.motors[0] = -0.5f;
	input.armed = true;

	ActuatorOutputOutput output;
	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 1000);
}

TEST(EdgeCaseTest, OverOneMotorInput)
{
	// motor > 1 should be clamped to 1 → PWM_MAX
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.mode = OutputMode::PWM;
	params.thrust_model_factor = 0.0f;
	params.pwm_min = 1000;
	params.pwm_max = 2000;
	wrapper.setParams(params);

	ActuatorOutputInput input;
	input.motors[0] = 1.5f;
	input.armed = true;

	ActuatorOutputOutput output;
	wrapper.update(input, output);

	EXPECT_EQ(output.values[0], 2000);
}

TEST(EdgeCaseTest, ParamsGetSet)
{
	ActuatorOutputWrapper wrapper;
	ActuatorOutputParams params;
	params.thrust_model_factor = 0.35f;
	params.pwm_min = 1100;
	params.pwm_max = 1900;
	params.pwm_disarmed = 950;
	wrapper.setParams(params);

	auto retrieved = wrapper.getParams();
	EXPECT_FLOAT_EQ(retrieved.thrust_model_factor, 0.35f);
	EXPECT_EQ(retrieved.pwm_min, 1100);
	EXPECT_EQ(retrieved.pwm_max, 1900);
	EXPECT_EQ(retrieved.pwm_disarmed, 950);
}

// ============================================================================
// Full Pipeline Test: Attitude → Rate → Allocator → Actuator Output → PWM
// ============================================================================

TEST(FullPipelineTest, AttitudeToRateToAllocatorToPWM)
{
	// Phase 1: Attitude control
	AttitudeControllerWrapper att_ctrl;
	AttitudeControlConfig att_config;
	att_config.roll_p = 6.5f;
	att_config.pitch_p = 6.5f;
	att_config.yaw_p = 2.8f;
	att_config.max_roll_rate = 220.f * (M_PI / 180.f);
	att_config.max_pitch_rate = 220.f * (M_PI / 180.f);
	att_config.max_yaw_rate = 200.f * (M_PI / 180.f);
	att_ctrl.setGains(att_config);

	AttitudeState att_state;
	att_state.q = Quatf(1.f, 0.f, 0.f, 0.f); // level

	AttitudeSetpoint att_sp;
	att_sp.q_d = Quatf(Eulerf(10.f * static_cast<float>(M_PI) / 180.f, 0.f, 0.f)); // ~10° roll
	att_sp.thrust_body = Vector3f(0.f, 0.f, -0.5f);

	RateSetpoint rate_sp = att_ctrl.update(att_state, att_sp);

	// Phase 2: Rate control
	RateControllerWrapper rate_ctrl;
	RateControlParams rate_params;
	rate_params.gain_p = Vector3f(0.15f, 0.15f, 0.2f);
	rate_params.gain_i = Vector3f(0.2f, 0.2f, 0.1f);
	rate_params.gain_d = Vector3f(0.003f, 0.003f, 0.0f);
	rate_params.gain_ff = Vector3f(0.f, 0.f, 0.f);
	rate_params.integrator_limit = Vector3f(0.3f, 0.3f, 0.3f);
	rate_ctrl.setParams(rate_params);

	RateControlInput rate_in;
	rate_in.rates_actual = Vector3f(0.f, 0.f, 0.f);   // no rotation yet
	rate_in.rates_sp = rate_sp.rates;
	rate_in.angular_accel = Vector3f(0.f, 0.f, 0.f);
	rate_in.dt = 0.004f;
	rate_in.landed = false;

	RateControlOutput rate_out;
	rate_ctrl.update(rate_in, rate_out);

	// Phase 3: Control allocator
	ControlAllocatorWrapper alloc;

	ControlAllocatorInput alloc_in;
	alloc_in.torque_sp = rate_out.torque_sp;
	alloc_in.thrust_sp = rate_sp.thrust_body;
	alloc_in.dt = 0.004f;

	ControlAllocatorOutput alloc_out;
	alloc.update(alloc_in, alloc_out);

	// Phase 4: Actuator output → PWM
	ActuatorOutputWrapper actuator;
	ActuatorOutputParams act_params;
	act_params.mode = OutputMode::PWM;
	act_params.thrust_model_factor = 0.3f; // typical racer
	act_params.pwm_min = 1000;
	act_params.pwm_max = 2000;
	act_params.pwm_disarmed = 900;
	actuator.setParams(act_params);

	ActuatorOutputInput act_in;
	act_in.armed = true;

	for (int i = 0; i < 4; ++i) {
		act_in.motors[i] = alloc_out.motors[i];
	}

	ActuatorOutputOutput act_out;
	actuator.update(act_in, act_out);

	// Verify all outputs are valid PWM values
	for (int i = 0; i < 4; ++i) {
		EXPECT_GE(act_out.values[i], 1000) << "Motor " << i << " below PWM_MIN";
		EXPECT_LE(act_out.values[i], 2000) << "Motor " << i << " above PWM_MAX";
	}

	// With a ~10° roll command, motors on one side should spin faster
	int max_pwm = 0, min_pwm = 2000;

	for (int i = 0; i < 4; ++i) {
		if (act_out.values[i] > max_pwm) { max_pwm = act_out.values[i]; }

		if (act_out.values[i] < min_pwm) { min_pwm = act_out.values[i]; }
	}

	EXPECT_GT(max_pwm - min_pwm, 10)
		<< "Expected PWM asymmetry from roll command, got uniform values";
}

TEST(FullPipelineTest, AttitudeToRateToAllocatorToDShot)
{
	// Same pipeline but with DShot output
	AttitudeControllerWrapper att_ctrl;
	AttitudeControlConfig att_config;
	att_config.roll_p = 6.5f;
	att_config.pitch_p = 6.5f;
	att_config.yaw_p = 2.8f;
	att_config.max_roll_rate = 220.f * (M_PI / 180.f);
	att_config.max_pitch_rate = 220.f * (M_PI / 180.f);
	att_config.max_yaw_rate = 200.f * (M_PI / 180.f);
	att_ctrl.setGains(att_config);

	AttitudeState att_state;
	att_state.q = Quatf(1.f, 0.f, 0.f, 0.f);

	AttitudeSetpoint att_sp;
	att_sp.q_d = Quatf(Eulerf(10.f * static_cast<float>(M_PI) / 180.f, 0.f, 0.f));
	att_sp.thrust_body = Vector3f(0.f, 0.f, -0.5f);

	RateSetpoint rate_sp = att_ctrl.update(att_state, att_sp);

	RateControllerWrapper rate_ctrl;
	RateControlParams rate_params;
	rate_params.gain_p = Vector3f(0.15f, 0.15f, 0.2f);
	rate_params.gain_i = Vector3f(0.2f, 0.2f, 0.1f);
	rate_params.gain_d = Vector3f(0.003f, 0.003f, 0.0f);
	rate_params.gain_ff = Vector3f(0.f, 0.f, 0.f);
	rate_params.integrator_limit = Vector3f(0.3f, 0.3f, 0.3f);
	rate_ctrl.setParams(rate_params);

	RateControlInput rate_in;
	rate_in.rates_actual = Vector3f(0.f, 0.f, 0.f);
	rate_in.rates_sp = rate_sp.rates;
	rate_in.angular_accel = Vector3f(0.f, 0.f, 0.f);
	rate_in.dt = 0.004f;
	rate_in.landed = false;

	RateControlOutput rate_out;
	rate_ctrl.update(rate_in, rate_out);

	ControlAllocatorWrapper alloc;

	ControlAllocatorInput alloc_in;
	alloc_in.torque_sp = rate_out.torque_sp;
	alloc_in.thrust_sp = rate_sp.thrust_body;
	alloc_in.dt = 0.004f;

	ControlAllocatorOutput alloc_out;
	alloc.update(alloc_in, alloc_out);

	// Phase 4: DShot output
	ActuatorOutputWrapper actuator;
	ActuatorOutputParams act_params;
	act_params.mode = OutputMode::DShot;
	act_params.thrust_model_factor = 0.3f;
	act_params.dshot_min_throttle = 1;
	act_params.dshot_max_throttle = 1999;
	act_params.dshot_disarmed = 0;
	act_params.dshot_min_fraction = 0.0f;
	actuator.setParams(act_params);

	ActuatorOutputInput act_in;
	act_in.armed = true;

	for (int i = 0; i < 4; ++i) {
		act_in.motors[i] = alloc_out.motors[i];
	}

	ActuatorOutputOutput act_out;
	actuator.update(act_in, act_out);

	// Verify all outputs are valid DShot values
	for (int i = 0; i < 4; ++i) {
		EXPECT_GE(act_out.values[i], 1) << "Motor " << i << " below DShot min";
		EXPECT_LE(act_out.values[i], 1999) << "Motor " << i << " above DShot max";
	}
}
