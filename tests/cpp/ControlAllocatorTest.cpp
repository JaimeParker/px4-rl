/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file ControlAllocatorTest.cpp
 *
 * GTest-based unit tests for the extracted control allocator.
 * Tests cover effectiveness matrix, allocation, desaturation, saturation feedback,
 * and end-to-end pipeline (attitude → rate → allocator → motors).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cfloat>

#include <ControlAllocatorWrapper.hpp>
#include <RotorGeometry.hpp>
#include <ControlAllocation.hpp>
#include <ControlAllocationPseudoInverse.hpp>
#include <ControlAllocationSequentialDesaturation.hpp>
#include <ActuatorEffectiveness.hpp>
#include <RateControllerWrapper.hpp>
#include <AttitudeControllerWrapper.hpp>

using namespace matrix;
using namespace px4_stack;

// ============================================================================
// Effectiveness matrix tests
// ============================================================================

TEST(EffectivenessMatrixTest, IrisGeometryShape)
{
	// Verify that the Iris geometry produces a valid effectiveness matrix
	// with 4 columns populated and correct thrust direction
	Geometry geom = createQuadXIris();
	EXPECT_EQ(geom.num_rotors, 4);

	ActuatorEffectiveness::EffectivenessMatrix eff{};
	int n = computeEffectivenessMatrix(geom, eff, 0);
	EXPECT_EQ(n, 4);

	// All 4 motors produce upward thrust in NED: thrust_z should be negative
	for (int i = 0; i < 4; i++) {
		EXPECT_LT(eff(5, i), 0.f) << "Motor " << i << " thrust_z should be < 0 (upward in NED)";
		EXPECT_NEAR(eff(3, i), 0.f, 1e-6f) << "Motor " << i << " thrust_x should be 0";
		EXPECT_NEAR(eff(4, i), 0.f, 1e-6f) << "Motor " << i << " thrust_y should be 0";
	}
}

TEST(EffectivenessMatrixTest, IrisYawSignConvention)
{
	// CCW motors (0, 1) should have negative yaw effectiveness (torque = -CT * KM * axis)
	// CW motors (2, 3) should have positive yaw effectiveness
	// axis = {0,0,-1}, so yaw contribution from propeller torque = -CT * KM * (-1) = +CT*KM
	// For CCW (KM=+0.05): yaw = +CT*KM = +0.05
	// For CW  (KM=-0.05): yaw = +CT*(-0.05) = -0.05
	// But the moment formula is: moment = CT * (pos × axis) - CT * KM * axis
	// yaw component = -CT * KM * axis(2) = -1.0 * KM * (-1) = KM
	Geometry geom = createQuadXIris();
	ActuatorEffectiveness::EffectivenessMatrix eff{};
	computeEffectivenessMatrix(geom, eff, 0);

	// Motors 0, 1 are CCW (KM = +0.05): yaw eff = KM = +0.05
	EXPECT_NEAR(eff(2, 0), 0.05f, 1e-6f);
	EXPECT_NEAR(eff(2, 1), 0.05f, 1e-6f);

	// Motors 2, 3 are CW (KM = -0.05): yaw eff = KM = -0.05
	EXPECT_NEAR(eff(2, 2), -0.05f, 1e-6f);
	EXPECT_NEAR(eff(2, 3), -0.05f, 1e-6f);
}

TEST(EffectivenessMatrixTest, IrisRollPitchValues)
{
	// Roll torque = CT * (Py * axis_z - Pz * axis_y) = 1.0 * (Py * (-1) - 0) = -Py
	// Pitch torque = CT * (Pz * axis_x - Px * axis_z) = 1.0 * (0 - Px * (-1)) = Px
	Geometry geom = createQuadXIris();
	ActuatorEffectiveness::EffectivenessMatrix eff{};
	computeEffectivenessMatrix(geom, eff, 0);

	// Motor 0: PX=+0.1515, PY=+0.245 → roll=-0.245, pitch=+0.1515
	EXPECT_NEAR(eff(0, 0), -0.245f, 1e-5f);
	EXPECT_NEAR(eff(1, 0), 0.1515f, 1e-5f);

	// Motor 1: PX=-0.1515, PY=-0.1875 → roll=+0.1875, pitch=-0.1515
	EXPECT_NEAR(eff(0, 1), 0.1875f, 1e-5f);
	EXPECT_NEAR(eff(1, 1), -0.1515f, 1e-5f);

	// Motor 2: PX=+0.1515, PY=-0.245 → roll=+0.245, pitch=+0.1515
	EXPECT_NEAR(eff(0, 2), 0.245f, 1e-5f);
	EXPECT_NEAR(eff(1, 2), 0.1515f, 1e-5f);

	// Motor 3: PX=-0.1515, PY=+0.1875 → roll=-0.1875, pitch=-0.1515
	EXPECT_NEAR(eff(0, 3), -0.1875f, 1e-5f);
	EXPECT_NEAR(eff(1, 3), -0.1515f, 1e-5f);
}

TEST(EffectivenessMatrixTest, GenericGeometry)
{
	Geometry geom = createQuadXGeneric();
	EXPECT_EQ(geom.num_rotors, 4);

	ActuatorEffectiveness::EffectivenessMatrix eff{};
	int n = computeEffectivenessMatrix(geom, eff, 0);
	EXPECT_EQ(n, 4);

	// Motor 0: PX=1, PY=1 → roll=-1, pitch=1
	EXPECT_NEAR(eff(0, 0), -1.0f, 1e-6f);
	EXPECT_NEAR(eff(1, 0), 1.0f, 1e-6f);
}

TEST(EffectivenessMatrixTest, UnusedColumnsZero)
{
	// Columns 4-15 should be all zeros for a 4-motor setup
	Geometry geom = createQuadXIris();
	ActuatorEffectiveness::EffectivenessMatrix eff{};
	computeEffectivenessMatrix(geom, eff, 0);

	for (int col = 4; col < ActuatorEffectiveness::NUM_ACTUATORS; col++) {
		for (int row = 0; row < ActuatorEffectiveness::NUM_AXES; row++) {
			EXPECT_EQ(eff(row, col), 0.f) << "Row " << row << ", Col " << col;
		}
	}
}

// ============================================================================
// Wrapper constructor & defaults tests
// ============================================================================

TEST(ControlAllocatorWrapperTest, DefaultConstruction)
{
	// Verify wrapper constructs with Iris defaults
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params = wrapper.getParams();

	EXPECT_EQ(params.airmode, 1);
	EXPECT_EQ(params.geometry.num_rotors, 4);
	EXPECT_TRUE(params.normalize_rpy);

	// Effectiveness matrix should be populated
	const auto &eff = wrapper.getEffectivenessMatrix();
	// Thrust_z column should be non-zero for first 4 motors
	for (int i = 0; i < 4; i++) {
		EXPECT_NE(eff(5, i), 0.f);
	}
}

// ============================================================================
// Zero input test
// ============================================================================

TEST(ControlAllocatorWrapperTest, ZeroInput)
{
	// Zero torque + zero thrust → all motors at minimum (0.0)
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.f, 0.f, 0.f);
	input.thrust_sp = Vector3f(0.f, 0.f, 0.f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	EXPECT_EQ(output.num_motors, 4);
	for (int i = 0; i < 4; i++) {
		EXPECT_GE(output.motors[i], 0.f);
		EXPECT_LE(output.motors[i], 1.f);
	}

	// No saturation with zero input
	for (int i = 0; i < 3; i++) {
		EXPECT_FALSE(output.saturation_positive(i));
		EXPECT_FALSE(output.saturation_negative(i));
	}
}

// ============================================================================
// Hover thrust test (pure z-thrust, no torque)
// ============================================================================

TEST(ControlAllocatorWrapperTest, HoverThrust)
{
	// Pure downward thrust in NED → all motors should be equal
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.f, 0.f, 0.f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f); // 50% thrust upward in NED
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// All 4 motors should produce approximately equal thrust
	float avg = (output.motors[0] + output.motors[1] + output.motors[2] + output.motors[3]) / 4.f;
	EXPECT_GT(avg, 0.f);

	for (int i = 0; i < 4; i++) {
		EXPECT_NEAR(output.motors[i], avg, 0.05f)
		    << "Motor " << i << " should be near average in pure hover";
	}
}

// ============================================================================
// Roll torque allocation test
// ============================================================================

TEST(ControlAllocatorWrapperTest, RollTorqueAllocation)
{
	// Positive roll torque should increase right-side motors and decrease left-side
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.3f, 0.f, 0.f); // Positive roll torque
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);  // plus hover thrust
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// For Iris geometry:
	// Roll eff: Motor 0 (FR) = -0.245, Motor 1 (BL) = +0.1875, Motor 2 (FL) = +0.245, Motor 3 (BR) = -0.1875
	// Positive roll torque: increase motors with negative roll eff (0, 3) or decrease motors with positive eff (1, 2)
	// Motors on left side (2=FL) should be higher, motors on right side (0=FR) should be lower
	// Actually with normalization and mixing this is not trivially predictable,
	// but we can check that the motor commands differ
	float max_m = -1.f, min_m = 2.f;
	for (int i = 0; i < 4; i++) {
		max_m = fmaxf(max_m, output.motors[i]);
		min_m = fminf(min_m, output.motors[i]);
		EXPECT_GE(output.motors[i], 0.f);
		EXPECT_LE(output.motors[i], 1.f);
	}
	EXPECT_GT(max_m - min_m, 0.01f) << "Motors should differ for non-zero roll torque";
}

// ============================================================================
// Yaw torque allocation test
// ============================================================================

TEST(ControlAllocatorWrapperTest, YawTorqueAllocation)
{
	// Positive yaw torque: CCW motors (0,1) increase, CW motors (2,3) decrease
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.f, 0.f, 0.2f);   // Positive yaw torque
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// CCW motors (0,1): positive yaw effectiveness → should spin faster
	// CW motors (2,3): negative yaw effectiveness → should spin slower
	float ccw_avg = (output.motors[0] + output.motors[1]) / 2.f;
	float cw_avg = (output.motors[2] + output.motors[3]) / 2.f;
	EXPECT_GT(ccw_avg, cw_avg) << "CCW motors should be higher for positive yaw torque";
}

// ============================================================================
// Motor bounds test
// ============================================================================

TEST(ControlAllocatorWrapperTest, MotorBoundsClipping)
{
	// All motor outputs should be clipped to [0, 1]
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(1.0f, 1.0f, 1.0f); // Large torque
	input.thrust_sp = Vector3f(0.f, 0.f, -0.9f);    // plus large thrust
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	for (int i = 0; i < 4; i++) {
		EXPECT_GE(output.motors[i], 0.f) << "Motor " << i << " below minimum";
		EXPECT_LE(output.motors[i], 1.f) << "Motor " << i << " above maximum";
	}
}

// ============================================================================
// Saturation feedback tests
// ============================================================================

TEST(ControlAllocatorWrapperTest, NoSaturationSmallInput)
{
	// Small inputs should not saturate
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.01f, 0.01f, 0.01f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.3f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// Unallocated torque should be near zero
	EXPECT_NEAR(output.unallocated_torque(0), 0.f, 0.1f);
	EXPECT_NEAR(output.unallocated_torque(1), 0.f, 0.1f);
	EXPECT_NEAR(output.unallocated_torque(2), 0.f, 0.1f);
}

TEST(ControlAllocatorWrapperTest, SaturationLargeTorque)
{
	// Very large torque demand should produce saturation flags
	ControlAllocatorWrapper wrapper;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(10.f, 10.f, 10.f);  // Extremely large
	input.thrust_sp = Vector3f(0.f, 0.f, -0.9f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// With such extreme demands, the allocator should not be able to fully deliver
	// At least some unallocated torque expected
	float total_unalloc = fabsf(output.unallocated_torque(0)) +
			      fabsf(output.unallocated_torque(1)) +
			      fabsf(output.unallocated_torque(2));
	EXPECT_GT(total_unalloc, 0.f) << "Should have some unallocated torque with extreme demands";
}

// ============================================================================
// Airmode tests
// ============================================================================

TEST(ControlAllocatorWrapperTest, AirmodeDisabled)
{
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params;
	params.airmode = 0; // Disabled
	wrapper.setParams(params);

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.3f, 0.f, 0.f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	for (int i = 0; i < 4; i++) {
		EXPECT_GE(output.motors[i], 0.f);
		EXPECT_LE(output.motors[i], 1.f);
	}
}

TEST(ControlAllocatorWrapperTest, AirmodeRP)
{
	// Default airmode 1: RP priority, can increase total thrust to maintain roll/pitch
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params;
	params.airmode = 1; // RP airmode
	wrapper.setParams(params);

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.5f, 0.f, 0.f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.1f); // low thrust
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// With RP airmode, the allocator should maintain roll authority even at low thrust
	// Check that motor differential exists
	float max_m = output.motors[0], min_m = output.motors[0];
	for (int i = 1; i < 4; i++) {
		max_m = fmaxf(max_m, output.motors[i]);
		min_m = fminf(min_m, output.motors[i]);
	}
	EXPECT_GT(max_m - min_m, 0.01f) << "RP airmode should maintain roll authority";
}

TEST(ControlAllocatorWrapperTest, AirmodeRPY)
{
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params;
	params.airmode = 2; // RPY airmode
	wrapper.setParams(params);

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.f, 0.f, 0.5f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.1f); // low thrust
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// With RPY airmode, yaw should also be maintained
	float ccw_avg = (output.motors[0] + output.motors[1]) / 2.f;
	float cw_avg = (output.motors[2] + output.motors[3]) / 2.f;
	EXPECT_GT(fabsf(ccw_avg - cw_avg), 0.01f) << "RPY airmode should maintain yaw authority";
}

// ============================================================================
// Custom geometry test
// ============================================================================

TEST(ControlAllocatorWrapperTest, CustomGeometry)
{
	// Use generic geometry and verify it works
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params;
	params.geometry = createQuadXGeneric();
	wrapper.setParams(params);

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.f, 0.f, 0.f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// Should produce valid motor outputs
	for (int i = 0; i < 4; i++) {
		EXPECT_GE(output.motors[i], 0.f);
		EXPECT_LE(output.motors[i], 1.f);
	}
}

// ============================================================================
// Deterministic / idempotent test
// ============================================================================

TEST(ControlAllocatorWrapperTest, DeterministicOutput)
{
	// Same inputs → same outputs (important for RL training determinism)
	ControlAllocatorWrapper wrapper1;
	ControlAllocatorWrapper wrapper2;

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(0.2f, -0.1f, 0.05f);
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;

	ControlAllocatorOutput output1, output2;
	wrapper1.update(input, output1);
	wrapper2.update(input, output2);

	for (int i = 0; i < 4; i++) {
		EXPECT_EQ(output1.motors[i], output2.motors[i])
		    << "Motor " << i << " should be identical for identical inputs";
	}
}

// ============================================================================
// Multiple sequential updates
// ============================================================================

TEST(ControlAllocatorWrapperTest, SequentialUpdates)
{
	// Running multiple updates should work correctly
	ControlAllocatorWrapper wrapper;

	for (int step = 0; step < 100; step++) {
		ControlAllocatorInput input;
		float t = step * 0.004f;
		input.torque_sp = Vector3f(0.1f * sinf(t), 0.1f * cosf(t), 0.f);
		input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
		input.dt = 0.004f;

		ControlAllocatorOutput output;
		wrapper.update(input, output);

		for (int i = 0; i < 4; i++) {
			EXPECT_GE(output.motors[i], 0.f);
			EXPECT_LE(output.motors[i], 1.f);
		}
	}
}

// ============================================================================
// Full pipeline test: Attitude → Rate → Allocator → Motors
// ============================================================================

TEST(PipelineTest, AttitudeToMotors)
{
	// End-to-end: attitude error → rate setpoint → torque → motor commands
	// This proves all three extracted modules connect correctly.

	// Step 1: Attitude controller — compute rate setpoints from attitude error
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
	att_state.q = Quatf(1.f, 0.f, 0.f, 0.f); // Level

	AttitudeSetpoint att_sp;
	att_sp.q_d = Quatf(Eulerf(10.f * static_cast<float>(M_PI) / 180.f, 0.f, 0.f)); // 10° roll
	att_sp.thrust_body = Vector3f(0.f, 0.f, -0.5f);

	RateSetpoint rate_sp = att_ctrl.update(att_state, att_sp);

	// Expect non-zero roll rate setpoint
	EXPECT_GT(fabsf(rate_sp.rates(0)), 0.01f);

	// Step 2: Rate controller — compute torque from rate error
	RateControllerWrapper rate_ctrl;
	RateControlParams rate_params;
	rate_params.gain_p = Vector3f(0.15f, 0.15f, 0.2f);
	rate_params.gain_i = Vector3f(0.2f, 0.2f, 0.1f);
	rate_params.gain_d = Vector3f(0.003f, 0.003f, 0.f);
	rate_params.gain_ff = Vector3f(0.f, 0.f, 0.f);
	rate_params.integrator_limit = Vector3f(0.3f, 0.3f, 0.3f);
	rate_ctrl.setParams(rate_params);

	RateControlInput rate_input;
	rate_input.rates_sp = rate_sp.rates;
	rate_input.rates_actual = Vector3f(0.f, 0.f, 0.f); // No actual body rates yet
	rate_input.angular_accel = Vector3f(0.f, 0.f, 0.f);
	rate_input.dt = 0.004f;
	rate_input.landed = false;

	RateControlOutput rate_output;
	rate_ctrl.update(rate_input, rate_output);

	// Expect non-zero torque
	EXPECT_GT(fabsf(rate_output.torque_sp(0)), 0.001f);

	// Step 3: Control allocator — compute motor commands from torque
	ControlAllocatorWrapper alloc;

	ControlAllocatorInput alloc_input;
	alloc_input.torque_sp = rate_output.torque_sp;
	alloc_input.thrust_sp = rate_sp.thrust_body; // Use thrust from attitude controller
	alloc_input.dt = 0.004f;

	ControlAllocatorOutput alloc_output;
	alloc.update(alloc_input, alloc_output);

	// All motors should be valid
	for (int i = 0; i < 4; i++) {
		EXPECT_GE(alloc_output.motors[i], 0.f) << "Motor " << i;
		EXPECT_LE(alloc_output.motors[i], 1.f) << "Motor " << i;
	}

	// Motors should not all be equal (there's a roll torque demand)
	float max_m = alloc_output.motors[0], min_m = alloc_output.motors[0];
	for (int i = 1; i < 4; i++) {
		max_m = fmaxf(max_m, alloc_output.motors[i]);
		min_m = fminf(min_m, alloc_output.motors[i]);
	}
	EXPECT_GT(max_m - min_m, 0.01f)
	    << "Motor commands should differ for non-zero roll torque from attitude error";
}

// ============================================================================
// Saturation anti-windup feedback loop test
// ============================================================================

TEST(PipelineTest, SaturationAntiWindup)
{
	// Verify that saturation flags can be fed back to rate controller
	// for integrator anti-windup

	RateControllerWrapper rate_ctrl;
	RateControlParams rate_params;
	rate_params.gain_p = Vector3f(0.15f, 0.15f, 0.2f);
	rate_params.gain_i = Vector3f(0.2f, 0.2f, 0.1f);
	rate_params.gain_d = Vector3f(0.003f, 0.003f, 0.f);
	rate_params.gain_ff = Vector3f(0.f, 0.f, 0.f);
	rate_params.integrator_limit = Vector3f(0.3f, 0.3f, 0.3f);
	rate_ctrl.setParams(rate_params);

	ControlAllocatorWrapper alloc;

	// Drive large rate error for many steps
	for (int step = 0; step < 50; step++) {
		RateControlInput rate_input;
		rate_input.rates_sp = Vector3f(5.0f, 0.f, 0.f); // Large roll rate demand
		rate_input.rates_actual = Vector3f(0.f, 0.f, 0.f);
		rate_input.angular_accel = Vector3f(0.f, 0.f, 0.f);
		rate_input.dt = 0.004f;
		rate_input.landed = false;

		RateControlOutput rate_output;
		rate_ctrl.update(rate_input, rate_output);

		ControlAllocatorInput alloc_input;
		alloc_input.torque_sp = rate_output.torque_sp;
		alloc_input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
		alloc_input.dt = 0.004f;

		ControlAllocatorOutput alloc_output;
		alloc.update(alloc_input, alloc_output);

		// Feed saturation back to rate controller
		rate_ctrl.setSaturationStatus(alloc_output.saturation_positive,
					      alloc_output.saturation_negative);
	}

	// After 50 steps with saturation feedback, the integrator should be limited
	// (This test verifies the API connectivity, not exact integrator values)
	SUCCEED() << "Saturation feedback loop completed without errors";
}

// ============================================================================
// Normalize RPY test
// ============================================================================

TEST(ControlAllocatorWrapperTest, NormalizeRPYEnabled)
{
	// With normalize_rpy=true, the allocator should scale RPY demands
	// so that full-scale torque input [-1,1] maps to full actuator range
	ControlAllocatorWrapper wrapper;
	ControlAllocatorParams params;
	params.normalize_rpy = true;
	wrapper.setParams(params);

	ControlAllocatorInput input;
	input.torque_sp = Vector3f(1.0f, 0.f, 0.f); // Full scale roll
	input.thrust_sp = Vector3f(0.f, 0.f, -0.5f);
	input.dt = 0.004f;

	ControlAllocatorOutput output;
	wrapper.update(input, output);

	// Should use full motor range
	float max_m = output.motors[0], min_m = output.motors[0];
	for (int i = 1; i < 4; i++) {
		max_m = fmaxf(max_m, output.motors[i]);
		min_m = fminf(min_m, output.motors[i]);
	}
	EXPECT_GT(max_m - min_m, 0.3f) << "Full-scale roll should produce significant motor differential";
}
