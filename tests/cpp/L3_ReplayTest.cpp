/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *   BSD 3-Clause License (same as PX4)
 *
 ****************************************************************************/

/**
 * @file L3_ReplayTest.cpp
 *
 * L3 PX4 SITL Log Replay Validation.
 *
 * Reads a CSV trace extracted from a PX4 SITL ULog (via extract_ulog_trace.py)
 * and replays each timestep through the extracted control pipeline.  Compares
 * computed outputs to PX4's logged values at each stage to validate equivalence.
 *
 * 4 test cases isolate errors per module:
 *   A — Attitude controller only  (pure math, no state)
 *   B — Rate controller isolated  (PX4 inputs → our torque)
 *   C — Control allocator isolated (PX4 torque → our motors)
 *   D — Full pipeline             (only external inputs, chain all 4 stages)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <AttitudeControllerWrapper.hpp>
#include <AttitudeControlTypes.hpp>

#include <RateControllerWrapper.hpp>
#include <RateControlTypes.hpp>

#include <ControlAllocatorWrapper.hpp>
#include <ControlAllocatorTypes.hpp>

#include <ActuatorOutputWrapper.hpp>
#include <ActuatorOutputTypes.hpp>

using namespace matrix;
using namespace px4_stack;

// ============================================================================
// CSV trace data
// ============================================================================

#ifndef L3_TRACE_PATH
#error "L3_TRACE_PATH must be defined (path to l3_trace.csv)"
#endif

struct TraceRow {
	uint64_t timestamp_us;
	// Attitude state
	float q[4];
	// Attitude setpoint
	float q_d[4];
	float thrust_body_z;
	float yaw_sp_move_rate;
	// Rate setpoint (attitude controller output, logged by PX4)
	float rollspeed_sp, pitchspeed_sp, yawspeed_sp;
	// Actual body rates
	float rollspeed, pitchspeed, yawspeed;
	// Angular acceleration
	float angular_accel[3];
	// Torque setpoint (rate controller output, logged by PX4)
	float torque_sp[3];
	// Motor outputs [0,1]
	float motors[4];
	// Actuator outputs (PWM as float)
	float pwm[4];
};

static std::vector<TraceRow> g_trace;

static void loadTrace()
{
	if (!g_trace.empty()) {
		return;  // already loaded
	}

	const char *path = L3_TRACE_PATH;
	std::ifstream file(path);

	ASSERT_TRUE(file.is_open()) << "Cannot open trace CSV: " << path;

	std::string line;
	std::getline(file, line);  // skip header

	while (std::getline(file, line)) {
		if (line.empty()) {
			continue;
		}

		TraceRow r{};
		std::istringstream ss(line);
		std::string token;
		auto next = [&]() -> float {
			std::getline(ss, token, ',');
			return std::stof(token);
		};

		// timestamp
		std::getline(ss, token, ',');
		r.timestamp_us = std::stoull(token);

		// q[0..3]
		for (int k = 0; k < 4; ++k) { r.q[k] = next(); }
		// q_d[0..3]
		for (int k = 0; k < 4; ++k) { r.q_d[k] = next(); }
		// thrust_body_z, yaw_sp_move_rate
		r.thrust_body_z = next();
		r.yaw_sp_move_rate = next();
		// rollspeed_sp, pitchspeed_sp, yawspeed_sp
		r.rollspeed_sp = next();
		r.pitchspeed_sp = next();
		r.yawspeed_sp = next();
		// rollspeed, pitchspeed, yawspeed
		r.rollspeed = next();
		r.pitchspeed = next();
		r.yawspeed = next();
		// angular_accel[0..2]
		for (int k = 0; k < 3; ++k) { r.angular_accel[k] = next(); }
		// torque_sp[0..2]
		for (int k = 0; k < 3; ++k) { r.torque_sp[k] = next(); }
		// motors[0..3]
		for (int k = 0; k < 4; ++k) { r.motors[k] = next(); }
		// pwm[0..3]
		for (int k = 0; k < 4; ++k) { r.pwm[k] = next(); }

		g_trace.push_back(r);
	}

	ASSERT_GT(g_trace.size(), 100u) << "Trace too short for meaningful validation";
	printf("[L3] Loaded %zu trace samples from %s\n", g_trace.size(), path);
}

// ============================================================================
// Helpers
// ============================================================================

static constexpr size_t WARMUP_SAMPLES = 50;  // skip initial convergence

struct StageStats {
	std::vector<float> errors;
	int match_count = 0;
	int total_count = 0;
	float tolerance = 0.f;

	void record(float err)
	{
		errors.push_back(err);
		total_count++;

		if (err <= tolerance) {
			match_count++;
		}
	}

	float matchRate() const { return total_count > 0 ? (float)match_count / total_count : 0.f; }

	float maxError() const
	{
		return errors.empty() ? 0.f : *std::max_element(errors.begin(), errors.end());
	}

	float meanError() const
	{
		if (errors.empty()) { return 0.f; }

		return std::accumulate(errors.begin(), errors.end(), 0.f) / (float)errors.size();
	}

	float p95Error() const
	{
		if (errors.empty()) { return 0.f; }

		std::vector<float> sorted = errors;
		std::sort(sorted.begin(), sorted.end());
		size_t idx = (size_t)(0.95f * (sorted.size() - 1));
		return sorted[idx];
	}

	void print(const char *name) const
	{
		printf("  %-25s match=%5d/%5d (%.1f%%)  max=%.6f  mean=%.6f  p95=%.6f  tol=%.4f\n",
		       name, match_count, total_count, matchRate() * 100.f,
		       maxError(), meanError(), p95Error(), tolerance);
	}
};

static float clampDt(uint64_t ts_prev, uint64_t ts_cur)
{
	float dt = (ts_cur > ts_prev) ? (float)(ts_cur - ts_prev) * 1e-6f : 0.004f;
	return std::min(std::max(dt, 0.001f), 0.02f);
}

// ============================================================================
// Test A: Attitude Controller (isolated, stateless)
// ============================================================================

TEST(L3_Replay, AttitudeController)
{
	loadTrace();

	AttitudeControllerWrapper att;
	// Defaults match PX4 Iris SITL (6.5, 6.5, 2.8, yaw_weight=0.4)

	StageStats stats;
	stats.tolerance = 0.05f;  // rad/s — allows small timestamp alignment jitter

	for (size_t i = WARMUP_SAMPLES; i < g_trace.size(); ++i) {
		const auto &row = g_trace[i];

		AttitudeState state;
		state.q = Quatf(row.q[0], row.q[1], row.q[2], row.q[3]);

		AttitudeSetpoint sp;
		sp.q_d = Quatf(row.q_d[0], row.q_d[1], row.q_d[2], row.q_d[3]);
		sp.thrust_body = Vector3f(0.f, 0.f, row.thrust_body_z);
		sp.yaw_sp_move_rate = row.yaw_sp_move_rate;

		RateSetpoint rs = att.update(state, sp);

		float err_roll  = fabsf(rs.rates(0) - row.rollspeed_sp);
		float err_pitch = fabsf(rs.rates(1) - row.pitchspeed_sp);
		float err_yaw   = fabsf(rs.rates(2) - row.yawspeed_sp);
		float max_err   = std::max({err_roll, err_pitch, err_yaw});

		stats.record(max_err);
	}

	printf("\n=== L3 Test A: Attitude Controller ===\n");
	stats.print("rate_setpoint_error");

	// ≥95%: timestamp alignment jitter between topics at different rates
	// (attitude_setpoint at 50ms vs angular_velocity at 10ms)
	EXPECT_GT(stats.matchRate(), 0.95f)
		<< "Attitude controller match rate below 95%";
}

// ============================================================================
// Test B: Rate Controller (isolated — uses PX4's logged rate setpoints)
// ============================================================================

TEST(L3_Replay, RateController)
{
	loadTrace();

	// Check if torque_sp data is available (not NaN)
	bool has_torque = !std::isnan(g_trace[WARMUP_SAMPLES].torque_sp[0]);

	if (!has_torque) {
		printf("\n=== L3 Test B: Rate Controller — SKIPPED (no torque_sp in trace) ===\n");
		GTEST_SKIP() << "vehicle_torque_setpoint not logged in trace CSV";
	}

	RateControllerWrapper rate;
	// Defaults match PX4 Iris SITL

	StageStats stats;
	stats.tolerance = 0.05f;  // normalized torque — allows integrator/LPF jitter

	uint64_t prev_ts = 0;

	for (size_t i = WARMUP_SAMPLES; i < g_trace.size(); ++i) {
		const auto &row = g_trace[i];
		float dt = (prev_ts > 0) ? clampDt(prev_ts, row.timestamp_us) : 0.004f;
		prev_ts = row.timestamp_us;

		RateControlInput ri;
		ri.rates_sp = Vector3f(row.rollspeed_sp, row.pitchspeed_sp, row.yawspeed_sp);
		ri.rates_actual = Vector3f(row.rollspeed, row.pitchspeed, row.yawspeed);
		ri.angular_accel = Vector3f(row.angular_accel[0], row.angular_accel[1], row.angular_accel[2]);
		ri.thrust_body = Vector3f(0.f, 0.f, row.thrust_body_z);
		ri.dt = dt;
		ri.landed = false;

		RateControlOutput ro;
		rate.update(ri, ro);

		float err_roll  = fabsf(ro.torque_sp(0) - row.torque_sp[0]);
		float err_pitch = fabsf(ro.torque_sp(1) - row.torque_sp[1]);
		float err_yaw   = fabsf(ro.torque_sp(2) - row.torque_sp[2]);
		float max_err   = std::max({err_roll, err_pitch, err_yaw});

		stats.record(max_err);
	}

	printf("\n=== L3 Test B: Rate Controller (isolated) ===\n");
	stats.print("torque_error");

	// ≥93%: integrator + yaw LPF accumulate small dt jitter
	EXPECT_GT(stats.matchRate(), 0.93f)
		<< "Rate controller match rate below 93%";
}

// ============================================================================
// Test C: Control Allocator (isolated — uses PX4's logged torque setpoints)
// ============================================================================

TEST(L3_Replay, ControlAllocator)
{
	loadTrace();

	bool has_torque = !std::isnan(g_trace[WARMUP_SAMPLES].torque_sp[0]);

	if (!has_torque) {
		printf("\n=== L3 Test C: Control Allocator — SKIPPED (no torque_sp in trace) ===\n");
		GTEST_SKIP() << "vehicle_torque_setpoint not logged in trace CSV";
	}

	ControlAllocatorWrapper alloc;
	ControlAllocatorParams ca_params;
	ca_params.airmode = 0;  // PX4 SITL Iris default: disabled
	alloc.setParams(ca_params);

	StageStats stats;
	stats.tolerance = 0.03f;

	for (size_t i = WARMUP_SAMPLES; i < g_trace.size(); ++i) {
		const auto &row = g_trace[i];

		ControlAllocatorInput ai;
		ai.torque_sp = Vector3f(row.torque_sp[0], row.torque_sp[1], row.torque_sp[2]);
		ai.thrust_sp = Vector3f(0.f, 0.f, row.thrust_body_z);
		ai.dt = 0.004f;

		ControlAllocatorOutput ao;
		alloc.update(ai, ao);

		float max_err = 0.f;

		for (int m = 0; m < 4; ++m) {
			float err = fabsf(ao.motors[m] - row.motors[m]);
			max_err = std::max(max_err, err);
		}

		stats.record(max_err);
	}

	printf("\n=== L3 Test C: Control Allocator (isolated) ===\n");
	stats.print("motor_error");

	EXPECT_GT(stats.matchRate(), 0.95f)
		<< "Control allocator match rate below 95%";
}

// ============================================================================
// Test D: Full Pipeline (attitude → rate → allocator → actuator output)
// ============================================================================

TEST(L3_Replay, FullPipeline)
{
	loadTrace();

	// --- Initialize all stages ---
	AttitudeControllerWrapper att;

	RateControllerWrapper rate;

	ControlAllocatorWrapper alloc;
	ControlAllocatorParams ca_params;
	ca_params.airmode = 0;  // match PX4 SITL
	alloc.setParams(ca_params);

	ActuatorOutputWrapper act_out;
	// Default PWM params match PX4: f=0.0, PWM [1000, 2000]

	StageStats motor_stats;
	motor_stats.tolerance = 0.05f;  // wider tolerance for full pipeline error accumulation

	StageStats pwm_stats;
	pwm_stats.tolerance = 100.f;  // actuator_outputs at ~100ms vs 10ms reference clock

	uint64_t prev_ts = 0;

	for (size_t i = WARMUP_SAMPLES; i < g_trace.size(); ++i) {
		const auto &row = g_trace[i];
		float dt = (prev_ts > 0) ? clampDt(prev_ts, row.timestamp_us) : 0.004f;
		prev_ts = row.timestamp_us;

		// Stage 1: Attitude → Rate setpoint
		AttitudeState state;
		state.q = Quatf(row.q[0], row.q[1], row.q[2], row.q[3]);

		AttitudeSetpoint sp;
		sp.q_d = Quatf(row.q_d[0], row.q_d[1], row.q_d[2], row.q_d[3]);
		sp.thrust_body = Vector3f(0.f, 0.f, row.thrust_body_z);
		sp.yaw_sp_move_rate = row.yaw_sp_move_rate;

		RateSetpoint rs = att.update(state, sp);

		// Stage 2: Rate → Torque
		RateControlInput ri;
		ri.rates_sp = rs.rates;
		ri.rates_actual = Vector3f(row.rollspeed, row.pitchspeed, row.yawspeed);
		ri.angular_accel = Vector3f(row.angular_accel[0], row.angular_accel[1], row.angular_accel[2]);
		ri.thrust_body = Vector3f(0.f, 0.f, row.thrust_body_z);
		ri.dt = dt;
		ri.landed = false;

		RateControlOutput ro;
		rate.update(ri, ro);

		// Stage 3: Torque → Motors
		ControlAllocatorInput ai;
		ai.torque_sp = ro.torque_sp;
		ai.thrust_sp = Vector3f(0.f, 0.f, row.thrust_body_z);
		ai.dt = dt;

		ControlAllocatorOutput ao;
		alloc.update(ai, ao);

		// Feed saturation back to rate controller (anti-windup)
		rate.setSaturationStatus(ao.saturation_positive, ao.saturation_negative);

		// Motor comparison
		float max_motor_err = 0.f;

		for (int m = 0; m < 4; ++m) {
			float err = fabsf(ao.motors[m] - row.motors[m]);
			max_motor_err = std::max(max_motor_err, err);
		}

		motor_stats.record(max_motor_err);

		// Stage 4: Motors → PWM
		ActuatorOutputInput aoi;

		for (int m = 0; m < 4; ++m) {
			aoi.motors[m] = ao.motors[m];
		}

		aoi.num_motors = 4;
		aoi.armed = true;

		ActuatorOutputOutput aoo;
		act_out.update(aoi, aoo);

		float max_pwm_err = 0.f;

		for (int m = 0; m < 4; ++m) {
			float err = fabsf((float)aoo.values[m] - row.pwm[m]);
			max_pwm_err = std::max(max_pwm_err, err);
		}

		pwm_stats.record(max_pwm_err);
	}

	printf("\n=== L3 Test D: Full Pipeline ===\n");
	motor_stats.print("motor_error");
	pwm_stats.print("pwm_error");

	EXPECT_GT(motor_stats.matchRate(), 0.90f)
		<< "Full pipeline motor match rate below 90%";
}
