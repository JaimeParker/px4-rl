// CpuFleet must agree bitwise with a hand-written loop over the px4_stack wrappers.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <ActuatorOutputWrapper.hpp>
#include <AttitudeControllerWrapper.hpp>
#include <ControlAllocatorWrapper.hpp>
#include <RateControllerWrapper.hpp>

#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"
#include "px4rl/hover_thrust_module.hpp"
#include "px4rl/position_module.hpp"

namespace {

constexpr int kVehicles = 200; // above CpuFleet's OpenMP threshold
constexpr int kSteps = 50;
constexpr float kDt = 0.004f;

struct Rng {
	uint32_t state;

	float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}

	float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

// The same parameter set expressed once as the C struct and once as the px4_stack structs.
struct Variant {
	px4rl_params flat;
	px4_stack::AttitudeControlConfig attitude;
	px4_stack::RateControlParams rate;
	px4_stack::ControlAllocatorParams allocator;
	px4_stack::ActuatorOutputParams output;
};

Variant makeVariant(int kind)
{
	Variant v;
	px4rl_params_default(&v.flat);

	if (kind == 1) {
		v.attitude.roll_p = 5.f;
		v.flat.att_roll_p = 5.f;
		v.attitude.yaw_p = 3.1f;
		v.flat.att_yaw_p = 3.1f;
		v.attitude.yaw_weight = 0.3f;
		v.flat.att_yaw_weight = 0.3f;
		v.attitude.max_roll_rate = 2.5f;
		v.flat.att_max_roll_rate = 2.5f;

		v.rate.gain_p = matrix::Vector3f(0.12f, 0.14f, 0.18f);
		v.rate.gain_i = matrix::Vector3f(0.25f, 0.22f, 0.12f);
		v.rate.gain_d = matrix::Vector3f(0.004f, 0.002f, 0.001f);
		v.rate.gain_ff = matrix::Vector3f(0.01f, 0.01f, 0.02f);
		v.rate.gain_k = matrix::Vector3f(1.1f, 0.9f, 1.2f);
		v.rate.integrator_limit = matrix::Vector3f(0.2f, 0.25f, 0.15f);
		v.rate.yaw_torque_lpf_cutoff_freq = 3.f;

		for (int a = 0; a < 3; ++a) {
			v.flat.rate_p[a] = v.rate.gain_p(a);
			v.flat.rate_i[a] = v.rate.gain_i(a);
			v.flat.rate_d[a] = v.rate.gain_d(a);
			v.flat.rate_ff[a] = v.rate.gain_ff(a);
			v.flat.rate_k[a] = v.rate.gain_k(a);
			v.flat.rate_integrator_limit[a] = v.rate.integrator_limit(a);
		}

		v.flat.rate_yaw_torque_cutoff_hz = 3.f;

		v.allocator.airmode = 2;
		v.flat.airmode = 2;
		v.allocator.normalize_rpy = false;
		v.flat.normalize_rpy = 0;

		for (int m = 0; m < 4; ++m) {
			v.allocator.geometry.rotors[m].thrust_coef = 1.2f;
			v.flat.rotor_ct[m] = 1.2f;
			v.allocator.actuator_max[m] = 0.95f;
			v.flat.actuator_max[m] = 0.95f;
			v.allocator.actuator_min[m] = 0.05f;
			v.flat.actuator_min[m] = 0.05f;
			v.allocator.slew_rate_limit[m] = 2.f;
			v.flat.actuator_slew_rate[m] = 2.f;
		}

		v.output.thrust_model_factor = 0.3f;
		v.flat.thr_mdl_fac = 0.3f;
		v.output.mode = px4_stack::OutputMode::DShot;
		v.flat.output_mode = 1;
		v.output.dshot_min_fraction = 0.05f;
		v.flat.dshot_min_fraction = 0.05f;

	} else if (kind == 2) {
		v.allocator.airmode = 0;
		v.flat.airmode = 0;
		v.output.thrust_model_factor = 0.5f;
		v.flat.thr_mdl_fac = 0.5f;
		v.output.pwm_min = 1100;
		v.flat.pwm_min = 1100.f;
		v.output.pwm_max = 1900;
		v.flat.pwm_max = 1900.f;
		v.rate.gain_k = matrix::Vector3f(0.8f, 0.8f, 0.8f);
		v.flat.rate_k[0] = v.flat.rate_k[1] = v.flat.rate_k[2] = 0.8f;
		v.rate.yaw_torque_lpf_cutoff_freq = 1.f;
		v.flat.rate_yaw_torque_cutoff_hz = 1.f;
	}

	return v;
}

// Hand-written chain, deliberately not using px4rl::Vehicle.
struct ReferenceVehicle {
	px4_stack::AttitudeControllerWrapper attitude;
	px4_stack::RateControllerWrapper rate;
	px4_stack::ControlAllocatorWrapper allocator;
	px4_stack::ActuatorOutputWrapper output;

	void configure(const Variant &v)
	{
		attitude.setGains(v.attitude);
		rate.setParams(v.rate);
		allocator.setParams(v.allocator);
		output.setParams(v.output);
	}
};

struct Buffers {
	std::vector<float> q, q_sp, yawspeed_ff, thrust_sp_body, rates, rates_prev, angular_accel;
	std::vector<uint8_t> landed;
};

struct Outputs {
	std::vector<float> rates_sp, thrust_body, motors, actuator, torque_sp;

	explicit Outputs(int n)
		: rates_sp(3 * n), thrust_body(3 * n), motors(4 * n), actuator(4 * n), torque_sp(3 * n)
	{}
};

void drawInputs(Rng &rng, int n, int step, Buffers &b)
{
	if (b.q.empty()) {
		b.q.resize(4 * n);
		b.q_sp.resize(4 * n);
		b.yawspeed_ff.resize(n);
		b.thrust_sp_body.resize(3 * n);
		b.rates.resize(3 * n);
		b.rates_prev.resize(3 * n);
		b.angular_accel.resize(3 * n);
		b.landed.resize(n);

		for (float &r : b.rates) {
			r = rng.range(-2.f, 2.f);
		}
	}

	b.rates_prev = b.rates;

	for (int i = 0; i < n; ++i) {
		const float roll = rng.range(-0.6f, 0.6f);
		const float pitch = rng.range(-0.6f, 0.6f);
		const float yaw = rng.range(-3.f, 3.f);
		const matrix::Quatf q(matrix::Eulerf(roll, pitch, yaw));

		for (int k = 0; k < 4; ++k) {
			b.q[4 * i + k] = q(k);
		}

		if (step % 10 == 0) {
			const float roll_sp = rng.range(-0.8f, 0.8f);
			const float pitch_sp = rng.range(-0.8f, 0.8f);
			const float yaw_sp = rng.range(-3.1f, 3.1f);
			const matrix::Quatf q_sp(matrix::Eulerf(roll_sp, pitch_sp, yaw_sp));

			for (int k = 0; k < 4; ++k) {
				b.q_sp[4 * i + k] = q_sp(k);
			}

			b.yawspeed_ff[i] = rng.range(-0.5f, 0.5f);
			b.thrust_sp_body[3 * i + 0] = 0.f;
			b.thrust_sp_body[3 * i + 1] = 0.f;
			b.thrust_sp_body[3 * i + 2] = -rng.range(0.3f, 0.7f);
		}

		for (int a = 0; a < 3; ++a) {
			b.rates[3 * i + a] = rng.range(-3.f, 3.f);
			b.angular_accel[3 * i + a] = (b.rates[3 * i + a] - b.rates_prev[3 * i + a]) / kDt;
		}

		b.landed[i] = (step >= 20 && step < 30 && i % 5 == 0) ? 1 : 0;
	}
}

void stepReference(std::vector<ReferenceVehicle> &vehicles, const Buffers &b, const uint8_t *landed,
		   const float *yawspeed_ff, Outputs &out)
{
	const int n = static_cast<int>(vehicles.size());

	for (int i = 0; i < n; ++i) {
		ReferenceVehicle &v = vehicles[i];

		px4_stack::AttitudeState state;
		state.q = matrix::Quatf(b.q[4 * i], b.q[4 * i + 1], b.q[4 * i + 2], b.q[4 * i + 3]);
		px4_stack::AttitudeSetpoint setpoint;
		setpoint.q_d = matrix::Quatf(b.q_sp[4 * i], b.q_sp[4 * i + 1], b.q_sp[4 * i + 2], b.q_sp[4 * i + 3]);
		setpoint.yaw_sp_move_rate = yawspeed_ff != nullptr ? yawspeed_ff[i] : 0.f;
		setpoint.thrust_body = matrix::Vector3f(b.thrust_sp_body[3 * i], b.thrust_sp_body[3 * i + 1],
							b.thrust_sp_body[3 * i + 2]);
		const px4_stack::RateSetpoint rate_setpoint = v.attitude.update(state, setpoint);

		px4_stack::RateControlInput rate_input;
		rate_input.rates_sp = rate_setpoint.rates;
		rate_input.rates_actual = matrix::Vector3f(b.rates[3 * i], b.rates[3 * i + 1], b.rates[3 * i + 2]);
		rate_input.angular_accel = matrix::Vector3f(b.angular_accel[3 * i], b.angular_accel[3 * i + 1],
							    b.angular_accel[3 * i + 2]);
		rate_input.thrust_body = rate_setpoint.thrust_body;
		rate_input.dt = kDt;
		rate_input.landed = landed != nullptr && landed[i] != 0;
		px4_stack::RateControlOutput rate_output;
		v.rate.update(rate_input, rate_output);

		px4_stack::ControlAllocatorInput allocator_input;
		allocator_input.torque_sp = rate_output.torque_sp;
		allocator_input.thrust_sp = rate_output.thrust_body;
		allocator_input.dt = kDt;
		px4_stack::ControlAllocatorOutput allocator_output;
		v.allocator.update(allocator_input, allocator_output);
		v.rate.setSaturationStatus(allocator_output.saturation_positive, allocator_output.saturation_negative);

		px4_stack::ActuatorOutputInput output_input;
		output_input.num_motors = allocator_output.num_motors;

		for (int m = 0; m < 4; ++m) {
			output_input.motors[m] = allocator_output.motors[m];
		}

		px4_stack::ActuatorOutputOutput output_output;
		v.output.update(output_input, output_output);

		for (int a = 0; a < 3; ++a) {
			out.rates_sp[3 * i + a] = rate_setpoint.rates(a);
			out.thrust_body[3 * i + a] = rate_setpoint.thrust_body(a);
			out.torque_sp[3 * i + a] = rate_output.torque_sp(a);
		}

		for (int m = 0; m < 4; ++m) {
			out.motors[4 * i + m] = allocator_output.motors[m];
			out.actuator[4 * i + m] = static_cast<float>(output_output.values[m]);
		}
	}
}

void expectBitwiseEqual(const std::vector<float> &expected, const std::vector<float> &actual, const char *field,
			int step)
{
	ASSERT_EQ(expected.size(), actual.size());

	for (size_t k = 0; k < expected.size(); ++k) {
		if (std::memcmp(&expected[k], &actual[k], sizeof(float)) != 0) {
			FAIL() << field << "[" << k << "] at step " << step << ": expected " << expected[k] << " got "
			       << actual[k];
		}
	}
}

// Drives a CpuFleet and the hand-written reference with the same inputs and compares every output.
// `between_steps(step, fleet, reference)` may reset vehicles on both sides.
template <typename BetweenSteps>
void runComparison(px4rl::CpuFleet &fleet, std::vector<ReferenceVehicle> &reference, bool use_optional_inputs,
		   BetweenSteps &&between_steps)
{
	Rng rng{12345u};
	Buffers buffers;
	Outputs fleet_out(kVehicles), reference_out(kVehicles);

	for (int step = 0; step < kSteps; ++step) {
		between_steps(step, fleet, reference);
		drawInputs(rng, kVehicles, step, buffers);

		const float *yawspeed_ff = use_optional_inputs ? buffers.yawspeed_ff.data() : nullptr;
		const uint8_t *landed = use_optional_inputs ? buffers.landed.data() : nullptr;

		fleet.attitudeStep(buffers.q.data(), buffers.q_sp.data(), yawspeed_ff, buffers.thrust_sp_body.data(),
				   fleet_out.rates_sp.data(), fleet_out.thrust_body.data());
		fleet.rateStep(fleet_out.rates_sp.data(), fleet_out.thrust_body.data(), buffers.rates.data(),
			       buffers.angular_accel.data(), kDt, landed, fleet_out.motors.data(),
			       fleet_out.actuator.data(), fleet_out.torque_sp.data());

		stepReference(reference, buffers, landed, yawspeed_ff, reference_out);

		expectBitwiseEqual(reference_out.rates_sp, fleet_out.rates_sp, "rates_sp", step);
		expectBitwiseEqual(reference_out.thrust_body, fleet_out.thrust_body, "thrust_body", step);
		expectBitwiseEqual(reference_out.motors, fleet_out.motors, "motors", step);
		expectBitwiseEqual(reference_out.actuator, fleet_out.actuator, "actuator", step);
		expectBitwiseEqual(reference_out.torque_sp, fleet_out.torque_sp, "torque_sp", step);

		if (::testing::Test::HasFatalFailure()) {
			return;
		}
	}
}

const auto kNoReset = [](int, px4rl::CpuFleet &, std::vector<ReferenceVehicle> &) {};

constexpr int kPositionPeriod = 5;      // attitude/rate at 250 Hz, position at 50 Hz
constexpr float kPositionDt = 0.02f;
constexpr uint64_t kPositionStepUs = 20000;

// Hand-written position + attitude + rate chain. It spells out the order the fleet must follow: hover
// thrust estimate, then the position controller with the setpoint stamped by the caller's clock, then the
// attitude controller on the stored setpoint, then rate control, allocator and output.
struct PositionReferenceVehicle {
	ReferenceVehicle chain;
	px4rl::PositionModule position;
	px4rl::HoverThrustModule hover;
	matrix::Quatf q;
	matrix::Vector3f thrust_sp_body;
	matrix::Vector3f unallocated_thrust;
	bool thrust_recorded{false};
	matrix::Quatf q_sp;
	float yawspeed{0.f};

	void configure(const px4rl_params &flat)
	{
		Variant variant; // px4_stack structs at their defaults, which is what `flat` holds for them
		variant.flat = flat;
		chain.configure(variant);
		position.configure(flat, flat.position_dt);
		position.setTiltLimitStateDeg(flat.mpc_tiltmax_air);
		hover.configure(flat);
	}
};

px4rl_params positionVariant(int kind)
{
	px4rl_params p;
	px4rl_params_default(&p);

	if (kind == 1) {
		p.position_dt = 0.01f;
		p.mpc_vel_lp = 4.f;
		p.mpc_veld_lp = 3.f;
		p.mpc_vel_nf_frq = 8.f;
		p.mpc_thr_hover = 0.4f;
		p.mpc_tiltmax_air = 30.f;

	} else if (kind == 2) {
		p.mpc_use_hte = 0;
		p.mpc_acc_decouple = 0;
		p.mpc_xy_vel_max = 4.f;
	}

	return p;
}

struct PositionInputs {
	std::vector<float> position, velocity, heading, acceleration_z, trajectory_sp;
	std::vector<uint8_t> in_air;
};

void drawPositionInputs(Rng &rng, int n, int position_index, PositionInputs &in)
{
	in.position.resize(3 * n);
	in.velocity.resize(3 * n);
	in.heading.resize(n);
	in.acceleration_z.resize(n);
	in.trajectory_sp.resize(11 * n);
	in.in_air.resize(n);

	for (int i = 0; i < n; ++i) {
		in.position[3 * i + 0] = rng.range(-5.f, 5.f);
		in.position[3 * i + 1] = rng.range(-5.f, 5.f);
		in.position[3 * i + 2] = rng.range(-10.f, -2.f);

		for (int a = 0; a < 3; ++a) {
			in.velocity[3 * i + a] = rng.range(-2.f, 2.f);
		}

		in.heading[i] = rng.range(-3.f, 3.f);
		in.acceleration_z[i] = rng.range(-1.f, 1.f);
		in.in_air[i] = (i % 5 != 0) ? 1 : 0;

		float *sp = &in.trajectory_sp[11 * i];
		std::fill(sp, sp + 11, NAN);
		sp[9] = rng.range(-3.f, 3.f);

		switch (i % 4) {
		case 0: // velocity + yaw; every third command is invalid and falls back to the last valid one
			if (position_index % 3 != 2) {
				for (int a = 0; a < 3; ++a) {
					sp[3 + a] = rng.range(-3.f, 3.f);
				}
			}

			break;

		case 1: // acceleration + yaw
			for (int a = 0; a < 3; ++a) {
				sp[6 + a] = rng.range(-2.f, 2.f);
			}

			break;

		case 2: // position + yaw
			for (int a = 0; a < 3; ++a) {
				sp[a] = in.position[3 * i + a] + rng.range(-2.f, 2.f);
			}

			break;

		default: // never a valid setpoint
			sp[9] = NAN;
			break;
		}
	}
}

std::vector<ReferenceVehicle> makeReference(const std::vector<int> &kind_per_vehicle)
{
	std::vector<ReferenceVehicle> reference(kind_per_vehicle.size());

	for (size_t i = 0; i < reference.size(); ++i) {
		reference[i].configure(makeVariant(kind_per_vehicle[i]));
	}

	return reference;
}

} // namespace

TEST(FleetCpu, DefaultParamsMatchStructDefaults)
{
	px4rl_params p;
	px4rl_params_default(&p);
	EXPECT_EQ(p.num_rotors, 4);
	EXPECT_EQ(p.airmode, 1);
	EXPECT_EQ(p.normalize_rpy, 1);
	EXPECT_EQ(p.output_mode, 0);
	EXPECT_EQ(p.rotor_axis[0][2], -1.f);
	EXPECT_EQ(p.pwm_min, 1000.f);
	EXPECT_EQ(p.pwm_max, 2000.f);

	px4rl::CpuFleet fleet(kVehicles);
	auto reference = makeReference(std::vector<int>(kVehicles, 0));
	runComparison(fleet, reference, true, kNoReset);
}

TEST(FleetCpu, NullOptionalInputsMatchReference)
{
	px4rl::CpuFleet fleet(kVehicles);
	auto reference = makeReference(std::vector<int>(kVehicles, 0));
	runComparison(fleet, reference, false, kNoReset);
}

TEST(FleetCpu, BroadcastParams)
{
	px4rl::CpuFleet fleet(kVehicles);
	const Variant variant = makeVariant(1);
	fleet.reset(nullptr, &variant.flat, 1);
	auto reference = makeReference(std::vector<int>(kVehicles, 1));
	runComparison(fleet, reference, true, kNoReset);
}

TEST(FleetCpu, PerVehicleParams)
{
	px4rl::CpuFleet fleet(kVehicles);
	std::vector<int> kinds(kVehicles);
	std::vector<px4rl_params> params(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		kinds[i] = i % 3;
		params[i] = makeVariant(kinds[i]).flat;
	}

	fleet.reset(nullptr, params.data(), kVehicles);
	auto reference = makeReference(kinds);
	runComparison(fleet, reference, true, kNoReset);
}

TEST(FleetCpu, MaskedResetRebuildsOnlySelectedVehicles)
{
	px4rl::CpuFleet fleet(kVehicles);
	std::vector<int> kinds(kVehicles, 0);
	auto reference = makeReference(kinds);

	std::vector<uint8_t> mask(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		mask[i] = (i % 3 == 0) ? 1 : 0;
	}

	const Variant broadcast = makeVariant(1);
	std::vector<px4rl_params> per_vehicle(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		per_vehicle[i] = makeVariant(i % 3).flat;
	}

	const auto between_steps = [&](int step, px4rl::CpuFleet &f, std::vector<ReferenceVehicle> &ref) {
		if (step == 15) {
			// Broadcast params on a subset; integrators and filters of the others must survive.
			f.reset(mask.data(), &broadcast.flat, 1);

			for (int i = 0; i < kVehicles; ++i) {
				if (mask[i]) {
					ref[i] = ReferenceVehicle();
					ref[i].configure(makeVariant(1));
				}
			}

		} else if (step == 30) {
			// Per-vehicle params on the same subset.
			f.reset(mask.data(), per_vehicle.data(), kVehicles);

			for (int i = 0; i < kVehicles; ++i) {
				if (mask[i]) {
					ref[i] = ReferenceVehicle();
					ref[i].configure(makeVariant(i % 3));
				}
			}
		}
	};

	runComparison(fleet, reference, true, between_steps);
}

TEST(FleetCpu, ResetWithoutMaskRebuildsEveryVehicle)
{
	px4rl::CpuFleet fleet(kVehicles);
	auto reference = makeReference(std::vector<int>(kVehicles, 0));
	const Variant variant = makeVariant(2);

	const auto between_steps = [&](int step, px4rl::CpuFleet &f, std::vector<ReferenceVehicle> &ref) {
		if (step == 20) {
			f.reset(nullptr, &variant.flat, 1);

			for (ReferenceVehicle &v : ref) {
				v = ReferenceVehicle();
				v.configure(variant);
			}
		}
	};

	runComparison(fleet, reference, true, between_steps);
}

TEST(FleetCpu, PositionChainMatchesReference)
{
	constexpr int kPositionSteps = 30;
	constexpr int kMaskedStep = 15;
	constexpr int kVariants = 3;

	px4rl::CpuFleet fleet(kVehicles);
	std::vector<px4rl_params> flat(kVehicles);
	std::vector<PositionReferenceVehicle> reference(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		flat[i] = positionVariant(i % kVariants);
		reference[i].configure(flat[i]);
	}

	fleet.reset(nullptr, flat.data(), kVehicles);

	std::vector<uint8_t> mask(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		mask[i] = (i % 3 == 1) ? 1 : 0;
	}

	Rng rng{777u};
	Rng position_rng{4242u};
	Buffers buffers;
	PositionInputs in;
	Outputs fleet_out(kVehicles), reference_out(kVehicles);
	std::vector<float> q_sp(4 * kVehicles), q_sp_ref(4 * kVehicles), thrust(3 * kVehicles), thrust_ref(3 * kVehicles);
	std::vector<float> yawspeed(kVehicles), yawspeed_ref(kVehicles), hover(kVehicles), hover_ref(kVehicles);
	std::vector<uint8_t> outcome(kVehicles), outcome_ref(kVehicles);
	int outcome_count[3] = {0, 0, 0};

	for (int step = 0; step < kPositionSteps * kPositionPeriod; ++step) {
		if (step == kMaskedStep * kPositionPeriod) {
			fleet.reset(mask.data(), flat.data(), kVehicles);

			for (int i = 0; i < kVehicles; ++i) {
				if (mask[i]) {
					reference[i] = PositionReferenceVehicle();
					reference[i].configure(flat[i]);
				}
			}
		}

		drawInputs(rng, kVehicles, step, buffers);

		if (step % kPositionPeriod == 0) {
			const int position_index = step / kPositionPeriod;
			const uint64_t time_us = 1000000u + kPositionStepUs * static_cast<uint64_t>(position_index);
			drawPositionInputs(position_rng, kVehicles, position_index, in);

			fleet.positionStep(in.position.data(), in.velocity.data(), in.heading.data(),
					   in.acceleration_z.data(), in.trajectory_sp.data(), in.in_air.data(), kPositionDt,
					   time_us, q_sp.data(), thrust.data(), yawspeed.data(), hover.data(), outcome.data());

			for (int i = 0; i < kVehicles; ++i) {
				PositionReferenceVehicle &v = reference[i];
				const float *sp = &in.trajectory_sp[11 * i];
				const matrix::Vector3f velocity(in.velocity[3 * i], in.velocity[3 * i + 1], in.velocity[3 * i + 2]);

				v.hover.update(true, in.in_air[i] != 0 && v.thrust_recorded, kPositionDt, in.acceleration_z[i],
					       velocity, v.q, v.thrust_sp_body, v.unallocated_thrust, time_us);
				v.position.applyHoverThrustEstimate(v.hover.hoverThrust(), v.hover.valid());

				trajectory_setpoint_s setpoint{time_us, {sp[0], sp[1], sp[2]}, {sp[3], sp[4], sp[5]},
							       {sp[6], sp[7], sp[8]}, {NAN, NAN, NAN}, sp[9], sp[10]};
				vehicle_attitude_setpoint_s attitude_setpoint{};
				const auto result = v.position.update(
					matrix::Vector3f(in.position[3 * i], in.position[3 * i + 1], in.position[3 * i + 2]), velocity,
					in.heading[i], kPositionDt, setpoint, time_us, attitude_setpoint);

				v.q_sp = matrix::Quatf(attitude_setpoint.q_d[0], attitude_setpoint.q_d[1], attitude_setpoint.q_d[2],
						       attitude_setpoint.q_d[3]);
				v.yawspeed = attitude_setpoint.yaw_sp_move_rate;

				for (int k = 0; k < 4; ++k) {
					q_sp_ref[4 * i + k] = v.q_sp(k);
				}

				for (int k = 0; k < 3; ++k) {
					thrust_ref[3 * i + k] = attitude_setpoint.thrust_body[k];
				}

				yawspeed_ref[i] = v.yawspeed;
				hover_ref[i] = v.hover.hoverThrust();
				outcome_ref[i] = static_cast<uint8_t>(result);
				++outcome_count[outcome_ref[i]];
			}

			expectBitwiseEqual(q_sp_ref, q_sp, "q_sp", step);
			expectBitwiseEqual(thrust_ref, thrust, "position thrust", step);
			expectBitwiseEqual(yawspeed_ref, yawspeed, "yawspeed", step);
			expectBitwiseEqual(hover_ref, hover, "hover_thrust", step);
			ASSERT_EQ(outcome_ref, outcome) << "outcome at step " << step;
		}

		fleet.attitudeStep(buffers.q.data(), nullptr, nullptr, thrust.data(), fleet_out.rates_sp.data(),
				   fleet_out.thrust_body.data());
		fleet.rateStep(fleet_out.rates_sp.data(), fleet_out.thrust_body.data(), buffers.rates.data(),
			       buffers.angular_accel.data(), kDt, nullptr, fleet_out.motors.data(),
			       fleet_out.actuator.data(), fleet_out.torque_sp.data());

		for (int i = 0; i < kVehicles; ++i) {
			PositionReferenceVehicle &v = reference[i];
			v.q = matrix::Quatf(buffers.q[4 * i], buffers.q[4 * i + 1], buffers.q[4 * i + 2], buffers.q[4 * i + 3]);

			px4_stack::AttitudeState state;
			state.q = v.q;
			px4_stack::AttitudeSetpoint setpoint;
			setpoint.q_d = v.q_sp;
			setpoint.yaw_sp_move_rate = v.yawspeed;
			setpoint.thrust_body = matrix::Vector3f(thrust_ref[3 * i], thrust_ref[3 * i + 1], thrust_ref[3 * i + 2]);
			const px4_stack::RateSetpoint rate_setpoint = v.chain.attitude.update(state, setpoint);

			px4_stack::RateControlInput rate_input;
			rate_input.rates_sp = rate_setpoint.rates;
			rate_input.rates_actual = matrix::Vector3f(buffers.rates[3 * i], buffers.rates[3 * i + 1], buffers.rates[3 * i + 2]);
			rate_input.angular_accel = matrix::Vector3f(buffers.angular_accel[3 * i], buffers.angular_accel[3 * i + 1],
								    buffers.angular_accel[3 * i + 2]);
			rate_input.thrust_body = rate_setpoint.thrust_body;
			rate_input.dt = kDt;
			rate_input.landed = false;
			px4_stack::RateControlOutput rate_output;
			v.chain.rate.update(rate_input, rate_output);

			px4_stack::ControlAllocatorInput allocator_input;
			allocator_input.torque_sp = rate_output.torque_sp;
			allocator_input.thrust_sp = rate_output.thrust_body;
			allocator_input.dt = kDt;
			px4_stack::ControlAllocatorOutput allocator_output;
			v.chain.allocator.update(allocator_input, allocator_output);
			v.chain.rate.setSaturationStatus(allocator_output.saturation_positive, allocator_output.saturation_negative);
			v.thrust_sp_body = allocator_input.thrust_sp;
			v.unallocated_thrust = allocator_output.unallocated_thrust;
			v.thrust_recorded = true;

			px4_stack::ActuatorOutputInput output_input;
			output_input.num_motors = allocator_output.num_motors;

			for (int m = 0; m < 4; ++m) {
				output_input.motors[m] = allocator_output.motors[m];
			}

			px4_stack::ActuatorOutputOutput output_output;
			v.chain.output.update(output_input, output_output);

			for (int a = 0; a < 3; ++a) {
				reference_out.rates_sp[3 * i + a] = rate_setpoint.rates(a);
				reference_out.thrust_body[3 * i + a] = rate_setpoint.thrust_body(a);
				reference_out.torque_sp[3 * i + a] = rate_output.torque_sp(a);
			}

			for (int m = 0; m < 4; ++m) {
				reference_out.motors[4 * i + m] = allocator_output.motors[m];
				reference_out.actuator[4 * i + m] = static_cast<float>(output_output.values[m]);
			}
		}

		expectBitwiseEqual(reference_out.rates_sp, fleet_out.rates_sp, "rates_sp", step);
		expectBitwiseEqual(reference_out.motors, fleet_out.motors, "motors", step);
		expectBitwiseEqual(reference_out.actuator, fleet_out.actuator, "actuator", step);
		expectBitwiseEqual(reference_out.torque_sp, fleet_out.torque_sp, "torque_sp", step);

		if (::testing::Test::HasFatalFailure()) {
			return;
		}
	}

	// The scenario must exercise all three outcomes, otherwise the comparison proves little.
	EXPECT_GT(outcome_count[0], 0);
	EXPECT_GT(outcome_count[1], 0);
	EXPECT_GT(outcome_count[2], 0);
}

TEST(FleetCpu, PositionStepOutcomeUsesCallerClock)
{
	px4rl::CpuFleet fleet(1);
	const float position[3] = {0.f, 0.f, -5.f};
	const float velocity[3] = {0.f, 0.f, 0.f};
	const float heading = 0.f;
	const float az = 0.f;
	float valid_sp[11], invalid_sp[11];
	std::fill(valid_sp, valid_sp + 11, NAN);
	std::fill(invalid_sp, invalid_sp + 11, NAN);
	valid_sp[3] = 1.f;
	valid_sp[4] = valid_sp[5] = 0.f;
	valid_sp[9] = 0.f;

	float q_sp[4], thrust[3], yawspeed;
	uint8_t outcome = 255;
	const auto step = [&](const float *sp, uint64_t time_us) {
		fleet.positionStep(position, velocity, &heading, &az, sp, nullptr, kPositionDt, time_us, q_sp, thrust,
				   &yawspeed, nullptr, &outcome);
		return outcome;
	};

	// A fresh vehicle was never given a valid setpoint: nothing to fall back to.
	EXPECT_EQ(step(invalid_sp, 1000000u), PX4RL_OUTCOME_FAILSAFE);
	EXPECT_EQ(step(valid_sp, 5000000u), PX4RL_OUTCOME_OK);
	// The valid setpoint was stamped 20 ms ago, so it is reused.
	EXPECT_EQ(step(invalid_sp, 5020000u), PX4RL_OUTCOME_FALLBACK);
	// Older than 200 ms.
	EXPECT_EQ(step(invalid_sp, 5300000u), PX4RL_OUTCOME_FAILSAFE);

	// outcome_out, hover_thrust_out and in_air are optional.
	fleet.positionStep(position, velocity, &heading, &az, valid_sp, nullptr, kPositionDt, 6000000u, q_sp, thrust,
			   &yawspeed, nullptr, nullptr);
	EXPECT_TRUE(std::isfinite(q_sp[0]) && std::isfinite(thrust[2]) && std::isfinite(yawspeed));
}

TEST(FleetCpu, TiltLimitStartsAtConfiguredLimit)
{
	px4rl_params p;
	px4rl_params_default(&p);
	p.mpc_tiltmax_air = 30.f;
	px4rl::CpuFleet fleet(1);
	fleet.reset(nullptr, &p, 1);

	const float position[3] = {0.f, 0.f, -5.f};
	const float velocity[3] = {0.f, 0.f, 0.f};
	const float heading = 0.f;
	const float az = 0.f;
	float sp[11];
	std::fill(sp, sp + 11, NAN);
	sp[3] = 10.f; // a large velocity error demands the full tilt at once
	sp[4] = sp[5] = 0.f;
	sp[9] = 0.f;

	float q_sp[4], thrust[3], yawspeed;
	fleet.positionStep(position, velocity, &heading, &az, sp, nullptr, kPositionDt, 1000000u, q_sp, thrust,
			   &yawspeed, nullptr, nullptr);

	const matrix::Dcmf rotation{matrix::Quatf(q_sp[0], q_sp[1], q_sp[2], q_sp[3])};
	const float tilt = acosf(rotation(2, 2));
	EXPECT_NEAR(tilt, 30.f * 3.14159265f / 180.f, 2e-3f);
}

TEST(FleetCpu, AttitudeStepWithoutSetpointUsesLevelStoredSetpoint)
{
	px4rl::CpuFleet fleet(2);
	const float q[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
	const float level_q_sp[8] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f};
	const float thrust[6] = {0.f, 0.f, -0.5f, 0.f, 0.f, -0.5f};
	float rates_stored[6], thrust_stored[6], rates_explicit[6], thrust_explicit[6];

	fleet.attitudeStep(q, nullptr, nullptr, thrust, rates_stored, thrust_stored);
	px4rl::CpuFleet explicit_fleet(2);
	explicit_fleet.attitudeStep(q, level_q_sp, nullptr, thrust, rates_explicit, thrust_explicit);

	EXPECT_EQ(std::memcmp(rates_stored, rates_explicit, sizeof(rates_stored)), 0);
	EXPECT_EQ(std::memcmp(thrust_stored, thrust_explicit, sizeof(thrust_stored)), 0);
}

TEST(FleetCpu, InvalidArgumentsThrow)
{
	EXPECT_THROW(px4rl::CpuFleet(0), std::invalid_argument);

	px4rl::CpuFleet fleet(8);
	px4rl_params two[2];
	px4rl_params_default(&two[0]);
	px4rl_params_default(&two[1]);
	EXPECT_THROW(fleet.reset(nullptr, two, 2), std::invalid_argument);
	EXPECT_THROW(fleet.reset(nullptr, nullptr, 1), std::invalid_argument);

	px4rl_params bad;
	px4rl_params_default(&bad);
	bad.num_rotors = 0;
	EXPECT_THROW(fleet.reset(nullptr, &bad, 1), std::invalid_argument);
	bad.num_rotors = PX4RL_MAX_ROTORS + 1;
	EXPECT_THROW(fleet.reset(nullptr, &bad, 1), std::invalid_argument);
	px4rl_params_default(&bad);
	bad.airmode = 3;
	EXPECT_THROW(fleet.reset(nullptr, &bad, 1), std::invalid_argument);
	px4rl_params_default(&bad);
	bad.position_dt = 0.f;
	EXPECT_THROW(fleet.reset(nullptr, &bad, 1), std::invalid_argument);
}
