// CpuFleet must agree bitwise with a hand-written loop over the px4_stack wrappers.
#include <gtest/gtest.h>

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
}
