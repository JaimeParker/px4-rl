// CpuFleet throughput at N = 4096 with 1, 4 and the OpenMP-default number of threads, printed for the log.
// Nothing is asserted about speed, because it depends on the load of the machine. What is asserted is that the
// thread count does not change a single bit of the results.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"

namespace {

constexpr int kVehicles = 4096;
constexpr int kSteps = 25;
constexpr float kDtPosition = 0.02f;
constexpr float kDtRate = 0.004f;

struct Rng {
	uint32_t state;

	float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}

	float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

// Per-vehicle states and setpoints for one run; the same seed gives the same run.
struct Scenario {
	std::vector<float> position, velocity, heading, acceleration_z, trajectory_sp, q, rates, angular_accel;

	explicit Scenario(uint32_t seed)
	{
		Rng rng{seed};
		const int n = kVehicles;
		position.resize(3 * n);
		velocity.resize(3 * n);
		heading.resize(n);
		acceleration_z.resize(n);
		trajectory_sp.resize(11 * n);
		q.resize(4 * n);
		rates.resize(3 * n);
		angular_accel.resize(3 * n);

		for (int i = 0; i < n; ++i) {
			for (int k = 0; k < 3; ++k) {
				position[3 * i + k] = rng.range(-2.f, 2.f);
				velocity[3 * i + k] = rng.range(-1.f, 1.f);
				rates[3 * i + k] = rng.range(-0.5f, 0.5f);
				angular_accel[3 * i + k] = rng.range(-1.f, 1.f);
			}

			heading[i] = rng.range(-1.f, 1.f);
			acceleration_z[i] = rng.range(-1.f, 1.f);

			// Small tilt about a random axis, normalized.
			const float w = 1.f, x = rng.range(-0.1f, 0.1f), y = rng.range(-0.1f, 0.1f), z = rng.range(-0.1f, 0.1f);
			const float norm = std::sqrt(w * w + x * x + y * y + z * z);
			q[4 * i + 0] = w / norm;
			q[4 * i + 1] = x / norm;
			q[4 * i + 2] = y / norm;
			q[4 * i + 3] = z / norm;

			// Velocity + yaw command: NaN position and acceleration.
			float *sp = &trajectory_sp[11 * i];

			for (int k = 0; k < 3; ++k) {
				sp[k] = NAN;
				sp[6 + k] = NAN;
				sp[3 + k] = rng.range(-2.f, 2.f);
			}

			sp[9] = rng.range(-1.f, 1.f);
			sp[10] = NAN;
		}
	}
};

struct Outputs {
	std::vector<float> q_sp, thrust_sp_body, yawspeed, hover_thrust, rates_sp, thrust_body, motors, actuator,
		torque_sp, force, torque;

	Outputs(int n)
		: q_sp(4 * n), thrust_sp_body(3 * n), yawspeed(n), hover_thrust(n), rates_sp(3 * n), thrust_body(3 * n),
		  motors(4 * n), actuator(4 * n), torque_sp(3 * n), force(3 * n), torque(3 * n)
	{
	}
};

bool sameBits(const std::vector<float> &a, const std::vector<float> &b)
{
	return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct RunResult {
	Outputs outputs;
	double seconds;
};

// kSteps steps of the whole chain (position, attitude, rate, rotor) after one untimed warm-up chain step.
RunResult run(int num_threads)
{
	px4rl::CpuFleet fleet(kVehicles, num_threads);
	px4rl_params params;
	px4rl_params_iris(&params);
	fleet.reset(nullptr, &params, 1);

	const Scenario s(12345u);
	Outputs out(kVehicles);
	uint64_t time_us = 0;

	const auto chain = [&]() {
		time_us += 20000;
		fleet.positionStep(s.position.data(), s.velocity.data(), s.heading.data(), s.acceleration_z.data(),
				   s.trajectory_sp.data(), nullptr, kDtPosition, time_us, out.q_sp.data(),
				   out.thrust_sp_body.data(), out.yawspeed.data(), out.hover_thrust.data(), nullptr);
		fleet.attitudeStep(s.q.data(), nullptr, nullptr, out.thrust_sp_body.data(), out.rates_sp.data(),
				   out.thrust_body.data());
		fleet.rateStep(out.rates_sp.data(), out.thrust_body.data(), s.rates.data(), s.angular_accel.data(),
			       kDtRate, nullptr, out.motors.data(), out.actuator.data(), out.torque_sp.data());
		fleet.rotorStep(out.actuator.data(), nullptr, s.velocity.data(), s.rates.data(), kDtRate,
				out.force.data(), out.torque.data(), nullptr);
	};

	chain();

	const auto start = std::chrono::steady_clock::now();

	for (int step = 0; step < kSteps; ++step) {
		chain();
	}

	const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
	return {std::move(out), elapsed.count()};
}

} // namespace

TEST(CpuFleetThreading, ThreadCountDoesNotChangeResultsAndThroughputIsPrinted)
{
	const int thread_counts[] = {1, 4, 0};
	std::vector<RunResult> results;

	std::printf("CpuFleet chain step (position + attitude + rate + rotor), N = %d, %d steps\n", kVehicles, kSteps);
	std::printf("  %-12s %12s %20s\n", "threads", "ms / step", "vehicle-steps / s");

	for (int threads : thread_counts) {
		results.push_back(run(threads));
		const double seconds_per_step = results.back().seconds / kSteps;
		char label[32];

		if (threads == 0) {
			std::snprintf(label, sizeof(label), "default");

		} else {
			std::snprintf(label, sizeof(label), "%d", threads);
		}

		std::printf("  %-12s %12.3f %20.3e\n", label, 1e3 * seconds_per_step, kVehicles / seconds_per_step);
	}

	const Outputs &reference = results[0].outputs;

	for (size_t k = 1; k < results.size(); ++k) {
		const Outputs &o = results[k].outputs;
		EXPECT_TRUE(sameBits(reference.q_sp, o.q_sp)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.thrust_sp_body, o.thrust_sp_body)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.hover_thrust, o.hover_thrust)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.rates_sp, o.rates_sp)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.motors, o.motors)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.actuator, o.actuator)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.torque_sp, o.torque_sp)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.force, o.force)) << "threads=" << thread_counts[k];
		EXPECT_TRUE(sameBits(reference.torque, o.torque)) << "threads=" << thread_counts[k];
	}

	for (float v : reference.motors) {
		ASSERT_TRUE(std::isfinite(v));
	}
}

TEST(CpuFleetThreading, ConstructorValidatesAndReportsThreadCount)
{
	EXPECT_EQ(px4rl::CpuFleet(8).numThreads(), 0);
	EXPECT_EQ(px4rl::CpuFleet(8, 3).numThreads(), 3);
	EXPECT_THROW(px4rl::CpuFleet(8, -1), std::invalid_argument);
	EXPECT_THROW(px4rl::CpuFleet(0, 1), std::invalid_argument);
}
