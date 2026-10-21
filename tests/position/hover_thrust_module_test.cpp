// HoverThrustModule: convergence on synthetic data, validity hysteresis and gating.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "px4rl/hover_thrust_module.hpp"
#include "px4rl/params.hpp"

namespace {

using matrix::Quatf;
using matrix::Vector3f;
using px4rl::HoverThrustModule;

constexpr float kGravity = 9.80665f;
constexpr float kTrueHoverThrust = 0.35f;
constexpr float kDt = 0.02f; // 50 Hz
constexpr uint64_t kStepUs = 20000;
constexpr uint64_t kStartUs = 5000000;

// Vertical acceleration (NED, positive down) of a vehicle that hovers at kTrueHoverThrust.
float verticalAcceleration(float thrust, float noise)
{
	return kGravity - kGravity * thrust / kTrueHoverThrust + noise;
}

struct Outcome {
	float final_estimate{0.f};
	bool final_valid{false};
	double first_within_tolerance_s{-1.0};
	double first_valid_s{-1.0};
};

Outcome simulate(HoverThrustModule &module, double duration_s, bool armed, bool in_air)
{
	std::mt19937 generator(42);
	std::normal_distribution<float> noise(0.f, 0.05f);
	Outcome run;
	const int steps = static_cast<int>(duration_s / kDt);

	for (int k = 0; k < steps; ++k) {
		const double t = (k + 1) * static_cast<double>(kDt);
		// thrust command around the true hover value, so the measurement is informative
		const float thrust = kTrueHoverThrust + 0.04f * sinf(2.f * 3.14159265f * 0.4f * static_cast<float>(t));

		module.update(armed, in_air, kDt, verticalAcceleration(thrust, noise(generator)), Vector3f(), Quatf(),
			      Vector3f(0.f, 0.f, -thrust), Vector3f(), kStartUs + kStepUs * (k + 1));

		if (run.first_within_tolerance_s < 0.0 && fabsf(module.hoverThrust() - kTrueHoverThrust) <= 0.01f) {
			run.first_within_tolerance_s = t;
		}

		if (run.first_valid_s < 0.0 && module.valid()) {
			run.first_valid_s = t;
		}
	}

	run.final_estimate = module.hoverThrust();
	run.final_valid = module.valid();
	return run;
}

px4rl_params paramsWithInitialHoverThrust(float hover)
{
	px4rl_params p = px4rl::defaultParams();
	p.mpc_thr_hover = hover;
	return p;
}

} // namespace

TEST(HoverThrustModule, ConvergesToTrueHoverThrustWithin10sAndBecomesValid)
{
	HoverThrustModule module;
	module.configure(paramsWithInitialHoverThrust(0.5f));
	EXPECT_FLOAT_EQ(module.hoverThrust(), 0.5f);

	const Outcome run = simulate(module, 10.0, true, true);

	std::printf("hover thrust: final estimate %.4f (true %.2f, initial 0.50), within 0.01 from t = %.2f s, valid from t = %.2f s\n",
		    run.final_estimate, kTrueHoverThrust, run.first_within_tolerance_s, run.first_valid_s);

	EXPECT_NEAR(run.final_estimate, kTrueHoverThrust, 0.01f);
	EXPECT_GT(run.first_within_tolerance_s, 0.0);
	EXPECT_LT(run.first_within_tolerance_s, 10.0);
	EXPECT_TRUE(run.final_valid);
}

TEST(HoverThrustModule, ValidityWaitsForTheTwoSecondHysteresis)
{
	HoverThrustModule module;
	module.configure(paramsWithInitialHoverThrust(0.5f));

	const Outcome run = simulate(module, 10.0, true, true);

	ASSERT_GT(run.first_valid_s, 0.0);
	EXPECT_GE(run.first_valid_s, 2.0);
}

TEST(HoverThrustModule, NoUpdateAndNoValidityWhileNotInAir)
{
	HoverThrustModule module;
	module.configure(paramsWithInitialHoverThrust(0.5f));

	const Outcome run = simulate(module, 5.0, true, false);

	EXPECT_FLOAT_EQ(run.final_estimate, 0.5f);
	EXPECT_FALSE(run.final_valid);
}

TEST(HoverThrustModule, DisarmingResetsTheEstimate)
{
	HoverThrustModule module;
	module.configure(paramsWithInitialHoverThrust(0.5f));
	simulate(module, 5.0, true, true);
	ASSERT_NE(module.hoverThrust(), 0.5f);

	module.update(false, false, kDt, 0.f, Vector3f(), Quatf(), Vector3f(0.f, 0.f, -0.35f), Vector3f(), kStartUs + 60000000);

	EXPECT_FLOAT_EQ(module.hoverThrust(), 0.5f);
	EXPECT_FALSE(module.valid());
}

TEST(HoverThrustModule, UnallocatedThrustIsSubtractedFromTheSetpoint)
{
	// Same acceleration, but half of the commanded thrust was not allocated: the estimator must see
	// thrust 0.35 and converge to the same value as with a plain 0.35 command.
	HoverThrustModule module;
	module.configure(paramsWithInitialHoverThrust(0.5f));

	for (int k = 0; k < 500; ++k) {
		module.update(true, true, kDt, 0.f, Vector3f(), Quatf(), Vector3f(0.f, 0.f, -0.70f), Vector3f(0.f, 0.f, -0.35f),
			      kStartUs + kStepUs * (k + 1));
	}

	EXPECT_NEAR(module.hoverThrust(), kTrueHoverThrust, 0.01f);
}
