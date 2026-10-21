// PositionModule: velocity filters, failsafe and fallback, resetIntegralXY rule, hover thrust update.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "px4rl/params.hpp"
#include "px4rl/position_module.hpp"

namespace {

using matrix::Vector3f;
using px4rl::PositionModule;

constexpr float kDt = 0.02f; // 50 Hz position loop
constexpr float kGravity = 9.80665f;

// Tilt of the commanded body z axis from vertical, rad.
float tiltOf(const vehicle_attitude_setpoint_s &attitude)
{
	const matrix::Dcmf rotation(matrix::Quatf(attitude.q_d));
	return acosf(rotation(2, 2));
}

px4rl_params paramsWithoutFilters()
{
	px4rl_params p = px4rl::defaultParams();
	p.mpc_vel_lp = 0.f;
	p.mpc_veld_lp = 0.f;
	p.mpc_vel_nf_frq = 0.f;
	p.mpc_xy_vel_d_acc = 0.f; // the derivative term would add the filtered-acceleration spike
	return p;
}

trajectory_setpoint_s velocitySetpoint(float vx, float vy, float vz, uint64_t timestamp_us = 0)
{
	trajectory_setpoint_s sp = px4rl::emptyTrajectorySetpoint();
	sp.timestamp = timestamp_us;
	sp.velocity[0] = vx;
	sp.velocity[1] = vy;
	sp.velocity[2] = vz;
	return sp;
}

PositionModule makeModule(const px4rl_params &p)
{
	PositionModule module;
	module.configure(p, kDt);
	module.setTiltLimitStateDeg(p.mpc_tiltmax_air); // the vehicle is already in flight
	return module;
}

const Vector3f kHoverPosition{0.f, 0.f, -5.f};

} // namespace

TEST(PositionModule, HoverIsLevelWithHoverThrust)
{
	const px4rl_params p = px4rl::defaultParams();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};

	const auto outcome = module.update(kHoverPosition, Vector3f(), 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f), 1000000, attitude);

	EXPECT_EQ(outcome, PositionModule::Outcome::Ok);
	EXPECT_NEAR(tiltOf(attitude), 0.f, 1e-5f);
	EXPECT_NEAR(attitude.thrust_body[2], -p.mpc_thr_hover, 1e-5f);
	EXPECT_EQ(attitude.timestamp, 1000000u);
}

TEST(PositionModule, TiltLimitStartsAtZeroAndSlewsLikePx4)
{
	const px4rl_params p = paramsWithoutFilters();
	PositionModule module;
	module.configure(p, kDt); // no forced tilt state: PX4 starts the slew rate at 0
	vehicle_attitude_setpoint_s attitude{};

	module.update(kHoverPosition, Vector3f(), 0.f, kDt, velocitySetpoint(5.f, 0.f, 0.f), 1000000, attitude);

	// 0.2 rad/s * 0.02 s
	EXPECT_NEAR(tiltOf(attitude), 0.004f, 1e-5f);
}

TEST(PositionModule, DisabledVelocityFilterPassesTheMeasurementThrough)
{
	const px4rl_params p = paramsWithoutFilters();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};

	module.update(kHoverPosition, Vector3f(2.f, 0.f, 0.f), 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f), 1000000, attitude);

	// acc_x = P * (0 - 2) with P = 1.8 and no filtering
	EXPECT_NEAR(tiltOf(attitude), atanf(2.f * p.mpc_xy_vel_p_acc / kGravity), 1e-4f);
}

TEST(PositionModule, EnabledVelocityLowPassSoftensAMeasurementStep)
{
	const px4rl_params disabled = paramsWithoutFilters();
	px4rl_params enabled = disabled;
	enabled.mpc_vel_lp = 5.f;

	vehicle_attitude_setpoint_s attitude_disabled{};
	vehicle_attitude_setpoint_s attitude_enabled{};
	makeModule(disabled).update(kHoverPosition, Vector3f(2.f, 0.f, 0.f), 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f), 1000000,
				    attitude_disabled);
	makeModule(enabled).update(kHoverPosition, Vector3f(2.f, 0.f, 0.f), 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f), 1000000,
				   attitude_enabled);

	EXPECT_GT(tiltOf(attitude_enabled), 0.f);
	EXPECT_LT(tiltOf(attitude_enabled), 0.7f * tiltOf(attitude_disabled));
}

TEST(PositionModule, EnabledNotchFilterRemovesTheNotchFrequency)
{
	const px4rl_params disabled = paramsWithoutFilters();
	px4rl_params enabled = disabled;
	enabled.mpc_vel_nf_frq = 10.f;
	enabled.mpc_vel_nf_bw = 5.f;

	float peak_tilt[2] = {0.f, 0.f};
	const px4rl_params *const configs[2] = {&disabled, &enabled};

	for (int c = 0; c < 2; ++c) {
		PositionModule module = makeModule(*configs[c]);
		vehicle_attitude_setpoint_s attitude{};

		for (int k = 0; k < 150; ++k) {
			const float vx = 0.5f * sinf(2.f * 3.14159265f * 10.f * k * kDt);
			module.update(kHoverPosition, Vector3f(vx, 0.f, 0.f), 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f),
				      1000000u + 20000u * k, attitude);

			if (k >= 100) {
				peak_tilt[c] = fmaxf(peak_tilt[c], tiltOf(attitude));
			}
		}
	}

	EXPECT_GT(peak_tilt[0], 0.05f);
	EXPECT_LT(peak_tilt[1], 0.25f * peak_tilt[0]);
}

TEST(PositionModule, AllNanSetpointUsesFailsafeAndHoldsPosition)
{
	const px4rl_params p = px4rl::defaultParams();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};

	const auto outcome = module.update(kHoverPosition, Vector3f(), 0.f, kDt, px4rl::emptyTrajectorySetpoint(), 5000000, attitude);

	EXPECT_EQ(outcome, PositionModule::Outcome::Failsafe);
	// stop and wait: zero velocity setpoint on all axes
	EXPECT_NEAR(tiltOf(attitude), 0.f, 1e-5f);
	EXPECT_NEAR(attitude.thrust_body[2], -p.mpc_thr_hover, 1e-5f);
}

TEST(PositionModule, FailsafeWithoutVelocityEstimateDescendsBelowHoverThrust)
{
	const px4rl_params p = px4rl::defaultParams();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};
	const Vector3f unknown_velocity{NAN, NAN, NAN};

	const auto outcome = module.update(kHoverPosition, unknown_velocity, 0.f, kDt, px4rl::emptyTrajectorySetpoint(), 5000000, attitude);

	EXPECT_EQ(outcome, PositionModule::Outcome::Failsafe);
	// blind descent: acc_z = 0.3, thrust = 0.3 * hover / g - hover
	EXPECT_NEAR(attitude.thrust_body[2], 0.3f * p.mpc_thr_hover / kGravity - p.mpc_thr_hover, 1e-5f);
	EXPECT_NEAR(tiltOf(attitude), 0.f, 1e-5f);
	for (float q : attitude.q_d) {
		EXPECT_TRUE(std::isfinite(q));
	}
}

TEST(PositionModule, InvalidSetpointFallsBackToTheLastValidOneFor200ms)
{
	const px4rl_params p = paramsWithoutFilters();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};
	const Vector3f velocity{0.f, 0.f, 0.f};

	// valid command: 1 m/s forward, stamped 1.000 s
	EXPECT_EQ(module.update(kHoverPosition, velocity, 0.f, kDt, velocitySetpoint(1.f, 0.f, 0.f, 1000000), 1000000, attitude),
		  PositionModule::Outcome::Ok);
	const float valid_tilt = tiltOf(attitude);
	EXPECT_GT(valid_tilt, 0.05f);

	// invalid command 100 ms later: the last valid setpoint is executed
	EXPECT_EQ(module.update(kHoverPosition, velocity, 0.f, kDt, px4rl::emptyTrajectorySetpoint(), 1100000, attitude),
		  PositionModule::Outcome::Fallback);
	EXPECT_GT(tiltOf(attitude), 0.05f);

	// invalid command 300 ms later: failsafe, the vehicle is held
	EXPECT_EQ(module.update(kHoverPosition, velocity, 0.f, kDt, px4rl::emptyTrajectorySetpoint(), 1300000, attitude),
		  PositionModule::Outcome::Failsafe);
	EXPECT_LT(tiltOf(attitude), 0.5f * valid_tilt);
}

class ResetIntegralXyTest : public ::testing::Test
{
protected:
	// Accumulates a horizontal velocity integral, optionally passes through a setpoint that does not
	// control horizontal velocity, then commands zero velocity error and returns the resulting tilt.
	float tiltAfter(bool pass_through_uncontrolled_xy)
	{
		const px4rl_params p = paramsWithoutFilters();
		PositionModule module = makeModule(p);
		vehicle_attitude_setpoint_s attitude{};
		const Vector3f velocity{0.f, 0.f, 0.f};
		uint64_t time_us = 1000000;

		for (int k = 0; k < 25; ++k, time_us += 20000) {
			module.update(kHoverPosition, velocity, 0.f, kDt, velocitySetpoint(1.f, 0.f, 0.f, time_us), time_us, attitude);
		}

		if (pass_through_uncontrolled_xy) {
			// xy controlled by acceleration only, z by velocity
			trajectory_setpoint_s sp = px4rl::emptyTrajectorySetpoint();
			sp.timestamp = time_us;
			sp.acceleration[0] = sp.acceleration[1] = 0.f;
			sp.velocity[2] = 0.f;
			EXPECT_EQ(module.update(kHoverPosition, velocity, 0.f, kDt, sp, time_us, attitude), PositionModule::Outcome::Ok);
			time_us += 20000;
		}

		module.update(kHoverPosition, velocity, 0.f, kDt, velocitySetpoint(0.f, 0.f, 0.f, time_us), time_us, attitude);
		return tiltOf(attitude);
	}
};

TEST_F(ResetIntegralXyTest, IntegralSurvivesWhenHorizontalVelocityStaysControlled)
{
	// 25 steps * 0.02 s * I = 0.4 * error 1 m/s => about 0.2 m/s^2 left in the integrator
	EXPECT_NEAR(tiltAfter(false), atanf(0.2f / kGravity), 2e-3f);
}

TEST_F(ResetIntegralXyTest, IntegralIsClearedWhenHorizontalVelocityIsNotControlled)
{
	EXPECT_NEAR(tiltAfter(true), 0.f, 1e-5f);
}

TEST(PositionModule, HoverThrustEstimateIsAbsorbedWithoutAStepInThrust)
{
	const px4rl_params p = px4rl::defaultParams();
	PositionModule module = makeModule(p);
	vehicle_attitude_setpoint_s attitude{};
	const trajectory_setpoint_s hold = velocitySetpoint(0.f, 0.f, 0.f);

	module.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1000000, attitude);
	module.applyHoverThrustEstimate(0.4f, true);
	module.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1020000, attitude);

	// PositionControl::updateHoverThrust moves the difference into the integrator
	EXPECT_NEAR(attitude.thrust_body[2], -p.mpc_thr_hover, 1e-4f);
}

TEST(PositionModule, HoverThrustEstimateIsIgnoredWhenDisabledOrInvalid)
{
	px4rl_params p = px4rl::defaultParams();
	const trajectory_setpoint_s hold = velocitySetpoint(0.f, 0.f, 0.f);
	vehicle_attitude_setpoint_s attitude{};

	PositionModule invalid = makeModule(p);
	invalid.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1000000, attitude);
	invalid.applyHoverThrustEstimate(0.4f, false);
	invalid.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1020000, attitude);
	EXPECT_NEAR(attitude.thrust_body[2], -p.mpc_thr_hover, 1e-6f);

	p.mpc_use_hte = 0;
	PositionModule disabled = makeModule(p);
	disabled.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1000000, attitude);
	disabled.applyHoverThrustEstimate(0.4f, true);
	disabled.update(kHoverPosition, Vector3f(), 0.f, kDt, hold, 1020000, attitude);
	EXPECT_NEAR(attitude.thrust_body[2], -p.mpc_thr_hover, 1e-6f);
}
