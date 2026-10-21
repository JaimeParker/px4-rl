// Closed loop of the full CpuFleet chain (position -> attitude -> rate -> allocator -> PWM) with the Gazebo
// Iris rotor plant (CpuFleet::rotorStep) and a rigid body integrated here, the way a GPU simulator would:
// world ENU, body FLU, Iris mass and inertia (px4rl_rigid_body_params_iris), gravity 9.80665 m/s^2,
// semi-implicit Euler. The controllers see PX4's frames (NED / FRD), converted with frames.hpp, and the
// rotor wrench returns to the body in FLU, so the whole frame path is exercised.
//
// Rates: physics, attitude and rate control at 250 Hz (4 ms, the step size of PX4's Gazebo SITL worlds:
// gazebo_mavlink_interface.cpp:380-381 requires a multiple of 250 Hz for lockstep), position control at 50 Hz.
//
// Initial condition. A flight starts from rest on the ground in PX4 SITL; here the vehicle starts in the air
// and the rotors are first spun up to the hover command with the body held still. The hover test starts
// with PX4's default MPC_THR_HOVER = 0.5, far below the Iris hover command, so the hover thrust estimator has
// to converge. The velocity and acceleration tests start from a converged hover: from the cold start an
// acceleration-only command has no position or velocity feedback and would fall while the estimate converges.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>

#include "px4rl/airframes.h"
#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"
#include "px4rl/frames.hpp"

namespace {

using matrix::Quatf;
using matrix::Vector3f;

constexpr float kGravity = 9.80665f;
constexpr float kPi = 3.14159265f;
constexpr float kDt = 0.004f;
constexpr int kStepsPerPosition = 5;
constexpr float kPositionDt = kDt * kStepsPerPosition;
constexpr uint64_t kStartTimeUs = 1000000;
constexpr float kHoverAltitude = 10.f;

float wrapPi(float angle)
{
	while (angle > kPi) {
		angle -= 2.f * kPi;
	}

	while (angle < -kPi) {
		angle += 2.f * kPi;
	}

	return angle;
}

// Hover motor command u of the plant: 4 k_f (1000 u + 100)^2 = m g, from the Iris parameters.
float theoreticalHoverU()
{
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();
	const px4rl_rotor_params &rotor = plant.rotor[0];
	const float rotor_speed = std::sqrt(body.mass * kGravity / (static_cast<float>(plant.num_rotors) * rotor.motor_constant));
	return (rotor_speed - rotor.zero_position_armed) / rotor.input_scaling;
}

// World ENU, body FLU. The attitude is the body FLU -> world ENU quaternion.
class RigidBody {
public:
	RigidBody(const Vector3f &start_position_enu, float start_heading_enu)
		: position_(start_position_enu), attitude_(matrix::Eulerf(0.f, 0.f, start_heading_enu))
	{
		const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();
		mass_ = body.mass;
		inertia_ = matrix::SquareMatrix3f(body.inertia);
		inertia_inverse_ = inertia_.I();
	}

	void step(const Vector3f &force_flu, const Vector3f &torque_flu, float dt)
	{
		acceleration_ = matrix::Dcmf(attitude_) * force_flu / mass_ + Vector3f(0.f, 0.f, -kGravity);
		const Vector3f rates_dot = inertia_inverse_ * (torque_flu - rates_.cross(inertia_ * rates_));

		rates_ += rates_dot * dt;
		velocity_ += acceleration_ * dt;
		position_ += velocity_ * dt;
		attitude_ = attitude_ * Quatf(matrix::AxisAnglef(rates_ * dt));
		attitude_.normalize();
	}

	const Vector3f &position() const { return position_; }
	const Vector3f &velocity() const { return velocity_; }
	const Vector3f &rates() const { return rates_; }
	const Vector3f &acceleration() const { return acceleration_; }
	const Quatf &attitude() const { return attitude_; }
	Vector3f velocityBody() const { return matrix::Dcmf(attitude_).transpose() * velocity_; }

private:
	float mass_{0.f};
	matrix::SquareMatrix3f inertia_;
	matrix::SquareMatrix3f inertia_inverse_;
	Vector3f position_;
	Vector3f velocity_;
	Vector3f rates_;
	Vector3f acceleration_;
	Quatf attitude_;
};

using TrajectorySetpoint = std::array<float, 11>; // position[3] velocity[3] acceleration[3] yaw yawspeed (NED)

TrajectorySetpoint positionYaw(float north, float east, float down, float yaw)
{
	TrajectorySetpoint sp;
	sp.fill(NAN);
	sp[0] = north;
	sp[1] = east;
	sp[2] = down;
	sp[9] = yaw;
	return sp;
}

TrajectorySetpoint velocityYaw(float vn, float ve, float vd, float yaw)
{
	TrajectorySetpoint sp;
	sp.fill(NAN);
	sp[3] = vn;
	sp[4] = ve;
	sp[5] = vd;
	sp[9] = yaw;
	return sp;
}

TrajectorySetpoint accelerationYaw(float an, float ae, float ad, float yaw)
{
	TrajectorySetpoint sp;
	sp.fill(NAN);
	sp[6] = an;
	sp[7] = ae;
	sp[8] = ad;
	sp[9] = yaw;
	return sp;
}

class IrisLoop {
public:
	IrisLoop(const px4rl_params &params, const Vector3f &start_position_enu, float start_heading_enu)
		: fleet_(1), body_(start_position_enu, start_heading_enu)
	{
		fleet_.reset(nullptr, &params, 1);
		const px4rl_plant_params plant = px4rl_plant_params_iris();
		fleet_.resetPlant(nullptr, &plant, 1);

		// Spin the rotors up to the hover command with the body held still.
		const float hover_pwm = std::round(1000.f + 1000.f * theoreticalHoverU());
		std::fill(pwm_, pwm_ + 4, hover_pwm);
		const float zero[3] = {0.f, 0.f, 0.f};
		float force[3], torque[3];

		for (int k = 0; k < 250; ++k) {
			fleet_.rotorStep(pwm_, nullptr, zero, nullptr, kDt, force, torque, nullptr);
		}

		hover_force_z_ = force[2];
	}

	// Runs `duration_s`; `command(t)` gives the NED trajectory setpoint, `observe(*this)` is called every step.
	void run(float duration_s, const std::function<TrajectorySetpoint(float)> &command,
		 const std::function<void(const IrisLoop &)> &observe)
	{
		const int steps = static_cast<int>(std::lround(duration_s / kDt));

		for (int k = 0; k < steps; ++k, ++tick_) {
			const uint64_t time_us = kStartTimeUs + static_cast<uint64_t>(tick_) * 4000u;

			if (tick_ % kStepsPerPosition == 0) {
				positionStep(command(time_s()), time_us);
			}

			controlStep();
			plantStep();
			observe(*this);
		}
	}

	float time_s() const { return static_cast<float>(tick_) * kDt; }
	const RigidBody &body() const { return body_; }
	float hoverThrustEstimate() const { return hover_thrust_; }
	float collectiveThrust() const { return collective_thrust_; }
	const int *outcomeCounts() const { return outcome_counts_; }
	float hoverForceZ() const { return hover_force_z_; }

	// PX4 heading of the body, rad.
	float heading() const { return px4rl::headingNedFromEnuFlu(body_.attitude()); }

private:
	void positionStep(const TrajectorySetpoint &setpoint, uint64_t time_us)
	{
		const Vector3f position = px4rl::enuToNed(body_.position());
		const Vector3f velocity = px4rl::enuToNed(body_.velocity());
		const float position_ned[3] = {position(0), position(1), position(2)};
		const float velocity_ned[3] = {velocity(0), velocity(1), velocity(2)};
		const float heading_ned = heading();
		const float acceleration_z_ned = px4rl::enuToNed(body_.acceleration())(2);
		uint8_t outcome = 0;
		fleet_.positionStep(position_ned, velocity_ned, &heading_ned, &acceleration_z_ned, setpoint.data(), nullptr,
				    kPositionDt, time_us, q_sp_, thrust_sp_body_, &yawspeed_, &hover_thrust_, &outcome);
		++outcome_counts_[outcome];
	}

	void controlStep()
	{
		const Quatf q = px4rl::quatWorldBodyEnuFluToNedFrd(body_.attitude());
		const float attitude[4] = {q(0), q(1), q(2), q(3)};
		float rates_sp[3], thrust_body[3];
		fleet_.attitudeStep(attitude, nullptr, nullptr, thrust_sp_body_, rates_sp, thrust_body);
		collective_thrust_ = -thrust_body[2];

		const Vector3f rates_frd = px4rl::fluToFrd(body_.rates());
		const float rates[3] = {rates_frd(0), rates_frd(1), rates_frd(2)};
		const Vector3f accel = (rates_frd - previous_rates_frd_) / kDt;
		const float angular_accel[3] = {accel(0), accel(1), accel(2)};
		previous_rates_frd_ = rates_frd;
		float motors[4];
		fleet_.rateStep(rates_sp, thrust_body, rates, angular_accel, kDt, nullptr, motors, pwm_, nullptr);
	}

	void plantStep()
	{
		const Vector3f velocity_body = body_.velocityBody();
		const float velocity[3] = {velocity_body(0), velocity_body(1), velocity_body(2)};
		const Vector3f &w = body_.rates();
		const float rates[3] = {w(0), w(1), w(2)};
		float force[3], torque[3];
		fleet_.rotorStep(pwm_, nullptr, velocity, rates, kDt, force, torque, nullptr);
		body_.step(Vector3f(force), Vector3f(torque), kDt);
	}

	px4rl::CpuFleet fleet_;
	RigidBody body_;
	int tick_{0};
	float pwm_[4];
	float q_sp_[4]{};
	float thrust_sp_body_[3]{};
	float yawspeed_{0.f};
	float hover_thrust_{0.f};
	float collective_thrust_{0.f};
	float hover_force_z_{0.f};
	Vector3f previous_rates_frd_;
	int outcome_counts_[3]{};
};

constexpr float kNorthHeadingEnu = kPi / 2.f; // body x points north: PX4 heading 0

px4rl_params irisParams()
{
	px4rl_params params;
	px4rl_params_iris(&params);
	return params;
}

// 20 s hover at the hover altitude from the cold start (MPC_THR_HOVER = 0.5), so that the hover thrust estimate converges.
void settleHover(IrisLoop &loop)
{
	loop.run(20.f, [](float) { return positionYaw(0.f, 0.f, -kHoverAltitude, 0.f); }, [](const IrisLoop &) {});
}

} // namespace

TEST(IrisClosedLoop, RotorsAtHoverCommandBalanceWeight)
{
	IrisLoop loop(irisParams(), Vector3f(0.f, 0.f, kHoverAltitude), kNorthHeadingEnu);
	const float weight = px4rl_rigid_body_params_iris().mass * kGravity;
	std::printf("iris hover: theoretical u = %.4f, force at the rounded PWM = %.4f N, weight = %.4f N, initial heading = "
		    "%.4f rad\n",
		    theoreticalHoverU(), loop.hoverForceZ(), weight, loop.heading());
	// The PWM is an integer, so u is quantised to 0.001, which is 0.3 % in thrust.
	EXPECT_NEAR(loop.hoverForceZ(), weight, 0.01f * weight);
	EXPECT_NEAR(loop.heading(), 0.f, 1e-5f);
}

TEST(IrisClosedLoop, HoverHoldsAndHoverThrustEstimateConverges)
{
	IrisLoop loop(irisParams(), Vector3f(0.f, 0.f, kHoverAltitude), kNorthHeadingEnu);

	float max_horizontal_drift = 0.f;
	float max_altitude_error_after_5s = 0.f;
	float collective_sum = 0.f;
	int collective_count = 0;
	loop.run(20.f, [](float) { return positionYaw(0.f, 0.f, -kHoverAltitude, 0.f); }, [&](const IrisLoop &l) {
		const Vector3f &p = l.body().position();
		max_horizontal_drift = std::max(max_horizontal_drift, std::hypot(p(0), p(1)));

		if (l.time_s() > 5.f) {
			max_altitude_error_after_5s = std::max(max_altitude_error_after_5s, std::fabs(p(2) - kHoverAltitude));
		}

		if (l.time_s() > 15.f) {
			collective_sum += l.collectiveThrust();
			++collective_count;
		}
	});

	const float measured_hover_thrust = collective_sum / static_cast<float>(collective_count);
	const float estimate = loop.hoverThrustEstimate();
	std::printf("iris hover 20 s: max horizontal drift = %.4f m, max |altitude error| after 5 s = %.4f m, hover thrust "
		    "estimate = %.4f, measured collective thrust (mean of 15-20 s) = %.4f, theoretical motor u = %.4f, "
		    "outcomes ok/fallback/failsafe = %d/%d/%d\n",
		    max_horizontal_drift, max_altitude_error_after_5s, estimate, measured_hover_thrust, theoreticalHoverU(),
		    loop.outcomeCounts()[0], loop.outcomeCounts()[1], loop.outcomeCounts()[2]);
	std::printf("iris hover: estimator upper limit MPC_THR_HOVER + HTE_THR_RANGE = 0.7; estimate %s the limit\n",
		    estimate >= 0.7f - 1e-3f ? "is at" : "is below");
	EXPECT_LE(max_horizontal_drift, 0.3f);
	EXPECT_LE(max_altitude_error_after_5s, 0.3f);
	EXPECT_NEAR(estimate, measured_hover_thrust, 0.03f);
	EXPECT_EQ(loop.outcomeCounts()[1] + loop.outcomeCounts()[2], 0);
}

TEST(IrisClosedLoop, VelocityStepInNorthOvershootsLittleAndSettles)
{
	IrisLoop loop(irisParams(), Vector3f(0.f, 0.f, kHoverAltitude), kNorthHeadingEnu);
	settleHover(loop);

	const float start_altitude = loop.body().position()(2);
	float peak_overshoot = 0.f;
	float altitude_deviation = 0.f;
	float max_heading_error = 0.f;
	loop.run(8.f, [](float) { return velocityYaw(1.f, 0.f, 0.f, 0.f); }, [&](const IrisLoop &l) {
		// ENU y is north.
		peak_overshoot = std::max(peak_overshoot, l.body().velocity()(1) - 1.f);
		altitude_deviation = std::max(altitude_deviation, std::fabs(l.body().position()(2) - start_altitude));
		max_heading_error = std::max(max_heading_error, std::fabs(wrapPi(l.heading())));
	});
	const float error_at_8s = std::fabs(loop.body().velocity()(1) - 1.f);

	std::printf("iris vel step: peak overshoot = %.4f m/s, |vx - 1| at 8 s = %.4f m/s, max altitude deviation = %.4f m, "
		    "max |yaw error| = %.4f rad, hover thrust estimate = %.4f, outcomes ok/fallback/failsafe = %d/%d/%d\n",
		    peak_overshoot, error_at_8s, altitude_deviation, max_heading_error, loop.hoverThrustEstimate(),
		    loop.outcomeCounts()[0], loop.outcomeCounts()[1], loop.outcomeCounts()[2]);
	EXPECT_LE(peak_overshoot, 0.2f);
	EXPECT_LE(error_at_8s, 0.05f);
	EXPECT_LE(altitude_deviation, 0.3f);
}

TEST(IrisClosedLoop, AccelerationHoverTracksYawStep)
{
	IrisLoop loop(irisParams(), Vector3f(0.f, 0.f, kHoverAltitude), kNorthHeadingEnu);
	settleHover(loop);

	const Vector3f start_position = loop.body().position();
	float max_drift = 0.f;
	loop.run(3.f, [](float) { return accelerationYaw(0.f, 0.f, 0.f, 1.f); }, [&](const IrisLoop &l) {
		max_drift = std::max(max_drift, (l.body().position() - start_position).norm());
	});
	const float yaw_error = std::fabs(wrapPi(loop.heading() - 1.f));

	std::printf("iris acc+yaw: |yaw error| 3 s after a +1 rad step = %.4f rad, max drift from the start = %.4f m, hover "
		    "thrust estimate = %.4f\n",
		    yaw_error, max_drift, loop.hoverThrustEstimate());
	EXPECT_LE(yaw_error, 0.05f);
}
