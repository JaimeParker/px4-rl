// Closed-loop tests of the full CpuFleet chain (position -> attitude -> rate -> allocator) against a
// minimal rigid-body plant that this file owns: quad-X with the Iris geometry and inertia, thrust linear
// in the motor command (hover at 0.5), no motor lag, no drag. It checks the controller wiring, not the
// vehicle model.
//
// Rates: physics 1 kHz, rate and attitude control 250 Hz, position control 50 Hz.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>

#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"

namespace {

using matrix::Quatf;
using matrix::Vector3f;

constexpr float kGravity = 9.80665f;
constexpr float kPi = 3.14159265f;
constexpr float kPhysicsDt = 0.001f;
constexpr int kPhysicsPerRate = 4;
constexpr int kRatePerPosition = 5;
constexpr float kRateDt = kPhysicsDt * kPhysicsPerRate;
constexpr float kPositionDt = kRateDt * kRatePerPosition;
constexpr uint64_t kStartTimeUs = 1000000;

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

// NED world, FRD body. Rotor i produces thrust T_i = u_i * T_max along its axis and the torque
// T_i * (position x axis - km * axis), the same sign convention as the control allocator.
class RigidBody {
public:
	explicit RigidBody(const px4rl_params &p, const Vector3f &start_position) : position_(start_position)
	{
		for (int i = 0; i < 4; ++i) {
			rotor_position_[i] = Vector3f(p.rotor_position[i][0], p.rotor_position[i][1], p.rotor_position[i][2]);
			rotor_axis_[i] = Vector3f(p.rotor_axis[i][0], p.rotor_axis[i][1], p.rotor_axis[i][2]);
			rotor_km_[i] = p.rotor_km[i];
		}
	}

	void step(const float *motors, float dt)
	{
		Vector3f force_body;
		Vector3f torque;

		for (int i = 0; i < 4; ++i) {
			const float thrust = motors[i] * kMaxThrustPerRotor;
			force_body += thrust * rotor_axis_[i];
			torque += thrust * (rotor_position_[i].cross(rotor_axis_[i]) - rotor_km_[i] * rotor_axis_[i]);
		}

		const matrix::Dcmf rotation(attitude_);
		acceleration_ = Vector3f(0.f, 0.f, kGravity) + rotation * force_body / kMass;

		const Vector3f inertia_omega(kInertia(0) * rates_(0), kInertia(1) * rates_(1), kInertia(2) * rates_(2));
		const Vector3f rates_dot = (torque - rates_.cross(inertia_omega)).edivide(kInertia);

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
	float yaw() const { return matrix::Eulerf(attitude_).psi(); }

private:
	static constexpr float kMass = 1.5f;
	// Hover at motor command 0.5.
	static constexpr float kMaxThrustPerRotor = kMass * kGravity / 2.f;
	static inline const Vector3f kInertia{0.029125f, 0.029125f, 0.055225f};

	Vector3f position_;
	Vector3f velocity_;
	Vector3f rates_;
	Vector3f acceleration_;
	Quatf attitude_;
	Vector3f rotor_position_[4];
	Vector3f rotor_axis_[4];
	float rotor_km_[4]{};
};

using TrajectorySetpoint = std::array<float, 11>; // position[3] velocity[3] acceleration[3] yaw yawspeed

TrajectorySetpoint velocityYaw(float vx, float vy, float vz, float yaw)
{
	TrajectorySetpoint sp;
	sp.fill(NAN);
	sp[3] = vx;
	sp[4] = vy;
	sp[5] = vz;
	sp[9] = yaw;
	return sp;
}

TrajectorySetpoint accelerationYaw(float ax, float ay, float az, float yaw)
{
	TrajectorySetpoint sp;
	sp.fill(NAN);
	sp[6] = ax;
	sp[7] = ay;
	sp[8] = az;
	sp[9] = yaw;
	return sp;
}

class ClosedLoop {
public:
	ClosedLoop(const px4rl_params &params, const Vector3f &start_position)
		: fleet_(1), body_(params, start_position)
	{
		fleet_.reset(nullptr, &params, 1);
		std::fill(motors_, motors_ + 4, 0.5f);
	}

	// Runs `duration_s`; `command(t)` gives the trajectory setpoint, `observe(*this)` is called every physics step.
	void run(float duration_s, const std::function<TrajectorySetpoint(float)> &command,
		 const std::function<void(const ClosedLoop &)> &observe)
	{
		const int steps = static_cast<int>(std::lround(duration_s / kPhysicsDt));

		for (int k = 0; k < steps; ++k, ++tick_) {
			if (tick_ % kPhysicsPerRate == 0) {
				controlStep(command(time_s()));
			}

			body_.step(motors_, kPhysicsDt);
			observe(*this);
		}
	}

	float time_s() const { return static_cast<float>(tick_) * kPhysicsDt; }
	const RigidBody &body() const { return body_; }
	float hoverThrustEstimate() const { return hover_thrust_; }
	const int *outcomeCounts() const { return outcome_counts_; }

private:
	void controlStep(const TrajectorySetpoint &setpoint)
	{
		const uint64_t time_us = kStartTimeUs + static_cast<uint64_t>(tick_) * 1000u;

		if ((tick_ / kPhysicsPerRate) % kRatePerPosition == 0) {
			const Vector3f &p = body_.position();
			const Vector3f &v = body_.velocity();
			const float position[3] = {p(0), p(1), p(2)};
			const float velocity[3] = {v(0), v(1), v(2)};
			const float heading = body_.yaw();
			const float acceleration_z = body_.acceleration()(2);
			uint8_t outcome = 0;
			fleet_.positionStep(position, velocity, &heading, &acceleration_z, setpoint.data(), nullptr,
					    kPositionDt, time_us, q_sp_, thrust_sp_body_, &yawspeed_, &hover_thrust_, &outcome);
			++outcome_counts_[outcome];
		}

		const Quatf &q = body_.attitude();
		const float attitude[4] = {q(0), q(1), q(2), q(3)};
		float rates_sp[3], thrust_body[3];
		fleet_.attitudeStep(attitude, nullptr, nullptr, thrust_sp_body_, rates_sp, thrust_body);

		const Vector3f &w = body_.rates();
		const float rates[3] = {w(0), w(1), w(2)};
		const float angular_accel[3] = {(w(0) - previous_rates_(0)) / kRateDt, (w(1) - previous_rates_(1)) / kRateDt,
						(w(2) - previous_rates_(2)) / kRateDt};
		previous_rates_ = w;
		fleet_.rateStep(rates_sp, thrust_body, rates, angular_accel, kRateDt, nullptr, motors_, nullptr, nullptr);
	}

	px4rl::CpuFleet fleet_;
	RigidBody body_;
	int tick_{0};
	float motors_[4];
	float q_sp_[4]{};
	float thrust_sp_body_[3]{};
	float yawspeed_{0.f};
	float hover_thrust_{0.f};
	Vector3f previous_rates_;
	int outcome_counts_[3]{};
};

} // namespace

TEST(ClosedLoop, VelocityStepInXHoldsAltitude)
{
	px4rl_params params;
	px4rl_params_default(&params);
	ClosedLoop loop(params, Vector3f(0.f, 0.f, -5.f));

	float altitude_deviation = 0.f;
	float yaw_error = 0.f;
	float peak_overshoot = 0.f;
	loop.run(3.f, [](float) { return velocityYaw(1.f, 0.f, 0.f, 0.f); }, [&](const ClosedLoop &l) {
		altitude_deviation = std::max(altitude_deviation, std::fabs(l.body().position()(2) + 5.f));
		yaw_error = std::max(yaw_error, std::fabs(wrapPi(l.body().yaw())));
		peak_overshoot = std::max(peak_overshoot, l.body().velocity()(0) - 1.f);
	});
	const float velocity_error_at_3s = std::fabs(loop.body().velocity()(0) - 1.f);

	loop.run(5.f, [](float) { return velocityYaw(1.f, 0.f, 0.f, 0.f); }, [&](const ClosedLoop &l) {
		altitude_deviation = std::max(altitude_deviation, std::fabs(l.body().position()(2) + 5.f));
		peak_overshoot = std::max(peak_overshoot, l.body().velocity()(0) - 1.f);
	});
	const float velocity_error_at_8s = std::fabs(loop.body().velocity()(0) - 1.f);

	std::printf("vel step: |vx - 1| at 3 s = %.4f m/s, at 8 s = %.4f m/s, peak overshoot %.4f, max altitude deviation = "
		    "%.4f m, max |yaw| = %.4f rad, outcomes ok/fallback/failsafe = %d/%d/%d\n",
		    velocity_error_at_3s, velocity_error_at_8s, peak_overshoot, altitude_deviation, yaw_error,
		    loop.outcomeCounts()[0], loop.outcomeCounts()[1], loop.outcomeCounts()[2]);
	// PX4's default xy velocity PI (MPC_XY_VEL_P_ACC 1.8, MPC_XY_VEL_I_ACC 0.4) overshoots by about 10 %
	// on this drag-free plant and then settles slowly, so the criterion is a step-response envelope.
	EXPECT_LE(peak_overshoot, 0.15f);
	EXPECT_LE(velocity_error_at_8s, 0.05f);
	EXPECT_LE(altitude_deviation, 0.2f);
	EXPECT_EQ(loop.outcomeCounts()[1] + loop.outcomeCounts()[2], 0);
}

TEST(ClosedLoop, AccelerationHoverTracksYawSteps)
{
	px4rl_params params;
	px4rl_params_default(&params);
	ClosedLoop loop(params, Vector3f(0.f, 0.f, -5.f));

	const auto command = [](float t) { return accelerationYaw(0.f, 0.f, 0.f, t < 3.f ? 1.0f : -0.5f); };
	float drift = 0.f;
	const auto observe = [&](const ClosedLoop &l) {
		drift = std::max(drift, (l.body().position() - Vector3f(0.f, 0.f, -5.f)).norm());
	};

	loop.run(3.f, command, observe);
	const float first_error = std::fabs(wrapPi(loop.body().yaw() - 1.0f));
	loop.run(3.f, command, observe);
	const float second_error = std::fabs(wrapPi(loop.body().yaw() + 0.5f));

	std::printf("yaw steps: |yaw error| at 3 s = %.4f rad, 3 s after the second step = %.4f rad, max drift = %.4f m\n",
		    first_error, second_error, drift);
	EXPECT_LE(first_error, 0.05f);
	EXPECT_LE(second_error, 0.05f);
	EXPECT_LE(drift, 0.5f);
}

TEST(ClosedLoop, HoverThrustEstimateConvergesFromWrongParameter)
{
	px4rl_params params;
	px4rl_params_default(&params);
	params.mpc_thr_hover = 0.3f; // the plant hovers at 0.5
	ClosedLoop loop(params, Vector3f(0.f, 0.f, -100.f));

	float first_within_tolerance_s = -1.f;
	loop.run(60.f, [](float) { return velocityYaw(0.f, 0.f, 0.f, 0.f); }, [&](const ClosedLoop &l) {
		if (first_within_tolerance_s < 0.f && std::fabs(l.hoverThrustEstimate() - 0.5f) <= 0.02f) {
			first_within_tolerance_s = l.time_s();
		}
	});

	const float final_estimate = loop.hoverThrustEstimate();
	std::printf("hover thrust: final estimate = %.4f (true 0.5, initial parameter 0.3), within 0.02 from %.2f s, "
		    "final altitude %.2f m\n",
		    final_estimate, first_within_tolerance_s, -loop.body().position()(2));
	EXPECT_NEAR(final_estimate, 0.5f, 0.02f);
	EXPECT_GE(first_within_tolerance_s, 0.f);
}
