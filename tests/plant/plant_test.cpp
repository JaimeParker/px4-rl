// Rotor model, Iris plant data and frame conversions (CPU).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "px4rl/airframes.h"
#include "px4rl/frames.hpp"
#include "px4rl/gazebo_motor_model.hpp"

namespace {

constexpr float kG = 9.80665f; // PX4 CONSTANTS_ONE_G
constexpr float kPi = 3.14159265358979f;

px4rl_rotor_params irisRotor(int i)
{
	return px4rl_plant_params_iris().rotor[i];
}

const matrix::Vector3f kZero(0.f, 0.f, 0.f);

// Steps one rotor at constant input until the speed has converged.
px4rl::RotorState settle(const px4rl_rotor_params &p, float u, bool armed = true, float seconds = 2.f)
{
	px4rl::RotorState state;
	const float dt = 0.001f;

	for (int k = 0; k < static_cast<int>(seconds / dt); ++k) {
		px4rl::rotorStep(p, state, u, armed, kZero, dt);
	}

	return state;
}

} // namespace

TEST(RotorModel, ReferenceSpeedMapping)
{
	const px4rl_rotor_params p = irisRotor(0);
	// gazebo_mavlink_interface.cpp:1190-1194: (u + 0) * 1000 + 100 armed, 0 disarmed
	EXPECT_FLOAT_EQ(px4rl::rotorReferenceSpeed(p, 0.f, true), 100.f);
	EXPECT_FLOAT_EQ(px4rl::rotorReferenceSpeed(p, 0.5f, true), 600.f);
	EXPECT_FLOAT_EQ(px4rl::rotorReferenceSpeed(p, 0.5f, false), 0.f);
	// gazebo_motor_model.cpp:189: capped at maxRotVelocity in both branches
	EXPECT_FLOAT_EQ(px4rl::rotorReferenceSpeed(p, 1.f, true), 1100.f);
	EXPECT_FLOAT_EQ(px4rl::rotorReferenceSpeed(p, 3.f, true), 1100.f);
	EXPECT_FLOAT_EQ(px4rl::pwmToActuatorInput(1000.f), 0.f);
	EXPECT_FLOAT_EQ(px4rl::pwmToActuatorInput(1500.f), 0.5f);
	EXPECT_FLOAT_EQ(px4rl::pwmToActuatorInput(2000.f), 1.f);
}

TEST(RotorModel, StepResponseMatchesAnalyticZohFilter)
{
	const px4rl_rotor_params p = irisRotor(0);
	const float dt = 0.002f;
	const double reference = 600.0; // u = 0.5
	px4rl::RotorState state;

	for (int n = 1; n <= 200; ++n) {
		px4rl::rotorStep(p, state, 0.5f, true, kZero, dt);
		const double t = n * static_cast<double>(dt);
		const double expected = reference * (1.0 - std::exp(-t / p.time_constant_up));
		EXPECT_NEAR(state.filtered_speed, expected, 1e-4 * reference) << "n=" << n;
	}

	// falling input uses time_constant_down: u = 0.2 -> reference 300
	const double start = state.filtered_speed;

	for (int n = 1; n <= 200; ++n) {
		px4rl::rotorStep(p, state, 0.2f, true, kZero, dt);
		const double t = n * static_cast<double>(dt);
		const double expected = 300.0 + (start - 300.0) * std::exp(-t / p.time_constant_down);
		EXPECT_NEAR(state.filtered_speed, expected, 1e-4 * start) << "n=" << n;
	}
}

TEST(RotorModel, ForcesUseTheStateBeforeTheFilterUpdate)
{
	const px4rl_rotor_params p = irisRotor(0);
	px4rl::RotorState state; // speed 0
	const px4rl::RotorOutput first = px4rl::rotorStep(p, state, 0.5f, true, kZero, 0.01f);
	EXPECT_FLOAT_EQ(first.force(2), 0.f); // computed from the initial zero speed
	EXPECT_GT(state.filtered_speed, 0.f);
	const float speed = state.filtered_speed;
	const px4rl::RotorOutput second = px4rl::rotorStep(p, state, 0.5f, true, kZero, 0.01f);
	EXPECT_NEAR(second.force(2), p.motor_constant * speed * speed, 1e-6f);
}

TEST(RotorModel, SteadyThrustMatchesMotorConstantTimesSpeedSquared)
{
	const px4rl_rotor_params p = irisRotor(2); // cw rotor: thrust must still be positive
	const float us[] = {0.1f, 0.3f, 0.694f, 0.9f};

	for (const float u : us) {
		const px4rl::RotorState state = settle(p, u);
		const px4rl::RotorOutput out = px4rl::rotorForcesAndMoments(p, state, kZero);
		const float omega = u * 1000.f + 100.f;
		EXPECT_NEAR(out.force(2), p.motor_constant * omega * omega, 2e-4f * p.motor_constant * omega * omega) << "u=" << u;
		EXPECT_FLOAT_EQ(out.force(0), 0.f);
		EXPECT_FLOAT_EQ(out.force(1), 0.f);
	}

	// saturation at maxRotVelocity
	const px4rl::RotorState saturated = settle(p, 5.f);
	EXPECT_NEAR(saturated.filtered_speed, 1100.f, 0.1f);

	// disarmed: speed decays to zero
	EXPECT_NEAR(settle(p, 0.7f, false, 3.f).filtered_speed, 0.f, 1e-3f);
}

TEST(RotorModel, ReactionTorqueSignFollowsTurningDirection)
{
	// gazebo_motor_model.cpp:243: torque about the axis = -dir * F * k_m
	for (int i = 0; i < 4; ++i) {
		const px4rl_rotor_params p = irisRotor(i);
		const px4rl::RotorState state = settle(p, 0.5f);
		const px4rl::RotorOutput out = px4rl::rotorForcesAndMoments(p, state, kZero);
		const float expected = -static_cast<float>(p.turning_direction) * out.force(2) * p.moment_constant;
		EXPECT_NEAR(out.moment(2), expected, 1e-5f);
		EXPECT_EQ(out.moment(2) < 0.f, p.turning_direction > 0) << "rotor " << i;
	}
}

TEST(RotorModel, ThrustScalesDownWithAirspeedAlongTheAxis)
{
	const px4rl_rotor_params p = irisRotor(0);
	const px4rl::RotorState state = settle(p, 0.5f);
	const float thrust0 = px4rl::rotorForcesAndMoments(p, state, kZero).force(2);
	ASSERT_GT(thrust0, 0.f);
	EXPECT_NEAR(px4rl::rotorForcesAndMoments(p, state, matrix::Vector3f(0.f, 0.f, 10.f)).force(2), 0.6f * thrust0, 1e-5f);
	EXPECT_NEAR(px4rl::rotorForcesAndMoments(p, state, matrix::Vector3f(0.f, 0.f, -10.f)).force(2), 0.6f * thrust0, 1e-5f);
	EXPECT_FLOAT_EQ(px4rl::rotorForcesAndMoments(p, state, matrix::Vector3f(0.f, 0.f, 25.f)).force(2), 0.f);
	EXPECT_FLOAT_EQ(px4rl::rotorForcesAndMoments(p, state, matrix::Vector3f(0.f, 0.f, 40.f)).force(2), 0.f);
	// perpendicular airspeed does not reduce thrust
	EXPECT_NEAR(px4rl::rotorForcesAndMoments(p, state, matrix::Vector3f(10.f, 0.f, 0.f)).force(2), thrust0, 1e-5f);
}

TEST(RotorModel, DragOpposesPerpendicularAirVelocityAndRollingMomentFollowsDirection)
{
	const matrix::Vector3f v_air(3.f, -2.f, 0.f);

	for (int i = 0; i < 4; ++i) {
		const px4rl_rotor_params p = irisRotor(i);
		const px4rl::RotorState state = settle(p, 0.5f);
		const float omega = state.filtered_speed;
		const px4rl::RotorOutput out = px4rl::rotorForcesAndMoments(p, state, v_air);
		// gazebo_motor_model.cpp:231: air_drag = -|w| c_d v_perp
		EXPECT_NEAR(out.force(0), -omega * p.rotor_drag_coefficient * 3.f, 1e-5f);
		EXPECT_NEAR(out.force(1), -omega * p.rotor_drag_coefficient * -2.f, 1e-5f);
		// gazebo_motor_model.cpp:250: rolling = -|w| dir c_r v_perp (pure moment, x/y components)
		const float d = static_cast<float>(p.turning_direction);
		EXPECT_NEAR(out.moment(0), -omega * d * p.rolling_moment_coefficient * 3.f, 1e-7f);
		EXPECT_NEAR(out.moment(1), -omega * d * p.rolling_moment_coefficient * -2.f, 1e-7f);
		EXPECT_LT(out.force(0) * 3.f + out.force(1) * -2.f, 0.f); // dissipative
	}
}

TEST(RotorModel, ReversibleRotorKeepsTheSignOfTheForce)
{
	px4rl_rotor_params p = irisRotor(0); // ccw: joint velocity has the sign of the reference
	p.input_offset = -0.5f;              // u = 0 -> reference -400 rad/s
	p.reversible = 1;
	px4rl::RotorState state = settle(p, 0.f);
	EXPECT_LT(state.filtered_speed, 0.f);
	EXPECT_LT(px4rl::rotorForcesAndMoments(p, state, kZero).force(2), 0.f);
	p.reversible = 0;
	EXPECT_GT(px4rl::rotorForcesAndMoments(p, state, kZero).force(2), 0.f);
}

TEST(IrisPlant, ChannelMappingAndSdfValues)
{
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	ASSERT_EQ(plant.num_rotors, 4);
	const int directions[4] = {+1, +1, -1, -1};
	const float x[4] = {0.13f, -0.13f, 0.13f, -0.13f};
	const float y[4] = {-0.22f, 0.2f, 0.22f, -0.2f};

	for (int i = 0; i < 4; ++i) {
		const px4rl_rotor_params &r = plant.rotor[i];
		EXPECT_EQ(r.px4_output_index, i);
		EXPECT_EQ(r.turning_direction, directions[i]);
		EXPECT_FLOAT_EQ(r.position[0], x[i]);
		EXPECT_FLOAT_EQ(r.position[1], y[i]);
		EXPECT_FLOAT_EQ(r.position[2], 0.023f);
		EXPECT_FLOAT_EQ(r.axis[2], 1.f);
		EXPECT_FLOAT_EQ(r.motor_constant, 5.84e-06f);
		EXPECT_FLOAT_EQ(r.moment_constant, 0.06f);
		EXPECT_FLOAT_EQ(r.max_rot_velocity, 1100.f);
		EXPECT_FLOAT_EQ(r.max_relative_airspeed, 25.f);
		EXPECT_FLOAT_EQ(r.input_scaling, 1000.f);
		EXPECT_FLOAT_EQ(r.zero_position_armed, 100.f);
		EXPECT_FLOAT_EQ(r.zero_position_disarmed, 0.f);
	}

	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();
	EXPECT_FLOAT_EQ(body.mass, 1.5f);
	EXPECT_FLOAT_EQ(body.inertia[0], 0.029125f);
	EXPECT_FLOAT_EQ(body.inertia[4], 0.029125f);
	EXPECT_FLOAT_EQ(body.inertia[8], 0.055225f);
	EXPECT_FLOAT_EQ(body.inertia[1], 0.f);
}

TEST(IrisPlant, HoverActuatorInput)
{
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();
	const px4rl_rotor_params &p = plant.rotor[0];

	// k_f * (1000 u + 100)^2 = m g / 4
	const double omega_hover = std::sqrt(static_cast<double>(body.mass) * kG / 4.0 / p.motor_constant);
	const double u_hover = (omega_hover - p.zero_position_armed) / p.input_scaling;
	std::printf("Iris hover: omega = %.4f rad/s, u = %.6f\n", omega_hover, u_hover);
	EXPECT_NEAR(u_hover, 0.694, 1e-3);

	// simulated: all four rotors at u_hover give a net force of m g and no torque
	px4rl::RotorState states[4];
	const float u_all[4] = {static_cast<float>(u_hover), static_cast<float>(u_hover),
				static_cast<float>(u_hover), static_cast<float>(u_hover)};
	px4rl::Wrench wrench;

	for (int k = 0; k < 4000; ++k) {
		wrench = px4rl::plantStep(plant, states, u_all, true, kZero, kZero, 0.001f);
	}

	EXPECT_NEAR(wrench.force(2), body.mass * kG, 2e-4f * body.mass * kG);
	EXPECT_NEAR(wrench.force(0), 0.f, 1e-4f);
	EXPECT_NEAR(wrench.force(1), 0.f, 1e-4f);
	EXPECT_NEAR(wrench.torque(0), 0.f, 1e-3f);
	EXPECT_NEAR(wrench.torque(1), 0.f, 1e-3f);
	EXPECT_NEAR(wrench.torque(2), 0.f, 1e-3f);
}

TEST(IrisPlant, WrenchUsesPx4OutputIndexAndLeverArms)
{
	px4rl_plant_params plant = px4rl_plant_params_iris();
	px4rl::RotorState states[4];
	// output 0 high, others low: rotor_0 is front-right (+x, -y)
	const float u[4] = {0.8f, 0.3f, 0.3f, 0.3f};
	px4rl::Wrench wrench;

	for (int k = 0; k < 3000; ++k) {
		wrench = px4rl::plantStep(plant, states, u, true, kZero, kZero, 0.001f);
	}

	EXPECT_GT(states[0].filtered_speed, states[1].filtered_speed);
	EXPECT_LT(wrench.torque(0), 0.f); // roll: y*Fz with y < 0 dominates -> negative about x
	EXPECT_LT(wrench.torque(1), 0.f); // pitch: -x*Fz with x > 0 dominates -> negative about y
	EXPECT_LT(wrench.torque(2), 0.f); // yaw: ccw rotor sped up -> reaction torque negative about z

	// reroute: rotor 1 reads output 0 instead
	plant.rotor[1].px4_output_index = 0;
	px4rl::RotorState states2[4];

	for (int k = 0; k < 3000; ++k) {
		px4rl::plantStep(plant, states2, u, true, kZero, kZero, 0.001f);
	}

	EXPECT_NEAR(states2[1].filtered_speed, states2[0].filtered_speed, 1e-3f);
}

TEST(Frames, VectorMapsAndFixedCases)
{
	using namespace px4rl;
	const matrix::Vector3f ned = enuToNed(matrix::Vector3f(1.f, 2.f, 3.f)); // east 1, north 2, up 3
	EXPECT_FLOAT_EQ(ned(0), 2.f);  // north
	EXPECT_FLOAT_EQ(ned(1), 1.f);  // east
	EXPECT_FLOAT_EQ(ned(2), -3.f); // down
	const matrix::Vector3f back = nedToEnu(ned);
	EXPECT_FLOAT_EQ(back(0), 1.f);
	EXPECT_FLOAT_EQ(back(1), 2.f);
	EXPECT_FLOAT_EQ(back(2), 3.f);

	// FLU body z (up) is FRD -z (down); left is FRD -y
	const matrix::Vector3f up = fluToFrd(matrix::Vector3f(0.f, 0.f, 1.f));
	EXPECT_FLOAT_EQ(up(2), -1.f);
	const matrix::Vector3f left = fluToFrd(matrix::Vector3f(0.f, 1.f, 0.f));
	EXPECT_FLOAT_EQ(left(1), -1.f);
	const matrix::Vector3f round = frdToFlu(fluToFrd(matrix::Vector3f(1.f, 2.f, 3.f)));
	EXPECT_FLOAT_EQ(round(1), 2.f);
	EXPECT_FLOAT_EQ(round(2), 3.f);

	// gravity: ENU (0, 0, -g) is NED (0, 0, +g)
	EXPECT_FLOAT_EQ(enuToNed(matrix::Vector3f(0.f, 0.f, -kG))(2), kG);
}

TEST(Frames, HeadingFromEnuQuaternion)
{
	using namespace px4rl;
	// ENU yaw psi: rotation about z up. ENU yaw 0 faces east (NED heading +90 deg).
	const float yaws_enu[] = {0.f, kPi / 2.f, kPi, -kPi / 2.f, 0.3f};
	const float expected_ned[] = {kPi / 2.f, 0.f, -kPi / 2.f, kPi, kPi / 2.f - 0.3f};

	for (int i = 0; i < 5; ++i) {
		const matrix::Quatf q(std::cos(yaws_enu[i] / 2.f), 0.f, 0.f, std::sin(yaws_enu[i] / 2.f));
		const float heading = headingNedFromEnuFlu(q);
		const float diff = std::atan2(std::sin(heading - expected_ned[i]), std::cos(heading - expected_ned[i]));
		EXPECT_NEAR(diff, 0.f, 1e-5f) << "yaw_enu=" << yaws_enu[i];
	}

	// the converted attitude must agree with PX4's own euler yaw
	const matrix::Quatf q(std::cos(0.2f), 0.1f, -0.2f, std::sin(0.2f));
	const matrix::Quatf q_n = matrix::Quatf(q).normalized();
	EXPECT_NEAR(headingNedFromEnuFlu(q_n), matrix::Eulerf(quatWorldBodyEnuFluToNedFrd(q_n)).psi(), 1e-5f);
}

namespace {

struct Lcg {
	uint32_t state;
	float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}
	float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

} // namespace

TEST(Frames, QuaternionConversionsAreConsistentAndRoundTrip)
{
	using namespace px4rl;
	Lcg rng{12345u};

	for (int n = 0; n < 200; ++n) {
		matrix::Quatf q(rng.range(-1.f, 1.f), rng.range(-1.f, 1.f), rng.range(-1.f, 1.f), rng.range(-1.f, 1.f));
		q.normalize();
		const matrix::Vector3f v(rng.range(-5.f, 5.f), rng.range(-5.f, 5.f), rng.range(-5.f, 5.f));

		// q_enu_flu (FLU -> ENU) and its converted q_ned_frd must describe the same physical rotation
		const matrix::Quatf q_ned_frd = quatWorldBodyEnuFluToNedFrd(q);
		const matrix::Vector3f lhs = q_ned_frd.rotateVector(fluToFrd(v));
		const matrix::Vector3f rhs = enuToNed(q.rotateVector(v));
		EXPECT_NEAR((lhs - rhs).norm(), 0.f, 1e-5f);

		// body_world direction: q_flu_enu = q_enu_flu^-1
		const matrix::Quatf q_flu_enu = q.inversed();
		const matrix::Quatf q_frd_ned = quatBodyWorldEnuFluToNedFrd(q_flu_enu);
		const matrix::Vector3f lhs2 = q_frd_ned.rotateVector(enuToNed(v));
		const matrix::Vector3f rhs2 = fluToFrd(q_flu_enu.rotateVector(v));
		EXPECT_NEAR((lhs2 - rhs2).norm(), 0.f, 1e-5f);
		// and it is the inverse of q_ned_frd
		EXPECT_NEAR((q_frd_ned - q_ned_frd.inversed()).norm(), 0.f, 1e-5f);

		// round trips (a quaternion and its negative are the same rotation)
		const matrix::Quatf r1 = quatWorldBodyNedFrdToEnuFlu(q_ned_frd);
		const matrix::Quatf r2 = quatBodyWorldNedFrdToEnuFlu(q_frd_ned);
		EXPECT_NEAR(std::abs(r1.dot(q)), 1.f, 1e-5f);
		EXPECT_NEAR(std::abs(r2.dot(q_flu_enu)), 1.f, 1e-5f);
	}
}

TEST(Frames, LevelBodyFacingEastIsLevelNedFacingEast)
{
	using namespace px4rl;
	const matrix::Quatf identity_enu_flu(1.f, 0.f, 0.f, 0.f); // FLU axes = ENU axes: forward = east, up = up
	const matrix::Quatf q_ned_frd = quatWorldBodyEnuFluToNedFrd(identity_enu_flu);
	// FRD forward must point east in NED, FRD down must point down
	const matrix::Vector3f forward = q_ned_frd.rotateVector(matrix::Vector3f(1.f, 0.f, 0.f));
	EXPECT_NEAR(forward(0), 0.f, 1e-6f);
	EXPECT_NEAR(forward(1), 1.f, 1e-6f);
	EXPECT_NEAR(forward(2), 0.f, 1e-6f);
	const matrix::Vector3f down = q_ned_frd.rotateVector(matrix::Vector3f(0.f, 0.f, 1.f));
	EXPECT_NEAR(down(2), 1.f, 1e-6f);
}

// ---- hover thrust (px4rl_hover_thrust) ----

namespace {

px4rl_params irisController()
{
	px4rl_params params;
	px4rl_params_iris(&params);
	return params;
}

// The hover actuator value of the Iris plant, closed form: 4 k_f (1000 u + 100)^2 = m g.
float irisHoverActuatorValue(float mass)
{
	const px4rl_rotor_params rotor = irisRotor(0);
	const float speed = std::sqrt(mass * kG / (4.f * rotor.motor_constant));
	return (speed - rotor.zero_position_armed) / rotor.input_scaling;
}

} // namespace

TEST(HoverThrust, IrisHoversAtTheDocumentedActuatorValue)
{
	const px4rl_params params = irisController();
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();

	const float thrust = px4rl_hover_thrust(&params, &plant, &body, kG);
	std::printf("iris hover thrust = %.5f (THR_MDL_FAC = %.2f)\n", thrust, params.thr_mdl_fac);
	EXPECT_NEAR(thrust, 0.6935f, 5e-4f);
	// THR_MDL_FAC is 0 for the Iris, so the collective thrust equals the actuator value.
	EXPECT_NEAR(thrust, irisHoverActuatorValue(body.mass), 1e-5f);
}

TEST(HoverThrust, ThrustModelFactorMapsTheActuatorValueThroughTheForwardCurve)
{
	px4rl_params params = irisController();
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();

	const float actuator_value = px4rl_hover_thrust(&params, &plant, &body, kG);
	params.thr_mdl_fac = 0.3f;
	EXPECT_NEAR(px4rl_hover_thrust(&params, &plant, &body, kG),
		    0.3f * actuator_value * actuator_value + 0.7f * actuator_value, 1e-6f);
}

TEST(HoverThrust, HeavierVehicleNeedsMoreThrustAndOneBeyondTheRotorsCannotHover)
{
	const px4rl_params params = irisController();
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();
	const float iris_thrust = px4rl_hover_thrust(&params, &plant, &body, kG);

	body.mass = 1.8f;
	EXPECT_NEAR(px4rl_hover_thrust(&params, &plant, &body, kG), irisHoverActuatorValue(1.8f), 1e-5f);
	EXPECT_GT(px4rl_hover_thrust(&params, &plant, &body, kG), iris_thrust);

	// Full thrust is 4 k_f max_rot_velocity^2 = 28.3 N, below the weight of 3 kg (29.4 N).
	body.mass = 3.f;
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &plant, &body, kG)));
}

TEST(HoverThrust, InvalidInputsGiveNaN)
{
	const px4rl_params params = irisController();
	const px4rl_plant_params plant = px4rl_plant_params_iris();
	const px4rl_rigid_body_params body = px4rl_rigid_body_params_iris();

	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(nullptr, &plant, &body, kG)));
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, nullptr, &body, kG)));
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &plant, nullptr, kG)));
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &plant, &body, 0.f)));
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &plant, &body, -kG)));

	px4rl_rigid_body_params massless = body;
	massless.mass = 0.f;
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &plant, &massless, kG)));

	px4rl_params bad_curve = params;
	bad_curve.thr_mdl_fac = 1.5f;
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&bad_curve, &plant, &body, kG)));

	px4rl_plant_params no_rotors = plant;
	no_rotors.num_rotors = 0;
	EXPECT_TRUE(std::isnan(px4rl_hover_thrust(&params, &no_rotors, &body, kG)));
}
