// Steady-hover thrust of a vehicle: the controller / plant / rigid-body combination in, PX4's normalized
// collective thrust (the meaning of MPC_THR_HOVER) out. Host-only; shared by the CPU library and
// libpx4rl_cuda.so, which wrap it as px4rl_hover_thrust (airframes.h).
#pragma once

#include <cmath>

#include <ThrustCurve.hpp>

#include "px4rl/c_api.h"
#include "px4rl/plant.h"

namespace px4rl {

namespace detail {

inline const char *validateHoverInputs(const px4rl_params &params, const px4rl_plant_params &plant,
				       const px4rl_rigid_body_params &body, float gravity)
{
	if (!(body.mass > 0.f) || !std::isfinite(body.mass)) {
		return "px4rl_hover_thrust: mass must be finite and > 0";
	}

	if (!(gravity > 0.f) || !std::isfinite(gravity)) {
		return "px4rl_hover_thrust: gravity must be finite and > 0";
	}

	if (!(params.thr_mdl_fac >= 0.f && params.thr_mdl_fac <= 1.f)) {
		return "px4rl_hover_thrust: thr_mdl_fac must be in [0, 1]";
	}

	if (plant.num_rotors < 1 || plant.num_rotors > PX4RL_MAX_ROTORS) {
		return "px4rl_hover_thrust: num_rotors must be in [1, PX4RL_MAX_ROTORS]";
	}

	for (int i = 0; i < plant.num_rotors; ++i) {
		const px4rl_rotor_params &r = plant.rotor[i];

		if (!std::isfinite(r.axis[2]) || !std::isfinite(r.motor_constant) || !std::isfinite(r.max_rot_velocity) ||
		    !std::isfinite(r.input_offset) || !std::isfinite(r.input_scaling) || !std::isfinite(r.zero_position_armed)) {
			return "px4rl_hover_thrust: rotor parameters must be finite";
		}
	}

	return nullptr;
}

/// Net thrust of the plant along the upward vertical, N, when every rotor is commanded the actuator value u
/// and has settled at its reference speed.
inline double hoverVerticalThrust(const px4rl_plant_params &plant, double u)
{
	double vertical = 0.;

	for (int i = 0; i < plant.num_rotors; ++i) {
		const px4rl_rotor_params &r = plant.rotor[i];

		// gazebo_mavlink_interface.cpp:1190-1194 (armed branch), then the clamp of gazebo_motor_model.cpp:189.
		// The model's first-order filter (common.h:142-161) has this value as its fixed point.
		double speed = (u + r.input_offset) * r.input_scaling + r.zero_position_armed;
		speed = std::fmin(speed, static_cast<double>(r.max_rot_velocity));

		// gazebo_motor_model.cpp:202-206: force = w |w| k_f along the axis, |force| unless <reversible>.
		// The turning direction cancels (it enters twice). Zero airspeed, so the airspeed factor of lines
		// 219-224 is 1 and there is no rotor drag (lines 230-233).
		double force = speed * std::fabs(speed) * r.motor_constant;

		if (!r.reversible) {
			force = std::fabs(force);
		}

		// Upright: the body FLU z axis is the world up, so the vertical share of a rotor's force is axis[2].
		// Horizontal components (non-vertical axes) are not balanced here.
		vertical += force * r.axis[2];
	}

	return vertical;
}

} // namespace detail

/// PX4 normalized collective thrust (MPC_THR_HOVER units) at which the vehicle hovers, or NaN if it cannot.
/// On NaN, `error` (if given) is set to a static message.
///
/// Model, steady upright hover, armed, zero airspeed, every rotor at its reference speed:
///  1. One actuator value u in [0, 1] drives all rotors (a symmetric hover). Rotor i settles at
///     w_i = min((u + input_offset_i) * input_scaling_i + zero_position_armed_i, max_rot_velocity_i)
///     (gazebo_mavlink_interface.cpp:1190-1194, gazebo_motor_model.cpp:189) and produces
///     k_f,i * w_i^2 along its axis (gazebo_motor_model.cpp:202-206).
///  2. u solves sum_i axis_i,z * thrust_i(u) = mass * gravity. Each rotor's thrust is nondecreasing in u when
///     input_scaling_i >= 0 (w_i is then nondecreasing in u, and w^2 is nondecreasing for w >= 0), so with
///     k_f >= 0 and axis z >= 0 the sum is nondecreasing and a bisection on [0, 1] converges to the root.
///     For other plants the sum is still continuous, so a bracket (f(0) <= 0 <= f(1)) guarantees a root,
///     though not a unique one. No bracket means the vehicle cannot hover: NaN.
///  3. u is what the PWM driver sends, i.e. after PX4's thrust-curve inverse (ThrustCurve.hpp). The collective
///     thrust the position controller commands is the forward curve of it, rel = f * u^2 + (1 - f) * u with
///     f = THR_MDL_FAC (thrustCurveForward). This is MPC_THR_HOVER's meaning because the allocator maps a
///     collective thrust to equal motor setpoints for a symmetric hover.
///  4. u is continuous: the integer PWM quantization (about 1e-3 in u) is ignored, and u is taken as the
///     simulation driver's (pwm - 1000) / 1000 with PX4's 1000..2000 PWM range.
inline float hoverThrust(const px4rl_params &params, const px4rl_plant_params &plant,
			 const px4rl_rigid_body_params &body, float gravity, const char **error = nullptr)
{
	const auto fail = [&](const char *message) {
		if (error != nullptr) {
			*error = message;
		}

		return std::nanf("");
	};

	if (const char *message = detail::validateHoverInputs(params, plant, body, gravity)) {
		return fail(message);
	}

	const double weight = static_cast<double>(body.mass) * static_cast<double>(gravity);

	if (detail::hoverVerticalThrust(plant, 0.) > weight || detail::hoverVerticalThrust(plant, 1.) < weight) {
		return fail("px4rl_hover_thrust: no actuator value in [0, 1] balances the weight");
	}

	double low = 0.;
	double high = 1.;

	for (int iteration = 0; iteration < 64; ++iteration) {
		const double middle = 0.5 * (low + high);

		if (detail::hoverVerticalThrust(plant, middle) < weight) {
			low = middle;

		} else {
			high = middle;
		}
	}

	const float actuator = static_cast<float>(0.5 * (low + high));
	return px4_stack::thrustCurveForward(actuator, params.thr_mdl_fac);
}

} // namespace px4rl
