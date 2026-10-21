/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 * Copyright 2015-2021 PX4 Development Team (gazebo_mavlink_interface.cpp input mapping)
 * Copyright 2026 Zhaohong Liu (modifications: port to batched host/device C++ for px4-rl)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file was modified from the original: see THIRD_PARTY_NOTICES.md.
 */
// Port of the Gazebo Classic motor model that PX4 v1.16.2 SITL flies Iris with:
//   PX4-SITL_gazebo-classic @ 70683dc, reference/gazebo_classic_iris/
//     gazebo_motor_model.cpp   (GazeboMotorModel::UpdateForcesAndMoments, VelocityCallback)
//     common.h                 (FirstOrderFilter)
//     gazebo_mavlink_interface.cpp (handle_actuator_controls)
//
// One rotor per call, all in the body FLU frame. Numerics are float; Gazebo uses double, so results
// differ from the plugin at the ~1e-7 relative level.
//
// What is not ported: motor-failure injection (UpdateMotorFail), wind (taken as 0, so the caller
// passes velocity relative to the air), the joint-velocity round trip and rotorVelocitySlowdownSim
// (the plugin sets joint velocity = dir * w_filtered / slowdown and reads it back times slowdown, so
// the slowdown cancels exactly and the real rotor speed is dir * w_filtered), and the disabled PID.
#pragma once

#include <cmath>

#include <matrix/matrix/math.hpp>

#include "px4rl/plant.h"

namespace px4rl {

/// State of one rotor: the FirstOrderFilter's previous output, i.e. the commanded-speed filter
/// state in rad/s, before the turning direction is applied (common.h:137-140, previousState_).
struct RotorState {
	float filtered_speed{0.f};
};

/// Force and moment one rotor applies to the vehicle, body FLU.
struct RotorOutput {
	matrix::Vector3f force{0.f, 0.f, 0.f};  ///< applied at the rotor position
	matrix::Vector3f moment{0.f, 0.f, 0.f}; ///< pure torque: drag torque + rolling moment; excludes r x force
};

/// Net force and torque about the body origin (centre of mass), body FLU.
struct Wrench {
	matrix::Vector3f force{0.f, 0.f, 0.f};
	matrix::Vector3f torque{0.f, 0.f, 0.f};
};

/// PX4 SITL scaling of an integer PWM output to the actuator value u that goes to Gazebo:
/// (pwm - PWM_SIM_PWM_MIN_MAGIC) / (PWM_SIM_PWM_MAX_MAGIC - PWM_SIM_PWM_MIN_MAGIC), with the magics
/// 1000 and 2000 (reference/PX4-Autopilot/src/modules/simulation/pwm_out_sim/PWMSim.cpp:81 and
/// PWMSim.hpp:81-82). Feed it rate_step's actuator_out.
PX4RL_HD inline float pwmToActuatorInput(float pwm)
{
	return (pwm - 1000.f) / (2000.f - 1000.f);
}

/// Commanded rotor speed in rad/s from the actuator value u.
PX4RL_HD inline float rotorReferenceSpeed(const px4rl_rotor_params &p, float u, bool armed)
{
	float reference;

	// gazebo_mavlink_interface.cpp:1190-1194
	if (armed) {
		reference = (u + p.input_offset) * p.input_scaling + p.zero_position_armed;

	} else {
		reference = p.zero_position_disarmed;
	}

	// gazebo_motor_model.cpp:189 (applies to both branches, no lower clamp)
	return fminf(reference, p.max_rot_velocity);
}

/// Velocity of the rotor relative to the air in body FLU: body velocity plus omega x r, wind 0.
/// The plugin uses the rotor link's own velocity (link_->WorldLinearVel(), gazebo_motor_model.cpp:212),
/// which includes the omega x r term. Callers that want to neglect it can pass the body velocity.
PX4RL_HD inline matrix::Vector3f rotorAirVelocity(const px4rl_rotor_params &p, const matrix::Vector3f &body_velocity,
						  const matrix::Vector3f &body_angular_velocity)
{
	const matrix::Vector3f position(p.position[0], p.position[1], p.position[2]);
	return body_velocity + body_angular_velocity.cross(position);
}

/// Forces and moments from the current rotor speed state (does not advance the state).
/// v_air: rotor velocity relative to the air, body FLU (wind 0).
PX4RL_HD inline RotorOutput rotorForcesAndMoments(const px4rl_rotor_params &p, const RotorState &state,
						  const matrix::Vector3f &v_air)
{
	const matrix::Vector3f axis(p.axis[0], p.axis[1], p.axis[2]);
	const float direction = static_cast<float>(p.turning_direction);

	// gazebo_motor_model.cpp:197,201: joint velocity is dir * filtered speed (set at the end of the
	// previous update, line 278); the slowdown factor cancels.
	const float real_motor_velocity = direction * state.filtered_speed;

	// gazebo_motor_model.cpp:202-206
	float force = real_motor_velocity * fabsf(real_motor_velocity) * p.motor_constant;

	if (!p.reversible) {
		force = fabsf(force);
	}

	// gazebo_motor_model.cpp:219-224: thrust falls off linearly with airspeed along the rotor axis
	const float v_along_axis = v_air.dot(axis);
	float scalar = 1.f - fabsf(v_along_axis) / p.max_relative_airspeed;
	scalar = fminf(fmaxf(scalar, 0.f), 1.f);

	// gazebo_motor_model.cpp:230-233: rotor drag on the velocity perpendicular to the axis
	const matrix::Vector3f v_perpendicular = v_air - axis * v_along_axis;
	const matrix::Vector3f air_drag = v_perpendicular * (-fabsf(real_motor_velocity) * p.rotor_drag_coefficient);

	// gazebo_motor_model.cpp:243-246: reaction torque about the rotor axis, from the unscaled force
	const matrix::Vector3f drag_torque = axis * (-direction * force * p.moment_constant);

	// gazebo_motor_model.cpp:248-251: rolling moment
	const matrix::Vector3f rolling_moment =
		v_perpendicular * (-fabsf(real_motor_velocity) * direction * p.rolling_moment_coefficient);

	RotorOutput out;
	out.force = axis * (force * scalar) + air_drag;
	out.moment = drag_torque + rolling_moment;
	return out;
}

/// Advances the speed state toward the reference (common.h:142-161, ZOH first-order filter with
/// tau_up while the input is above the state, tau_down otherwise). dt > 0.
PX4RL_HD inline void rotorAdvanceState(const px4rl_rotor_params &p, RotorState &state, float reference_speed, float dt)
{
	const float tau = (reference_speed > state.filtered_speed) ? p.time_constant_up : p.time_constant_down;
	const float alpha = expf(-dt / tau);
	state.filtered_speed = alpha * state.filtered_speed + (1.f - alpha) * reference_speed;
}

/// One simulation step, in the order of UpdateForcesAndMoments: forces from the current state
/// (gazebo_motor_model.cpp:197-251), then the filter update (lines 253-254, 278).
PX4RL_HD inline RotorOutput rotorStep(const px4rl_rotor_params &p, RotorState &state, float u, bool armed,
				      const matrix::Vector3f &v_air, float dt)
{
	const RotorOutput out = rotorForcesAndMoments(p, state, v_air);
	rotorAdvanceState(p, state, rotorReferenceSpeed(p, u, armed), dt);
	return out;
}

/// Adds one rotor's output to the wrench about the body origin: force, plus moment + r x force.
PX4RL_HD inline void accumulateWrench(const px4rl_rotor_params &p, const RotorOutput &out, Wrench &wrench)
{
	const matrix::Vector3f position(p.position[0], p.position[1], p.position[2]);
	wrench.force += out.force;
	wrench.torque += out.moment + position.cross(out.force);
}

/// Steps every rotor of a plant and returns the summed wrench about the body origin.
/// u_by_px4_output: actuator values indexed by PX4 output; rotor i reads
/// u_by_px4_output[plant.rotor[i].px4_output_index]. The air velocity of each rotor is
/// rotorAirVelocity() (body velocity plus omega x r). per_rotor, if not null, receives each rotor's
/// RotorOutput (num_rotors entries, indexed by plant rotor).
PX4RL_HD inline Wrench plantStep(const px4rl_plant_params &plant, RotorState *states, const float *u_by_px4_output,
				 bool armed, const matrix::Vector3f &body_velocity,
				 const matrix::Vector3f &body_angular_velocity, float dt, RotorOutput *per_rotor = nullptr)
{
	Wrench wrench;

	for (int i = 0; i < plant.num_rotors; ++i) {
		const px4rl_rotor_params &p = plant.rotor[i];
		const matrix::Vector3f v_air = rotorAirVelocity(p, body_velocity, body_angular_velocity);
		const RotorOutput out = rotorStep(p, states[i], u_by_px4_output[p.px4_output_index], armed, v_air, dt);
		accumulateWrench(p, out, wrench);

		if (per_rotor != nullptr) {
			per_rotor[i] = out;
		}
	}

	return wrench;
}

} // namespace px4rl
