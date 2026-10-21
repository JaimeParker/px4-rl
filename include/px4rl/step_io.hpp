// Row-major SoA buffer glue between the C API and Vehicle. Shared by CpuFleet (loop over i)
// and the CUDA kernels (one thread per i), so there is a single copy of the load/store code.
#pragma once

#include <cstdint>

#include "px4rl/vehicle.hpp"
#include "px4rl/vehicle_plant.hpp"

namespace px4rl {

static_assert(static_cast<uint8_t>(PositionModule::Outcome::Ok) == PX4RL_OUTCOME_OK &&
	      static_cast<uint8_t>(PositionModule::Outcome::Fallback) == PX4RL_OUTCOME_FALLBACK &&
	      static_cast<uint8_t>(PositionModule::Outcome::Failsafe) == PX4RL_OUTCOME_FAILSAFE,
	      "PX4RL_OUTCOME_* must match PositionModule::Outcome");

struct PositionStepBuffers {
	const float *position;         // n,3
	const float *velocity;         // n,3
	const float *heading;          // n
	const float *acceleration_z;   // n
	const float *trajectory_sp;    // n,11: position[3] velocity[3] acceleration[3] yaw yawspeed
	const uint8_t *in_air;         // n or nullptr
	float dt;
	uint64_t time_us;
	float *q_sp_out;               // n,4
	float *thrust_sp_body_out;     // n,3
	float *yawspeed_out;           // n
	float *hover_thrust_out;       // n or nullptr
	uint8_t *outcome_out;          // n or nullptr
};

struct AttitudeStepBuffers {
	const float *q;               // n,4
	const float *q_sp;            // n,4 or nullptr (= setpoint of the last position step)
	const float *yawspeed_ff;     // n or nullptr; ignored when q_sp is nullptr
	const float *thrust_sp_body;  // n,3
	float *rates_sp_out;          // n,3
	float *thrust_body_out;       // n,3
};

struct RateStepBuffers {
	const float *rates_sp;        // n,3
	const float *thrust_body;     // n,3
	const float *rates;           // n,3
	const float *angular_accel;   // n,3
	float dt;
	const uint8_t *landed;        // n or nullptr
	float *motors_out;            // n,4
	float *actuator_out;          // n,4 or nullptr
	float *torque_sp_out;         // n,3 or nullptr
};

struct RotorStepBuffers {
	const float *actuator;         // n,PX4RL_MAX_ROTORS: PWM by PX4 output
	const uint8_t *armed;          // n or nullptr (= all armed)
	const float *body_velocity;    // n,3 body FLU, relative to the air
	const float *body_rates;       // n,3 body FLU, or nullptr (= no omega x r term)
	float dt;
	float *force_out;              // n,3
	float *torque_out;             // n,3
	float *rotor_force_out;        // n,PX4RL_MAX_ROTORS,3 or nullptr
};

PX4RL_HD inline void positionStepAt(Vehicle &vehicle, int i, const PositionStepBuffers &b)
{
	const float *sp = b.trajectory_sp + 11 * i;
	trajectory_setpoint_s setpoint{b.time_us, {sp[0], sp[1], sp[2]}, {sp[3], sp[4], sp[5]}, {sp[6], sp[7], sp[8]},
				       {NAN, NAN, NAN}, sp[9], sp[10]};

	PositionStepResult result;
	vehicle.positionStep(detail::load(b.position + 3 * i), detail::load(b.velocity + 3 * i), b.heading[i],
			     b.acceleration_z[i], setpoint, b.in_air == nullptr || b.in_air[i] != 0, b.dt, b.time_us,
			     result);

	for (int k = 0; k < 4; ++k) {
		b.q_sp_out[4 * i + k] = result.q_sp(k);
	}

	detail::store(b.thrust_sp_body_out + 3 * i, result.thrust_sp_body);
	b.yawspeed_out[i] = result.yawspeed;

	if (b.hover_thrust_out != nullptr) {
		b.hover_thrust_out[i] = result.hover_thrust;
	}

	if (b.outcome_out != nullptr) {
		b.outcome_out[i] = static_cast<uint8_t>(result.outcome);
	}
}

PX4RL_HD inline void attitudeStepAt(Vehicle &vehicle, int i, const AttitudeStepBuffers &b)
{
	const float *q = b.q + 4 * i;
	const matrix::Quatf attitude(q[0], q[1], q[2], q[3]);
	const matrix::Vector3f thrust_sp_body = detail::load(b.thrust_sp_body + 3 * i);

	matrix::Vector3f rates_sp;
	matrix::Vector3f thrust_body;

	if (b.q_sp == nullptr) {
		vehicle.attitudeStepFromPosition(attitude, thrust_sp_body, rates_sp, thrust_body);

	} else {
		const float *q_sp = b.q_sp + 4 * i;
		const float yawspeed_ff = b.yawspeed_ff != nullptr ? b.yawspeed_ff[i] : 0.f;
		vehicle.attitudeStep(attitude, matrix::Quatf(q_sp[0], q_sp[1], q_sp[2], q_sp[3]), yawspeed_ff,
				     thrust_sp_body, rates_sp, thrust_body);
	}

	detail::store(b.rates_sp_out + 3 * i, rates_sp);
	detail::store(b.thrust_body_out + 3 * i, thrust_body);
}

PX4RL_HD inline void rateStepAt(Vehicle &vehicle, int i, const RateStepBuffers &b)
{
	RateStepResult result;
	vehicle.rateStep(detail::load(b.rates_sp + 3 * i), detail::load(b.thrust_body + 3 * i),
			 detail::load(b.rates + 3 * i), detail::load(b.angular_accel + 3 * i), b.dt,
			 b.landed != nullptr && b.landed[i] != 0, result);

	for (int k = 0; k < 4; ++k) {
		b.motors_out[4 * i + k] = result.motors[k];
	}

	if (b.actuator_out != nullptr) {
		for (int k = 0; k < 4; ++k) {
			b.actuator_out[4 * i + k] = result.actuator[k];
		}
	}

	if (b.torque_sp_out != nullptr) {
		detail::store(b.torque_sp_out + 3 * i, result.torque_sp);
	}
}

PX4RL_HD inline void rotorStepAt(VehiclePlant &plant, int i, const RotorStepBuffers &b)
{
	const matrix::Vector3f body_rates =
		b.body_rates != nullptr ? detail::load(b.body_rates + 3 * i) : matrix::Vector3f(0.f, 0.f, 0.f);
	RotorOutput per_rotor[PX4RL_MAX_ROTORS];

	const Wrench wrench = plant.step(b.actuator + PX4RL_MAX_ROTORS * i, b.armed == nullptr || b.armed[i] != 0,
					 detail::load(b.body_velocity + 3 * i), body_rates, b.dt,
					 b.rotor_force_out != nullptr ? per_rotor : nullptr);

	detail::store(b.force_out + 3 * i, wrench.force);
	detail::store(b.torque_out + 3 * i, wrench.torque);

	if (b.rotor_force_out != nullptr) {
		for (int k = 0; k < PX4RL_MAX_ROTORS; ++k) {
			detail::store(b.rotor_force_out + 3 * (PX4RL_MAX_ROTORS * i + k),
				      k < plant.params.num_rotors ? per_rotor[k].force : matrix::Vector3f(0.f, 0.f, 0.f));
		}
	}
}

} // namespace px4rl
