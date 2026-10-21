// Per-vehicle simulation state of the Gazebo rotor plant: the plant parameters and the speed state of
// each rotor. It is stored beside Vehicle, not inside it, so resetting the controllers does not touch the
// plant and the rotor step does not touch the controller state. CpuFleet and the CUDA kernels both hold
// one VehiclePlant per vehicle and call VehiclePlant::step through rotorStepAt (step_io.hpp).
#pragma once

#include <cmath>

#include "px4rl/gazebo_motor_model.hpp"
#include "px4rl/plant.h"

namespace px4rl {

struct VehiclePlant {
	px4rl_plant_params params{};
	RotorState rotor[PX4RL_MAX_ROTORS]{};

	/// Takes the given plant and puts every rotor at rest.
	PX4RL_HD void configure(const px4rl_plant_params &plant)
	{
		params = plant;

		for (int i = 0; i < PX4RL_MAX_ROTORS; ++i) {
			rotor[i] = RotorState{};
		}
	}

	/// PWM scaling, checked against PX4 v1.16.2. PX4 SITL hands Gazebo the output of the simulation PWM driver,
	/// which forces its output range to 1000..2000 (PWMSim.cpp:49-50 with PWM_SIM_PWM_MIN_MAGIC = 1000 and
	/// PWM_SIM_PWM_MAX_MAGIC = 2000, PWMSim.hpp:81-82) and scales a non-reversible motor output to
	/// (pwm - 1000) / (2000 - 1000) (PWMSim.cpp:78-81). px4-rl's ActuatorOutput defaults to the same range
	/// (pwm_min 1000, pwm_max 2000: px4/actuator_output/ActuatorOutputTypes.hpp:44,47), so its PWM maps back to
	/// the thrust command it came from. The driver passes a disarmed output (900, PWMSim.cpp:72) through
	/// unscaled, so a disarmed vehicle must be stepped with armed = false.
	///
	/// One plant step. `pwm` holds PX4RL_MAX_ROTORS values indexed by PX4 output, as produced by
	/// Vehicle::rateStep. They are scaled to the actuator value u with pwmToActuatorInput(), then every rotor
	/// runs plantStep(). `per_rotor`, if not null, receives the output of each plant rotor.
	/// Returns the wrench about the body origin, body FLU.
	PX4RL_HD Wrench step(const float *pwm, bool armed, const matrix::Vector3f &body_velocity,
			     const matrix::Vector3f &body_angular_velocity, float dt, RotorOutput *per_rotor)
	{
		float u_by_px4_output[PX4RL_MAX_ROTORS];

		for (int k = 0; k < PX4RL_MAX_ROTORS; ++k) {
			u_by_px4_output[k] = pwmToActuatorInput(pwm[k]);
		}

		return plantStep(params, rotor, u_by_px4_output, armed, body_velocity, body_angular_velocity, dt, per_rotor);
	}
};

/// Returns nullptr when `plant` can be simulated, otherwise a static message. Host only.
inline const char *validatePlantParams(const px4rl_plant_params &plant)
{
	if (plant.num_rotors < 1 || plant.num_rotors > PX4RL_MAX_ROTORS) {
		return "plant num_rotors must be in [1, PX4RL_MAX_ROTORS]";
	}

	for (int i = 0; i < plant.num_rotors; ++i) {
		const px4rl_rotor_params &r = plant.rotor[i];

		// Each rotor reads one entry of the n x PX4RL_MAX_ROTORS actuator row.
		if (r.px4_output_index < 0 || r.px4_output_index >= PX4RL_MAX_ROTORS) {
			return "plant px4_output_index must be in [0, PX4RL_MAX_ROTORS)";
		}

		if (!(r.time_constant_up > 0.f) || !(r.time_constant_down > 0.f)) {
			return "plant time constants must be > 0";
		}

		if (!(r.max_relative_airspeed > 0.f)) {
			return "plant max_relative_airspeed must be > 0";
		}

		for (int k = 0; k < 3; ++k) {
			if (!std::isfinite(r.position[k]) || !std::isfinite(r.axis[k])) {
				return "plant rotor position and axis must be finite";
			}
		}

		if (r.turning_direction != 1 && r.turning_direction != -1) {
			return "plant turning_direction must be +1 or -1";
		}
	}

	return nullptr;
}

} // namespace px4rl
