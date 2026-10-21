// Batched CPU counterpart of the CUDA fleet: same operations, host pointers, optional OpenMP.
#pragma once

#include <cstdint>
#include <vector>

#include "px4rl/c_api.h"
#include "px4rl/plant.h"
#include "px4rl/vehicle.hpp"
#include "px4rl/vehicle_plant.hpp"

namespace px4rl {

class CpuFleet {
public:
	/// Constructs n vehicles with px4rl_params_default() and gives each the Iris plant (px4rl_plant_params_iris())
	/// with its rotors at rest. Throws std::invalid_argument if n < 1.
	explicit CpuFleet(int n);

	int size() const { return static_cast<int>(vehicles_.size()); }

	/// mask: host uint8[n] or nullptr (= all). n_params must be 1 (broadcast) or n, otherwise
	/// std::invalid_argument is thrown, as it is for parameters that fail validateParams().
	void reset(const uint8_t *mask, const px4rl_params *params, int n_params);

	/// Replaces the plant of the selected vehicles and puts their rotors at rest; the controllers are untouched.
	/// Same arguments and exceptions as reset(), with n_plants in place of n_params.
	void resetPlant(const uint8_t *mask, const px4rl_plant_params *plants, int n_plants);

	/// Same arguments as px4rl_cuda_position_step, on host pointers; see c_api.h.
	void positionStep(const float *position, const float *velocity, const float *heading, const float *acceleration_z,
			  const float *trajectory_sp, const uint8_t *in_air, float dt, uint64_t time_us, float *q_sp_out,
			  float *thrust_sp_body_out, float *yawspeed_out, float *hover_thrust_out, uint8_t *outcome_out);

	/// q_sp == nullptr uses the setpoint stored by positionStep; see px4rl_cuda_attitude_step.
	void attitudeStep(const float *q, const float *q_sp, const float *yawspeed_ff, const float *thrust_sp_body,
			  float *rates_sp_out, float *thrust_body_out);

	void rateStep(const float *rates_sp, const float *thrust_body, const float *rates, const float *angular_accel,
		      float dt, const uint8_t *landed, float *motors_out, float *actuator_out, float *torque_sp_out);

	/// Same arguments as px4rl_cuda_rotor_step, on host pointers; see c_api.h. Throws std::invalid_argument if
	/// dt is not a positive finite number.
	void rotorStep(const float *actuator, const uint8_t *armed, const float *body_velocity_flu,
		       const float *body_rates_flu, float dt, float *force_flu_out, float *torque_flu_out,
		       float *rotor_force_flu_out);

private:
	std::vector<Vehicle> vehicles_;
	std::vector<VehiclePlant> plants_;
};

} // namespace px4rl
