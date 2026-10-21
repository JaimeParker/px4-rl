#include "px4rl/cpu_fleet.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "px4rl/airframes.h"
#include "px4rl/step_io.hpp"

namespace px4rl {

namespace {
// Below this size the OpenMP fork/join costs more than the loop.
constexpr int kMinParallelVehicles = 64;

// OpenMP requires a positive num_threads clause, so the "default" setting is resolved per call.
int resolveThreads(int num_threads)
{
#ifdef _OPENMP
	return num_threads > 0 ? num_threads : omp_get_max_threads();
#else
	(void)num_threads;
	return 1;
#endif
}
} // namespace

CpuFleet::CpuFleet(int n, int num_threads) : num_threads_(num_threads)
{
	if (n < 1) {
		throw std::invalid_argument("CpuFleet: n must be >= 1");
	}

	if (num_threads < 0) {
		throw std::invalid_argument("CpuFleet: num_threads must be >= 0");
	}

	vehicles_.resize(static_cast<size_t>(n));
	const px4rl_params defaults = defaultParams();

	for (Vehicle &vehicle : vehicles_) {
		vehicle.configure(defaults);
	}

	const px4rl_plant_params iris = px4rl_plant_params_iris();
	plants_.resize(static_cast<size_t>(n));

	for (VehiclePlant &plant : plants_) {
		plant.configure(iris);
	}
}

void CpuFleet::reset(const uint8_t *mask, const px4rl_params *params, int n_params)
{
	const int n = size();

	if (params == nullptr || (n_params != 1 && n_params != n)) {
		throw std::invalid_argument("CpuFleet::reset: n_params must be 1 or n");
	}

	for (int i = 0; i < n_params; ++i) {
		if (const char *error = validateParams(params[i])) {
			throw std::invalid_argument(std::string("CpuFleet::reset: ") + error);
		}
	}

	const int stride = n_params == 1 ? 0 : 1;
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		if (mask == nullptr || mask[i] != 0) {
			vehicles_[i] = Vehicle();
			vehicles_[i].configure(params[stride * i]);
		}
	}
}

void CpuFleet::resetPlant(const uint8_t *mask, const px4rl_plant_params *plants, int n_plants)
{
	const int n = size();

	if (plants == nullptr || (n_plants != 1 && n_plants != n)) {
		throw std::invalid_argument("CpuFleet::resetPlant: n_plants must be 1 or n");
	}

	for (int i = 0; i < n_plants; ++i) {
		if (const char *error = validatePlantParams(plants[i])) {
			throw std::invalid_argument(std::string("CpuFleet::resetPlant: ") + error);
		}
	}

	const int stride = n_plants == 1 ? 0 : 1;
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		if (mask == nullptr || mask[i] != 0) {
			plants_[i].configure(plants[stride * i]);
		}
	}
}

void CpuFleet::positionStep(const float *position, const float *velocity, const float *heading,
			    const float *acceleration_z, const float *trajectory_sp, const uint8_t *in_air, float dt,
			    uint64_t time_us, float *q_sp_out, float *thrust_sp_body_out, float *yawspeed_out,
			    float *hover_thrust_out, uint8_t *outcome_out)
{
	const PositionStepBuffers buffers{position,     velocity,           heading,   acceleration_z, trajectory_sp,
					  in_air,       dt,                 time_us,   q_sp_out,       thrust_sp_body_out,
					  yawspeed_out, hover_thrust_out,   outcome_out};
	const int n = size();
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		positionStepAt(vehicles_[i], i, buffers);
	}
}

void CpuFleet::attitudeStep(const float *q, const float *q_sp, const float *yawspeed_ff, const float *thrust_sp_body,
			    float *rates_sp_out, float *thrust_body_out)
{
	const AttitudeStepBuffers buffers{q, q_sp, yawspeed_ff, thrust_sp_body, rates_sp_out, thrust_body_out};
	const int n = size();
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		attitudeStepAt(vehicles_[i], i, buffers);
	}
}

void CpuFleet::rateStep(const float *rates_sp, const float *thrust_body, const float *rates,
			const float *angular_accel, float dt, const uint8_t *landed, float *motors_out,
			float *actuator_out, float *torque_sp_out)
{
	const RateStepBuffers buffers{rates_sp, thrust_body, rates, angular_accel, dt, landed, motors_out,
				      actuator_out, torque_sp_out};
	const int n = size();
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		rateStepAt(vehicles_[i], i, buffers);
	}
}

void CpuFleet::rotorStep(const float *actuator, const uint8_t *armed, const float *body_velocity_flu,
			 const float *body_rates_flu, float dt, float *force_flu_out, float *torque_flu_out,
			 float *rotor_force_flu_out)
{
	if (!(dt > 0.f) || !std::isfinite(dt)) {
		throw std::invalid_argument("CpuFleet::rotorStep: dt must be positive and finite");
	}

	const RotorStepBuffers buffers{actuator,      armed,          body_velocity_flu, body_rates_flu,
				       dt,            force_flu_out,  torque_flu_out,    rotor_force_flu_out};
	const int n = size();
	[[maybe_unused]] const int threads = resolveThreads(num_threads_);

#pragma omp parallel for schedule(static) if (n >= kMinParallelVehicles) num_threads(threads)
	for (int i = 0; i < n; ++i) {
		rotorStepAt(plants_[i], i, buffers);
	}
}

} // namespace px4rl
