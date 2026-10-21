// Time per position_step, attitude_step, rate_step and rotor_step call of the CUDA fleet, measured with CUDA events.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "px4rl/c_api.h"

namespace {

constexpr float kDt = 0.004f;
constexpr int kWarmup = 20;
constexpr int kIterations = 500;

#define CUDA_CALL(expr)                                                                                      \
	do {                                                                                                 \
		const cudaError_t cuda_call_error = (expr);                                                  \
		if (cuda_call_error != cudaSuccess) {                                                        \
			std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_call_error)); \
			std::abort();                                                                        \
		}                                                                                            \
	} while (0)

#define API_CALL(expr)                                                                                      \
	do {                                                                                                \
		if ((expr) != 0) {                                                                          \
			std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, px4rl_last_error());        \
			std::abort();                                                                       \
		}                                                                                           \
	} while (0)

float *deviceBuffer(const std::vector<float> &host)
{
	float *device = nullptr;
	CUDA_CALL(cudaMalloc(&device, host.size() * sizeof(float)));
	CUDA_CALL(cudaMemcpy(device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
	return device;
}

float *deviceScratch(size_t count)
{
	float *device = nullptr;
	CUDA_CALL(cudaMalloc(&device, count * sizeof(float)));
	return device;
}

// Mean microseconds per call of `call`, after warm-up, on the default stream.
template <typename Call>
double microsecondsPerCall(Call &&call)
{
	for (int i = 0; i < kWarmup; ++i) {
		call();
	}

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord(start, nullptr));

	for (int i = 0; i < kIterations; ++i) {
		call();
	}

	CUDA_CALL(cudaEventRecord(stop, nullptr));
	CUDA_CALL(cudaEventSynchronize(stop));
	float milliseconds = 0.f;
	CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
	CUDA_CALL(cudaEventDestroy(start));
	CUDA_CALL(cudaEventDestroy(stop));
	return 1000.0 * static_cast<double>(milliseconds) / kIterations;
}

void benchmark(int n)
{
	px4rl_cuda_fleet *fleet = nullptr;
	API_CALL(px4rl_cuda_fleet_create(n, 0, &fleet));

	// Constant, non-trivial inputs: the cost per step does not depend on the values.
	std::vector<float> q(4 * n, 0.f), q_sp(4 * n, 0.f), thrust_sp(3 * n, 0.f), rates(3 * n, 0.f);

	for (int i = 0; i < n; ++i) {
		q[4 * i] = 1.f;
		q_sp[4 * i] = 0.98f;
		q_sp[4 * i + 1] = 0.199f;
		thrust_sp[3 * i + 2] = -0.5f;
		rates[3 * i] = 0.1f;
		rates[3 * i + 1] = -0.2f;
	}

	float *d_q = deviceBuffer(q);
	float *d_q_sp = deviceBuffer(q_sp);
	float *d_thrust_sp = deviceBuffer(thrust_sp);
	float *d_rates = deviceBuffer(rates);
	float *d_accel = deviceBuffer(std::vector<float>(3 * n, 0.f));
	float *d_rates_sp = deviceScratch(3 * n);
	float *d_thrust_body = deviceScratch(3 * n);
	float *d_motors = deviceScratch(4 * n);
	float *d_actuator = deviceScratch(4 * n);
	float *d_torque = deviceScratch(3 * n);
	float *d_pwm = deviceBuffer(std::vector<float>(4 * n, 1694.f));
	float *d_body_velocity = deviceBuffer(std::vector<float>(3 * n, 0.5f));
	float *d_body_rates = deviceBuffer(std::vector<float>(3 * n, 0.1f));
	float *d_force = deviceScratch(3 * n);
	float *d_rotor_torque = deviceScratch(3 * n);
	float *d_rotor_force = deviceScratch(12 * n);

	// Velocity command in x at hover, 5 m above the origin.
	std::vector<float> position(3 * n, 0.f), trajectory_sp(11 * n, NAN);

	for (int i = 0; i < n; ++i) {
		position[3 * i + 2] = -5.f;
		trajectory_sp[11 * i + 3] = 1.f;
		trajectory_sp[11 * i + 4] = trajectory_sp[11 * i + 5] = 0.f;
		trajectory_sp[11 * i + 9] = 0.f;
	}

	float *d_position = deviceBuffer(position);
	float *d_velocity = deviceBuffer(std::vector<float>(3 * n, 0.f));
	float *d_heading = deviceBuffer(std::vector<float>(n, 0.f));
	float *d_az = deviceBuffer(std::vector<float>(n, 0.f));
	float *d_trajectory_sp = deviceBuffer(trajectory_sp);
	float *d_q_sp_out = deviceScratch(4 * n);
	float *d_thrust_sp_out = deviceScratch(3 * n);
	float *d_yawspeed = deviceScratch(n);
	float *d_hover_thrust = deviceScratch(n);
	uint64_t time_us = 1000000;

	const double position_us = microsecondsPerCall([&] {
		time_us += 20000;
		API_CALL(px4rl_cuda_position_step(fleet, d_position, d_velocity, d_heading, d_az, d_trajectory_sp, nullptr,
						  0.02f, time_us, d_q_sp_out, d_thrust_sp_out, d_yawspeed, d_hover_thrust,
						  nullptr, nullptr));
	});
	const double attitude_us = microsecondsPerCall([&] {
		API_CALL(px4rl_cuda_attitude_step(fleet, d_q, d_q_sp, nullptr, d_thrust_sp, d_rates_sp, d_thrust_body,
						  nullptr));
	});
	const double rate_us = microsecondsPerCall([&] {
		API_CALL(px4rl_cuda_rate_step(fleet, d_rates_sp, d_thrust_body, d_rates, d_accel, kDt, nullptr, d_motors,
					      d_actuator, d_torque, nullptr));
	});
	const double rotor_us = microsecondsPerCall([&] {
		API_CALL(px4rl_cuda_rotor_step(fleet, d_pwm, nullptr, d_body_velocity, d_body_rates, kDt, d_force,
					       d_rotor_torque, d_rotor_force, nullptr));
	});
	std::printf("bench N=%6d: position_step %9.2f us/call, attitude_step %9.2f us/call, rate_step %9.2f us/call, "
		    "rotor_step %9.2f us/call\n",
		    n, position_us, attitude_us, rate_us, rotor_us);

	for (float *p : {d_q, d_q_sp, d_thrust_sp, d_rates, d_accel, d_rates_sp, d_thrust_body, d_motors, d_actuator,
			 d_torque, d_position, d_velocity, d_heading, d_az, d_trajectory_sp, d_q_sp_out, d_thrust_sp_out,
			 d_yawspeed, d_hover_thrust, d_pwm, d_body_velocity, d_body_rates, d_force,
			 d_rotor_torque, d_rotor_force}) {
		CUDA_CALL(cudaFree(p));
	}

	API_CALL(px4rl_cuda_fleet_destroy(fleet));
}

} // namespace

int main()
{
	std::printf("bench: %d timed calls per measurement, CUDA events, default stream\n", kIterations);

	for (int n : {4096, 16384, 65536}) {
		benchmark(n);
	}

	return 0;
}
