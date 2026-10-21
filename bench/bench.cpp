// Time per attitude_step and rate_step call of the CUDA fleet, measured with CUDA events.
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

	const double attitude_us = microsecondsPerCall([&] {
		API_CALL(px4rl_cuda_attitude_step(fleet, d_q, d_q_sp, nullptr, d_thrust_sp, d_rates_sp, d_thrust_body,
						  nullptr));
	});
	const double rate_us = microsecondsPerCall([&] {
		API_CALL(px4rl_cuda_rate_step(fleet, d_rates_sp, d_thrust_body, d_rates, d_accel, kDt, nullptr, d_motors,
					      d_actuator, d_torque, nullptr));
	});
	std::printf("bench N=%6d: attitude_step %9.2f us/call, rate_step %9.2f us/call\n", n, attitude_us, rate_us);

	for (float *p : {d_q, d_q_sp, d_thrust_sp, d_rates, d_accel, d_rates_sp, d_thrust_body, d_motors, d_actuator,
			 d_torque}) {
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
