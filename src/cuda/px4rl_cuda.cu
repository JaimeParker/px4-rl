// CUDA fleet: persistent device-resident px4rl::Vehicle objects behind a C ABI.
//
// The px4 .cpp files are compiled into this translation unit (unity build) with
// PX4RL_HD = __host__ __device__, so this library does not link px4rl_px4. Only the px4rl_*
// functions declared in c_api.h are exported; everything else has hidden visibility, which lets
// the library coexist with px4rl_px4 in one process.

#ifndef PX4RL_HD
#error "PX4RL_HD must be defined as __host__ __device__ for this target"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include <cuda_runtime.h>

#include "px4rl/airframes.h"
#include "px4rl/c_api.h"
#include "px4rl/params.hpp"
#include "px4rl/step_io.hpp"
#include "px4rl/hover_thrust_module.hpp"
#include "px4rl/position_module.hpp"
#include "px4rl/vehicle_plant.hpp"

#include "AttitudeControl.cpp"
#include "AttitudeControllerWrapper.cpp"
#include "RateControl.cpp"
#include "RateControllerWrapper.cpp"
#include "ActuatorEffectiveness.cpp"
#include "ControlAllocation.cpp"
#include "ControlAllocationPseudoInverse.cpp"
#include "ControlAllocationSequentialDesaturation.cpp"
#include "ControlAllocatorWrapper.cpp"
#include "ActuatorOutputWrapper.cpp"
#include "PositionControl.cpp"
#include "ControlMath.cpp"
#include "zero_order_hover_thrust_ekf.cpp"
#include "hysteresis/hysteresis.cpp"

// px4rl_params_iris, px4rl_plant_params_iris and px4rl_rigid_body_params_iris, exported from this library.
#include "../airframes/iris.cpp"

struct px4rl_cuda_fleet {
	int n{0};
	int device{0};
	px4rl::Vehicle *vehicles{nullptr};
	px4rl_params *params{nullptr}; // n entries, written by create/reset
	px4rl::VehiclePlant *plants{nullptr};
	px4rl_plant_params *plant_params{nullptr}; // n entries, written by create/reset_plant
};

namespace {

constexpr int kThreadsPerBlock = 128;

constexpr int kErrorInvalidArgument = 1;
constexpr int kErrorCuda = 2;

thread_local char g_last_error[256] = "";

int failInvalid(const char *message)
{
	std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
	return kErrorInvalidArgument;
}

int failCuda(const char *what, cudaError_t error)
{
	std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", what, cudaGetErrorString(error));
	return kErrorCuda;
}

// Makes `device` current for the lifetime of the object and restores the previous device.
class DeviceScope {
public:
	explicit DeviceScope(int device)
	{
		error_ = cudaGetDevice(&previous_);

		if (error_ == cudaSuccess && previous_ != device) {
			error_ = cudaSetDevice(device);
			restore_ = error_ == cudaSuccess;
		}
	}

	~DeviceScope()
	{
		if (restore_) {
			cudaSetDevice(previous_);
		}
	}

	DeviceScope(const DeviceScope &) = delete;
	DeviceScope &operator=(const DeviceScope &) = delete;

	cudaError_t error() const { return error_; }

private:
	int previous_{0};
	bool restore_{false};
	cudaError_t error_{cudaSuccess};
};

int blocksFor(int n)
{
	return (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

// Constructs (placement new) the selected vehicles. params_stride is 0 for a broadcast parameter set.
__global__ void initKernel(px4rl::Vehicle *vehicles, int n, const uint8_t *mask, const px4rl_params *params,
			   int params_stride, bool destroy_previous)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n || (mask != nullptr && mask[i] == 0)) {
		return;
	}

	if (destroy_previous) {
		vehicles[i].~Vehicle();
	}

	new (&vehicles[i]) px4rl::Vehicle();
	vehicles[i].configure(params[params_stride * i]);
}

// Same as initKernel for the plants: constructs the selected ones from plant_params, rotors at rest.
__global__ void initPlantKernel(px4rl::VehiclePlant *plants, int n, const uint8_t *mask,
				const px4rl_plant_params *plant_params, int params_stride)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n || (mask != nullptr && mask[i] == 0)) {
		return;
	}

	new (&plants[i]) px4rl::VehiclePlant();
	plants[i].configure(plant_params[params_stride * i]);
}

__global__ void positionKernel(px4rl::Vehicle *vehicles, int n, px4rl::PositionStepBuffers buffers)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		px4rl::positionStepAt(vehicles[i], i, buffers);
	}
}

__global__ void attitudeKernel(px4rl::Vehicle *vehicles, int n, px4rl::AttitudeStepBuffers buffers)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		px4rl::attitudeStepAt(vehicles[i], i, buffers);
	}
}

__global__ void rateKernel(px4rl::Vehicle *vehicles, int n, px4rl::RateStepBuffers buffers)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		px4rl::rateStepAt(vehicles[i], i, buffers);
	}
}

__global__ void rotorKernel(px4rl::VehiclePlant *plants, int n, px4rl::RotorStepBuffers buffers)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		px4rl::rotorStepAt(plants[i], i, buffers);
	}
}

} // namespace

extern "C" {

void px4rl_params_default(px4rl_params *p)
{
	*p = px4rl::defaultParams();
}

const char *px4rl_last_error(void)
{
	return g_last_error;
}

int px4rl_cuda_fleet_create(int n, int device, px4rl_cuda_fleet **out)
{
	if (out == nullptr) {
		return failInvalid("px4rl_cuda_fleet_create: out is null");
	}

	*out = nullptr;

	if (n < 1) {
		return failInvalid("px4rl_cuda_fleet_create: n must be >= 1");
	}

	int device_count = 0;

	if (cudaError_t e = cudaGetDeviceCount(&device_count); e != cudaSuccess) {
		return failCuda("cudaGetDeviceCount", e);
	}

	if (device < 0 || device >= device_count) {
		return failInvalid("px4rl_cuda_fleet_create: device index out of range");
	}

	DeviceScope scope(device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	auto *fleet = new (std::nothrow) px4rl_cuda_fleet();

	if (fleet == nullptr) {
		return failInvalid("px4rl_cuda_fleet_create: out of host memory");
	}

	fleet->n = n;
	fleet->device = device;

	const auto release = [&](int status) {
		cudaFree(fleet->vehicles);
		cudaFree(fleet->params);
		cudaFree(fleet->plants);
		cudaFree(fleet->plant_params);
		delete fleet;
		return status;
	};

	if (cudaError_t e = cudaMalloc(&fleet->vehicles, sizeof(px4rl::Vehicle) * n); e != cudaSuccess) {
		return release(failCuda("cudaMalloc(vehicles)", e));
	}

	if (cudaError_t e = cudaMalloc(&fleet->params, sizeof(px4rl_params) * n); e != cudaSuccess) {
		return release(failCuda("cudaMalloc(params)", e));
	}

	if (cudaError_t e = cudaMalloc(&fleet->plants, sizeof(px4rl::VehiclePlant) * n); e != cudaSuccess) {
		return release(failCuda("cudaMalloc(plants)", e));
	}

	if (cudaError_t e = cudaMalloc(&fleet->plant_params, sizeof(px4rl_plant_params) * n); e != cudaSuccess) {
		return release(failCuda("cudaMalloc(plant params)", e));
	}

	const px4rl_params defaults = px4rl::defaultParams();

	if (cudaError_t e = cudaMemcpy(fleet->params, &defaults, sizeof(defaults), cudaMemcpyHostToDevice);
	    e != cudaSuccess) {
		return release(failCuda("cudaMemcpy(default params)", e));
	}

	const px4rl_plant_params iris_plant = px4rl_plant_params_iris();

	if (cudaError_t e = cudaMemcpy(fleet->plant_params, &iris_plant, sizeof(iris_plant), cudaMemcpyHostToDevice);
	    e != cudaSuccess) {
		return release(failCuda("cudaMemcpy(iris plant)", e));
	}

	initKernel<<<blocksFor(n), kThreadsPerBlock>>>(fleet->vehicles, n, nullptr, fleet->params, 0, false);
	initPlantKernel<<<blocksFor(n), kThreadsPerBlock>>>(fleet->plants, n, nullptr, fleet->plant_params, 0);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return release(failCuda("initKernel launch", e));
	}

	if (cudaError_t e = cudaStreamSynchronize(nullptr); e != cudaSuccess) {
		return release(failCuda("initKernel", e));
	}

	*out = fleet;
	return 0;
}

int px4rl_cuda_fleet_destroy(px4rl_cuda_fleet *f)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_fleet_destroy: fleet is null");
	}

	int status = 0;
	{
		DeviceScope scope(f->device);

		if (scope.error() != cudaSuccess) {
			status = failCuda("cudaSetDevice", scope.error());

		} else {
			cudaError_t e1 = cudaFree(f->vehicles);
			cudaError_t e2 = cudaFree(f->params);
			cudaError_t e3 = cudaFree(f->plants);
			cudaError_t e4 = cudaFree(f->plant_params);

			for (cudaError_t e : {e1, e2, e3, e4}) {
				if (e != cudaSuccess) {
					status = failCuda("cudaFree", e);
					break;
				}
			}
		}
	}
	delete f;
	return status;
}

int px4rl_cuda_fleet_size(const px4rl_cuda_fleet *f)
{
	if (f == nullptr) {
		failInvalid("px4rl_cuda_fleet_size: fleet is null");
		return -1;
	}

	return f->n;
}

int px4rl_cuda_fleet_reset(px4rl_cuda_fleet *f, const uint8_t *mask, const px4rl_params *params_host, int n_params,
			   void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_fleet_reset: fleet is null");
	}

	if (params_host == nullptr || (n_params != 1 && n_params != f->n)) {
		return failInvalid("px4rl_cuda_fleet_reset: n_params must be 1 or the fleet size");
	}

	for (int i = 0; i < n_params; ++i) {
		if (const char *error = px4rl::validateParams(params_host[i])) {
			return failInvalid(error);
		}
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const auto cuda_stream = static_cast<cudaStream_t>(stream);

	// Pageable host memory is staged before the call returns; the copy is ordered with the init kernel.
	if (cudaError_t e = cudaMemcpyAsync(f->params, params_host, sizeof(px4rl_params) * n_params,
					    cudaMemcpyHostToDevice, cuda_stream);
	    e != cudaSuccess) {
		return failCuda("cudaMemcpyAsync(params)", e);
	}

	initKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, cuda_stream>>>(f->vehicles, f->n, mask, f->params,
									   n_params == 1 ? 0 : 1, true);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("initKernel launch", e);
	}

	return 0;
}

int px4rl_cuda_fleet_reset_plant(px4rl_cuda_fleet *f, const uint8_t *mask, const px4rl_plant_params *plant_host,
				 int n_plant, void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_fleet_reset_plant: fleet is null");
	}

	if (plant_host == nullptr || (n_plant != 1 && n_plant != f->n)) {
		return failInvalid("px4rl_cuda_fleet_reset_plant: n_plant must be 1 or the fleet size");
	}

	for (int i = 0; i < n_plant; ++i) {
		if (const char *error = px4rl::validatePlantParams(plant_host[i])) {
			return failInvalid(error);
		}
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const auto cuda_stream = static_cast<cudaStream_t>(stream);

	// Pageable host memory is staged before the call returns; the copy is ordered with the init kernel.
	if (cudaError_t e = cudaMemcpyAsync(f->plant_params, plant_host, sizeof(px4rl_plant_params) * n_plant,
					    cudaMemcpyHostToDevice, cuda_stream);
	    e != cudaSuccess) {
		return failCuda("cudaMemcpyAsync(plant params)", e);
	}

	initPlantKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, cuda_stream>>>(f->plants, f->n, mask, f->plant_params,
									       n_plant == 1 ? 0 : 1);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("initPlantKernel launch", e);
	}

	return 0;
}

int px4rl_cuda_position_step(px4rl_cuda_fleet *f, const float *pos, const float *vel, const float *heading,
			     const float *az, const float *traj_sp, const uint8_t *in_air, float dt, uint64_t time_us,
			     float *q_sp_out, float *thrust_sp_body_out, float *yawspeed_out, float *hover_thrust_out,
			     uint8_t *outcome_out, void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_position_step: fleet is null");
	}

	if (pos == nullptr || vel == nullptr || heading == nullptr || az == nullptr || traj_sp == nullptr ||
	    q_sp_out == nullptr || thrust_sp_body_out == nullptr || yawspeed_out == nullptr) {
		return failInvalid("px4rl_cuda_position_step: required buffer is null");
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const px4rl::PositionStepBuffers buffers{pos,       vel,       heading,  az,           traj_sp,
						 in_air,    dt,        time_us,  q_sp_out,     thrust_sp_body_out,
						 yawspeed_out, hover_thrust_out, outcome_out};
	positionKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(f->vehicles, f->n,
												    buffers);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("positionKernel launch", e);
	}

	return 0;
}

int px4rl_cuda_attitude_step(px4rl_cuda_fleet *f, const float *q, const float *q_sp, const float *yawspeed_ff,
			     const float *thrust_sp_body, float *rates_sp_out, float *thrust_body_out, void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_attitude_step: fleet is null");
	}

	if (q == nullptr || thrust_sp_body == nullptr || rates_sp_out == nullptr || thrust_body_out == nullptr) {
		return failInvalid("px4rl_cuda_attitude_step: required buffer is null");
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const px4rl::AttitudeStepBuffers buffers{q, q_sp, yawspeed_ff, thrust_sp_body, rates_sp_out, thrust_body_out};
	attitudeKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(f->vehicles, f->n,
												    buffers);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("attitudeKernel launch", e);
	}

	return 0;
}

int px4rl_cuda_rate_step(px4rl_cuda_fleet *f, const float *rates_sp, const float *thrust_body, const float *rates,
			 const float *angular_accel, float dt, const uint8_t *landed, float *motors_out,
			 float *actuator_out, float *torque_sp_out, void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_rate_step: fleet is null");
	}

	if (rates_sp == nullptr || thrust_body == nullptr || rates == nullptr || angular_accel == nullptr ||
	    motors_out == nullptr) {
		return failInvalid("px4rl_cuda_rate_step: required buffer is null");
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const px4rl::RateStepBuffers buffers{rates_sp, thrust_body, rates, angular_accel, dt, landed, motors_out,
					     actuator_out, torque_sp_out};
	rateKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(f->vehicles, f->n,
											       buffers);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("rateKernel launch", e);
	}

	return 0;
}

int px4rl_cuda_rotor_step(px4rl_cuda_fleet *f, const float *actuator, const uint8_t *armed,
			  const float *body_velocity_flu, const float *body_rates_flu, float dt, float *force_flu_out,
			  float *torque_flu_out, float *rotor_force_flu_out, void *stream)
{
	if (f == nullptr) {
		return failInvalid("px4rl_cuda_rotor_step: fleet is null");
	}

	if (actuator == nullptr || body_velocity_flu == nullptr || force_flu_out == nullptr ||
	    torque_flu_out == nullptr) {
		return failInvalid("px4rl_cuda_rotor_step: required buffer is null");
	}

	if (!(dt > 0.f) || !std::isfinite(dt)) {
		return failInvalid("px4rl_cuda_rotor_step: dt must be positive and finite");
	}

	DeviceScope scope(f->device);

	if (scope.error() != cudaSuccess) {
		return failCuda("cudaSetDevice", scope.error());
	}

	const px4rl::RotorStepBuffers buffers{actuator,      armed,         body_velocity_flu, body_rates_flu,
					      dt,            force_flu_out, torque_flu_out,    rotor_force_flu_out};
	rotorKernel<<<blocksFor(f->n), kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(f->plants, f->n,
												 buffers);

	if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
		return failCuda("rotorKernel launch", e);
	}

	return 0;
}

} // extern "C"
