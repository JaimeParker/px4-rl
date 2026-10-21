// The rotor model and frame conversions are PX4RL_HD code: this checks that the device build agrees with
// the host build on random inputs. Plain executable (no gtest), exit code 0 = pass.
//
// Tolerance: per output field, max |device - host| <= 1e-5 * max |host| over that field. Differences
// come from fused multiply-add and the device expf/atan2f, about 1e-7 relative.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "px4rl/frames.hpp"
#include "px4rl/gazebo_motor_model.hpp"

namespace {

constexpr int kVehicles = 4096;
constexpr int kSteps = 20;
constexpr float kDt = 0.002f;
constexpr float kTolerance = 1e-5f;

struct Case {
	px4rl_plant_params plant;
	float u[PX4RL_MAX_ROTORS];
	int armed;
	float body_velocity[3];
	float body_angular_velocity[3];
	float q_enu_flu[4];
	float q_flu_enu[4];
	float v_enu[3];
	float v_flu[3];
};

// Per-vehicle outputs: wrench after kSteps steps, rotor-0 speed state, frame conversions.
struct Result {
	float force[3];
	float torque[3];
	float rotor0_speed;
	float q_ned_frd[4];
	float q_frd_ned[4];
	float heading;
	float v_ned[3];
	float v_frd[3];
};

PX4RL_HD void evaluate(const Case &c, Result &r)
{
	using matrix::Vector3f;
	px4rl::RotorState states[PX4RL_MAX_ROTORS];
	px4rl::Wrench wrench;
	const Vector3f v(c.body_velocity[0], c.body_velocity[1], c.body_velocity[2]);
	const Vector3f w(c.body_angular_velocity[0], c.body_angular_velocity[1], c.body_angular_velocity[2]);

	for (int k = 0; k < kSteps; ++k) {
		wrench = px4rl::plantStep(c.plant, states, c.u, c.armed != 0, v, w, kDt);
	}

	for (int i = 0; i < 3; ++i) {
		r.force[i] = wrench.force(i);
		r.torque[i] = wrench.torque(i);
	}

	r.rotor0_speed = states[0].filtered_speed;

	const matrix::Quatf q_ned_frd = px4rl::quatWorldBodyEnuFluToNedFrd(matrix::Quatf(c.q_enu_flu[0], c.q_enu_flu[1], c.q_enu_flu[2], c.q_enu_flu[3]));
	const matrix::Quatf q_frd_ned = px4rl::quatBodyWorldEnuFluToNedFrd(matrix::Quatf(c.q_flu_enu[0], c.q_flu_enu[1], c.q_flu_enu[2], c.q_flu_enu[3]));

	for (int i = 0; i < 4; ++i) {
		r.q_ned_frd[i] = q_ned_frd(i);
		r.q_frd_ned[i] = q_frd_ned(i);
	}

	r.heading = px4rl::headingNedFromEnuFlu(matrix::Quatf(c.q_enu_flu[0], c.q_enu_flu[1], c.q_enu_flu[2], c.q_enu_flu[3]));
	const Vector3f v_ned = px4rl::enuToNed(Vector3f(c.v_enu[0], c.v_enu[1], c.v_enu[2]));
	const Vector3f v_frd = px4rl::fluToFrd(Vector3f(c.v_flu[0], c.v_flu[1], c.v_flu[2]));

	for (int i = 0; i < 3; ++i) {
		r.v_ned[i] = v_ned(i);
		r.v_frd[i] = v_frd(i);
	}
}

__global__ void kernel(const Case *cases, Result *results, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n) {
		evaluate(cases[i], results[i]);
	}
}

struct Lcg {
	uint32_t state;
	float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}
	float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

void unitQuaternion(Lcg &rng, float *q)
{
	float norm = 0.f;

	do {
		norm = 0.f;

		for (int i = 0; i < 4; ++i) {
			q[i] = rng.range(-1.f, 1.f);
			norm += q[i] * q[i];
		}
	} while (norm < 1e-3f || norm > 1.f);

	norm = std::sqrt(norm);

	for (int i = 0; i < 4; ++i) {
		q[i] /= norm;
	}
}

Case makeCase(Lcg &rng)
{
	Case c = {};
	c.plant.num_rotors = 4;

	for (int i = 0; i < 4; ++i) {
		px4rl_rotor_params &p = c.plant.rotor[i];

		for (int k = 0; k < 3; ++k) {
			p.position[k] = rng.range(-0.3f, 0.3f);
		}

		float axis[3] = {rng.range(-0.2f, 0.2f), rng.range(-0.2f, 0.2f), 1.f};
		const float norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + 1.f);

		for (int k = 0; k < 3; ++k) {
			p.axis[k] = axis[k] / norm;
		}

		p.turning_direction = (rng.uniform() < 0.5f) ? 1 : -1;
		p.reversible = (rng.uniform() < 0.25f) ? 1 : 0;
		p.motor_constant = rng.range(3e-6f, 9e-6f);
		p.moment_constant = rng.range(0.01f, 0.08f);
		p.time_constant_up = rng.range(0.005f, 0.03f);
		p.time_constant_down = rng.range(0.01f, 0.05f);
		p.max_rot_velocity = rng.range(800.f, 1200.f);
		p.rotor_drag_coefficient = rng.range(0.f, 3e-4f);
		p.rolling_moment_coefficient = rng.range(0.f, 2e-6f);
		p.max_relative_airspeed = rng.range(15.f, 30.f);
		p.input_offset = rng.range(-0.2f, 0.2f);
		p.input_scaling = rng.range(800.f, 1200.f);
		p.zero_position_armed = rng.range(0.f, 150.f);
		p.zero_position_disarmed = 0.f;
		p.px4_output_index = (i + 1) % 4; // exercise the indirection
		c.u[i] = rng.range(0.f, 1.f);
	}

	c.armed = (rng.uniform() < 0.9f) ? 1 : 0;

	for (int k = 0; k < 3; ++k) {
		c.body_velocity[k] = rng.range(-12.f, 12.f);
		c.body_angular_velocity[k] = rng.range(-3.f, 3.f);
		c.v_enu[k] = rng.range(-10.f, 10.f);
		c.v_flu[k] = rng.range(-10.f, 10.f);
	}

	unitQuaternion(rng, c.q_enu_flu);
	unitQuaternion(rng, c.q_flu_enu);
	return c;
}

#define CHECK_CUDA(call)                                                                                   \
	do {                                                                                                   \
		const cudaError_t err = (call);                                                                    \
		if (err != cudaSuccess) {                                                                          \
			std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));              \
			return 2;                                                                                      \
		}                                                                                                  \
	} while (0)

// Compares `count` consecutive floats starting at byte `offset` of each Result.
bool compareFloats(const char *name, const std::vector<Result> &host, const std::vector<Result> &device,
		   size_t offset, int count)
{
	float max_host = 0.f;
	float max_diff = 0.f;

	for (size_t i = 0; i < host.size(); ++i) {
		const float *h = reinterpret_cast<const float *>(reinterpret_cast<const char *>(&host[i]) + offset);
		const float *d = reinterpret_cast<const float *>(reinterpret_cast<const char *>(&device[i]) + offset);

		for (int k = 0; k < count; ++k) {
			if (!std::isfinite(h[k]) || !std::isfinite(d[k])) {
				std::fprintf(stderr, "%s: non-finite value at vehicle %zu\n", name, i);
				return false;
			}

			max_host = std::fmax(max_host, std::fabs(h[k]));
			max_diff = std::fmax(max_diff, std::fabs(h[k] - d[k]));
		}
	}

	const float relative = max_diff / std::fmax(max_host, 1e-30f);
	std::printf("%-14s max|host| %.6g  max|diff| %.3g  relative %.3g\n", name, max_host, max_diff, relative);
	return relative <= kTolerance;
}

} // namespace

int main()
{
	Lcg rng{2024u};
	std::vector<Case> cases(kVehicles);

	for (Case &c : cases) {
		c = makeCase(rng);
	}

	std::vector<Result> host(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		evaluate(cases[i], host[i]);
	}

	Case *d_cases = nullptr;
	Result *d_results = nullptr;
	CHECK_CUDA(cudaMalloc(&d_cases, sizeof(Case) * kVehicles));
	CHECK_CUDA(cudaMalloc(&d_results, sizeof(Result) * kVehicles));
	CHECK_CUDA(cudaMemcpy(d_cases, cases.data(), sizeof(Case) * kVehicles, cudaMemcpyHostToDevice));
	kernel<<<(kVehicles + 127) / 128, 128>>>(d_cases, d_results, kVehicles);
	CHECK_CUDA(cudaGetLastError());
	std::vector<Result> device(kVehicles);
	CHECK_CUDA(cudaMemcpy(device.data(), d_results, sizeof(Result) * kVehicles, cudaMemcpyDeviceToHost));
	cudaFree(d_cases);
	cudaFree(d_results);

	bool ok = true;
	ok &= compareFloats("force", host, device, offsetof(Result, force), 3);
	ok &= compareFloats("torque", host, device, offsetof(Result, torque), 3);
	ok &= compareFloats("rotor0_speed", host, device, offsetof(Result, rotor0_speed), 1);
	ok &= compareFloats("q_ned_frd", host, device, offsetof(Result, q_ned_frd), 4);
	ok &= compareFloats("q_frd_ned", host, device, offsetof(Result, q_frd_ned), 4);
	ok &= compareFloats("heading", host, device, offsetof(Result, heading), 1);
	ok &= compareFloats("v_ned", host, device, offsetof(Result, v_ned), 3);
	ok &= compareFloats("v_frd", host, device, offsetof(Result, v_frd), 3);
	std::printf("plant_cuda_test: %s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
