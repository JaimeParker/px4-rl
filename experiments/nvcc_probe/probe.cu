// nvcc device-compilation probe for the PX4 inner loop in px4/.
//
// The same Drone::step (attitude -> rate -> allocator -> actuator output ->
// toy rigid-body plant) runs as a host loop and as a CUDA kernel over N
// drones. Drone objects persist in device memory between launches
// (placement-new init kernel, masked re-init kernel for resets). Each step
// both sides' records are compared.
//
// Build: see run_probe.sh. PX4RL_HD must expand to __host__ __device__.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include <cuda_runtime.h>

#include "AttitudeControllerWrapper.hpp"
#include "RateControllerWrapper.hpp"
#include "ControlAllocatorWrapper.hpp"
#include "ActuatorOutputWrapper.hpp"

// Unity build: every implementation file is compiled for host and device.
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

using namespace px4_stack;
using matrix::Quatf;
using matrix::Vector3f;

#define CUDA_CHECK(x)                                                                          \
	do {                                                                                       \
		cudaError_t err_ = (x);                                                                \
		if (err_ != cudaSuccess) {                                                             \
			std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
				     __LINE__);                                                    \
			std::exit(1);                                                                  \
		}                                                                                      \
	} while (0)

constexpr int kRecordWidth = 25;  // motors4 torque3 rates_sp3 omega3 q4 pwm4 base_motors4
constexpr float kDt = 0.004f;
constexpr int kSetpointPeriod = 100;

struct Rng {
	uint32_t state;
	PX4RL_HD float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}
	PX4RL_HD float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

struct Drone {
	AttitudeControllerWrapper att;
	RateControllerWrapper rate;
	ControlAllocatorWrapper alloc;
	ActuatorOutputWrapper out;
	// Same allocator again, driven only through a base-class pointer that is
	// written by the init kernel and dereferenced by later launches.
	ControlAllocationSequentialDesaturation seq;
	ControlAllocation *base{nullptr};

	Rng rng{};
	AttitudeSetpoint sp;
	Quatf q;
	Vector3f omega;
	Vector3f omega_prev;

	PX4RL_HD explicit Drone(uint32_t seed)
	{
		rng.state = seed * 2654435761u + 12345u;

		RateControlParams rp;
		const float k = rng.range(0.8f, 1.2f);
		rp.gain_k = Vector3f(k, k, k);
		rate.setParams(rp);

		ControlAllocatorParams ap;
		ap.airmode = static_cast<int>(seed % 3u);
		alloc.setParams(ap);

		ActuatorOutputParams op;
		op.thrust_model_factor = 0.3f;
		out.setParams(op);

		ControlAllocation::ActuatorVector zeros, ones;
		zeros.setAll(0.f);
		ones.setAll(1.f);
		base = &seq;
		base->setEffectivenessMatrix(alloc.getEffectivenessMatrix(), zeros, zeros, 4, true);
		base->setActuatorMin(zeros);
		base->setActuatorMax(ones);
		base->setNormalizeRPY(true);
		seq.setAirmode(ap.airmode);

		// Draw into named locals: argument evaluation order differs between host and device compilers.
		const float roll0 = rng.range(-0.3f, 0.3f);
		const float pitch0 = rng.range(-0.3f, 0.3f);
		const float yaw0 = rng.range(-3.f, 3.f);
		q = Quatf(matrix::Eulerf(roll0, pitch0, yaw0));
		const float wx = rng.range(-2.f, 2.f);
		const float wy = rng.range(-2.f, 2.f);
		const float wz = rng.range(-2.f, 2.f);
		omega = Vector3f(wx, wy, wz);
		omega_prev = omega;
		newSetpoint();
	}

	PX4RL_HD void newSetpoint()
	{
		const float roll = rng.range(-0.8f, 0.8f);
		const float pitch = rng.range(-0.8f, 0.8f);
		const float yaw = rng.range(-3.1f, 3.1f);
		sp.q_d = Quatf(matrix::Eulerf(roll, pitch, yaw));
		sp.thrust_body = Vector3f(0.f, 0.f, -rng.range(0.3f, 0.7f));
	}

	// open_loop: state (q, omega) is drawn from the RNG each step instead of
	// integrated, so controller outputs cannot feed back into their inputs.
	PX4RL_HD void step(int t, bool open_loop, float *rec)
	{
		if (t > 0 && t % kSetpointPeriod == 0) {
			newSetpoint();
		}

		if (open_loop) {
			const float roll = rng.range(-0.6f, 0.6f);
			const float pitch = rng.range(-0.6f, 0.6f);
			const float yaw = rng.range(-3.f, 3.f);
			q = Quatf(matrix::Eulerf(roll, pitch, yaw));
			const float wx = rng.range(-3.f, 3.f);
			const float wy = rng.range(-3.f, 3.f);
			const float wz = rng.range(-3.f, 3.f);
			omega = Vector3f(wx, wy, wz);
		}

		AttitudeState st;
		st.q = q;
		const RateSetpoint rsp = att.update(st, sp);

		RateControlInput ri;
		ri.rates_sp = rsp.rates;
		ri.rates_actual = omega;
		ri.angular_accel = (omega - omega_prev) / kDt;
		ri.thrust_body = rsp.thrust_body;
		ri.dt = kDt;
		RateControlOutput ro;
		rate.update(ri, ro);

		ControlAllocatorInput ai;
		ai.torque_sp = ro.torque_sp;
		ai.thrust_sp = ro.thrust_body;
		ai.dt = kDt;
		ControlAllocatorOutput ao;
		alloc.update(ai, ao);
		rate.setSaturationStatus(ao.saturation_positive, ao.saturation_negative);

		ActuatorOutputInput oi;
		for (int i = 0; i < 4; ++i) {
			oi.motors[i] = ao.motors[i];
		}
		ActuatorOutputOutput oo;
		out.update(oi, oo);

		matrix::Vector<float, 6> control_sp;
		control_sp(0) = ai.torque_sp(0);
		control_sp(1) = ai.torque_sp(1);
		control_sp(2) = ai.torque_sp(2);
		control_sp(3) = ai.thrust_sp(0);
		control_sp(4) = ai.thrust_sp(1);
		control_sp(5) = ai.thrust_sp(2);
		base->setControlSetpoint(control_sp);
		base->allocate();
		base->clipActuatorSetpoint();

		// Toy plant: allocated normalized torque -> body angular acceleration.
		omega_prev = omega;
		if (!open_loop) {
			const Vector3f tau = ai.torque_sp - ao.unallocated_torque;
			const Vector3f alpha(80.f * tau(0) - 0.5f * omega(0), 80.f * tau(1) - 0.5f * omega(1),
					     25.f * tau(2) - 0.5f * omega(2));
			omega += alpha * kDt;
			q = q * Quatf(matrix::AxisAnglef(omega * kDt));
			q.normalize();
		}

		int k = 0;
		for (int i = 0; i < 4; ++i) { rec[k++] = ao.motors[i]; }
		for (int i = 0; i < 3; ++i) { rec[k++] = ro.torque_sp(i); }
		for (int i = 0; i < 3; ++i) { rec[k++] = rsp.rates(i); }
		for (int i = 0; i < 3; ++i) { rec[k++] = omega(i); }
		for (int i = 0; i < 4; ++i) { rec[k++] = q(i); }
		for (int i = 0; i < 4; ++i) { rec[k++] = static_cast<float>(oo.values[i]); }
		for (int i = 0; i < 4; ++i) { rec[k++] = base->getActuatorSetpoint()(i); }
	}
};

__global__ void init_kernel(Drone *drones, int n, const uint8_t *reset_mask, uint32_t seed_offset)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) { return; }
	if (reset_mask != nullptr) {
		if (!reset_mask[i]) { return; }
		drones[i].~Drone();
	}
	new (&drones[i]) Drone(static_cast<uint32_t>(i) + seed_offset);
}

__global__ void step_kernel(Drone *drones, int n, int t, bool open_loop, float *records)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) { drones[i].step(t, open_loop, records + static_cast<size_t>(i) * kRecordWidth); }
}

// |host - device| histogram buckets: exact, then decades up to >1e-2.
constexpr int kBuckets = 8;
static const char *kBucketNames[kBuckets] = {"==", "<1e-7", "<1e-6", "<1e-5", "<1e-4", "<1e-3", "<1e-2", ">=1e-2"};
static int bucket_of(float h, float g)
{
	if (std::memcmp(&h, &g, sizeof(float)) == 0) { return 0; }
	const float d = std::fabs(h - g);
	float edge = 1e-7f;
	for (int b = 1; b < kBuckets - 1; ++b, edge *= 10.f) {
		if (d < edge) { return b; }
	}
	return kBuckets - 1;
}

static bool should_reset(int i) { return i % 7 == 0; }

static const char *kGroups[] = {"motors", "torque_sp", "rates_sp", "omega", "q", "pwm", "base_motors"};
static const int kGroupStart[] = {0, 4, 7, 10, 13, 17, 21, 25};

static void run_parity(int n, int steps, bool open_loop)
{
	const int reset_step = steps / 2;
	const uint32_t reset_seed_offset = 1000003u;
	const int threads = 128;
	const int blocks = (n + threads - 1) / threads;

	std::vector<Drone> host;
	host.reserve(n);
	for (int i = 0; i < n; ++i) { host.emplace_back(static_cast<uint32_t>(i)); }

	Drone *d_drones = nullptr;
	float *d_rec = nullptr;
	uint8_t *d_mask = nullptr;
	CUDA_CHECK(cudaMalloc(&d_drones, sizeof(Drone) * n));
	CUDA_CHECK(cudaMalloc(&d_rec, sizeof(float) * n * kRecordWidth));
	CUDA_CHECK(cudaMalloc(&d_mask, n));
	std::vector<uint8_t> mask(n);
	for (int i = 0; i < n; ++i) { mask[i] = should_reset(i) ? 1 : 0; }
	CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), n, cudaMemcpyHostToDevice));
	init_kernel<<<blocks, threads>>>(d_drones, n, nullptr, 0u);
	CUDA_CHECK(cudaGetLastError());

	std::vector<float> h_rec(static_cast<size_t>(n) * kRecordWidth), g_rec(h_rec.size());
	long long hist_all[7][kBuckets] = {};
	long long hist_step0[7][kBuckets] = {};
	long long nonfinite = 0;
	float max_base_vs_wrapper = 0.f;
	float max_all[7] = {}, max_step0[7] = {};
	float worst0_host[7] = {}, worst0_dev[7] = {};
	float max_diff_after_reset_step0[7] = {};

	for (int t = 0; t < steps; ++t) {
		if (t == reset_step) {
			for (int i = 0; i < n; ++i) {
				if (mask[i]) {
					host[i].~Drone();
					new (&host[i]) Drone(static_cast<uint32_t>(i) + reset_seed_offset);
				}
			}
			init_kernel<<<blocks, threads>>>(d_drones, n, d_mask, reset_seed_offset);
			CUDA_CHECK(cudaGetLastError());
		}
		for (int i = 0; i < n; ++i) { host[i].step(t, open_loop, &h_rec[static_cast<size_t>(i) * kRecordWidth]); }
		step_kernel<<<blocks, threads>>>(d_drones, n, t, open_loop, d_rec);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaMemcpy(g_rec.data(), d_rec, sizeof(float) * g_rec.size(), cudaMemcpyDeviceToHost));

		for (int i = 0; i < n; ++i) {
			const float *h = &h_rec[static_cast<size_t>(i) * kRecordWidth];
			const float *g = &g_rec[static_cast<size_t>(i) * kRecordWidth];
			for (int grp = 0; grp < 7; ++grp) {
				for (int k = kGroupStart[grp]; k < kGroupStart[grp + 1]; ++k) {
					if (!std::isfinite(h[k]) || !std::isfinite(g[k])) { ++nonfinite; }
					const int b = bucket_of(h[k], g[k]);
					const float d = std::fabs(h[k] - g[k]);
					++hist_all[grp][b];
					max_all[grp] = std::fmax(max_all[grp], d);
					if (t == 0) {
						++hist_step0[grp][b];
						if (d > max_step0[grp]) {
							max_step0[grp] = d;
							worst0_host[grp] = h[k];
							worst0_dev[grp] = g[k];
						}
					}
					if (t == reset_step && mask[i]) {
						max_diff_after_reset_step0[grp] =
							std::fmax(max_diff_after_reset_step0[grp], std::fabs(h[k] - g[k]));
					}
				}
			}
			for (int k = 0; k < 4; ++k) {
				max_base_vs_wrapper = std::fmax(max_base_vs_wrapper, std::fabs(h[21 + k] - h[k]));
			}
		}
	}

	auto print_hist = [](const char *title, long long (*hist)[kBuckets]) {
		std::printf("%s\n  %-12s", title, "field");
		for (int b = 0; b < kBuckets; ++b) { std::printf(" %10s", kBucketNames[b]); }
		std::printf("\n");
		for (int grp = 0; grp < 7; ++grp) {
			std::printf("  %-12s", kGroups[grp]);
			for (int b = 0; b < kBuckets; ++b) { std::printf(" %10lld", hist[grp][b]); }
			std::printf("\n");
		}
	};
	std::printf("\n== %s parity host vs device: %d steps x %d drones, reset at %d (i%%7==0) ==\n",
		    open_loop ? "OPEN-LOOP" : "CLOSED-LOOP", steps, n, reset_step);
	std::printf("non-finite values: %lld\n", nonfinite);
	print_hist("|host-device| histogram, step 0 only:", hist_step0);
	print_hist("|host-device| histogram, all steps:", hist_all);
	std::printf("max |host-device| per field (step 0 | all steps), with host/device value at the step-0 worst:\n");
	for (int grp = 0; grp < 7; ++grp) {
		std::printf("  %-12s %.3e | %.3e   step0 worst host=%.6f device=%.6f\n", kGroups[grp], max_step0[grp],
			    max_all[grp], worst0_host[grp], worst0_dev[grp]);
	}
	std::printf("re-initialised drones, first step after reset, max |host-device|:");
	for (int grp = 0; grp < 7; ++grp) { std::printf(" %s=%.1e", kGroups[grp], max_diff_after_reset_step0[grp]); }
	std::printf("\nhost: max |base-pointer allocator - wrapper allocator| motors = %.3e\n", max_base_vs_wrapper);
	float omega_abs = 0.f;
	for (int i = 0; i < n; ++i) {
		for (int k = 10; k < 13; ++k) { omega_abs = std::fmax(omega_abs, std::fabs(h_rec[(size_t)i * kRecordWidth + k])); }
	}
	std::printf("final max |omega| = %.3f rad/s\n", omega_abs);
	CUDA_CHECK(cudaFree(d_drones));
	CUDA_CHECK(cudaFree(d_rec));
	CUDA_CHECK(cudaFree(d_mask));
}

int main(int argc, char **argv)
{
	const int n = argc > 1 ? std::atoi(argv[1]) : 4096;
	const int steps = argc > 2 ? std::atoi(argv[2]) : 600;
	const int threads = 128;
	std::printf("sizeof: Drone=%zu AttitudeControllerWrapper=%zu RateControllerWrapper=%zu "
		    "ControlAllocatorWrapper=%zu ControlAllocationSequentialDesaturation=%zu ActuatorOutputWrapper=%zu\n",
		    sizeof(Drone), sizeof(AttitudeControllerWrapper), sizeof(RateControllerWrapper),
		    sizeof(ControlAllocatorWrapper), sizeof(ControlAllocationSequentialDesaturation),
		    sizeof(ActuatorOutputWrapper));
	size_t stack_limit = 0;
	CUDA_CHECK(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));
	std::printf("cudaLimitStackSize (default, not changed) = %zu bytes/thread\n", stack_limit);
	run_parity(n, steps, true);
	run_parity(n, steps, false);

	// Throughput: device step without per-step copies, host single-thread step.
	const int bench_steps = 500;
	for (int bench_n : {4096, 16384, 65536}) {
		Drone *bd = nullptr;
		float *br = nullptr;
		CUDA_CHECK(cudaMalloc(&bd, sizeof(Drone) * bench_n));
		CUDA_CHECK(cudaMalloc(&br, sizeof(float) * bench_n * kRecordWidth));
		const int bb = (bench_n + threads - 1) / threads;
		init_kernel<<<bb, threads>>>(bd, bench_n, nullptr, 0u);
		step_kernel<<<bb, threads>>>(bd, bench_n, 0, false, br);
		CUDA_CHECK(cudaDeviceSynchronize());
		cudaEvent_t a, b;
		cudaEventCreate(&a);
		cudaEventCreate(&b);
		cudaEventRecord(a);
		for (int t = 1; t <= bench_steps; ++t) { step_kernel<<<bb, threads>>>(bd, bench_n, t, false, br); }
		cudaEventRecord(b);
		CUDA_CHECK(cudaEventSynchronize(b));
		float ms = 0.f;
		cudaEventElapsedTime(&ms, a, b);
		std::printf("device step N=%6d: %8.1f us/step\n", bench_n, 1000.f * ms / bench_steps);
		CUDA_CHECK(cudaFree(bd));
		CUDA_CHECK(cudaFree(br));
	}
	{
		const int hn = 4096, hs = 50;
		std::vector<Drone> hb;
		hb.reserve(hn);
		for (int i = 0; i < hn; ++i) { hb.emplace_back(static_cast<uint32_t>(i)); }
		std::vector<float> r(static_cast<size_t>(hn) * kRecordWidth);
		const auto t0 = std::chrono::steady_clock::now();
		for (int t = 0; t < hs; ++t) {
			for (int i = 0; i < hn; ++i) { hb[i].step(t, false, &r[static_cast<size_t>(i) * kRecordWidth]); }
		}
		const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
		std::printf("host 1-thread step N=%6d: %8.1f us/step\n", hn, us / hs);
	}
	return 0;
}
