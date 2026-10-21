// CPU (CpuFleet) versus GPU (libpx4rl_cuda) agreement of the full Vehicle chain.
//
// Inputs are open loop: they are drawn on the host independently of any controller output and
// uploaded to both fleets, so differences come only from the two floating-point implementations.
// Thresholds: docs/architecture.md, "CPU vs GPU parity".
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include <matrix/matrix/math.hpp>

#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"

namespace {

constexpr int kVehicles = 4096;
constexpr int kSteps = 600;
constexpr int kResetStep = 300;
constexpr int kSetpointPeriod = 100;
constexpr float kDt = 0.004f;

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

template <typename T>
class DeviceArray {
public:
	explicit DeviceArray(size_t count) : count_(count) { CUDA_CALL(cudaMalloc(&data_, count * sizeof(T))); }
	~DeviceArray() { cudaFree(data_); }
	DeviceArray(const DeviceArray &) = delete;
	DeviceArray &operator=(const DeviceArray &) = delete;

	T *get() { return data_; }
	const T *get() const { return data_; }

	void upload(const std::vector<T> &host)
	{
		CUDA_CALL(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
	}

	void download(std::vector<T> &host) const
	{
		host.resize(count_);
		CUDA_CALL(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
	}

private:
	size_t count_;
	T *data_{nullptr};
};

struct Rng {
	uint32_t state;

	float uniform()
	{
		state = state * 1664525u + 1013904223u;
		return static_cast<float>(state >> 8) * (1.f / 16777216.f);
	}

	float range(float lo, float hi) { return lo + (hi - lo) * uniform(); }
};

struct Fields {
	std::vector<float> rates_sp, thrust_body, motors, actuator, torque_sp;

	explicit Fields(int n)
		: rates_sp(3 * n), thrust_body(3 * n), motors(4 * n), actuator(4 * n), torque_sp(3 * n)
	{}
};

enum FieldIndex { kRatesSp, kThrustBody, kMotors, kActuator, kTorqueSp, kFieldCount };
const char *const kFieldNames[kFieldCount] = {"rates_sp", "thrust_body", "motors", "actuator", "torque_sp"};
const int kFieldWidths[kFieldCount] = {3, 3, 4, 4, 3};

const std::vector<float> &fieldOf(const Fields &f, int index)
{
	switch (index) {
	case kRatesSp: return f.rates_sp;

	case kThrustBody: return f.thrust_body;

	case kMotors: return f.motors;

	case kActuator: return f.actuator;

	default: return f.torque_sp;
	}
}

struct FieldStats {
	double max_abs{0.0};
	double max_rel{0.0}; // |d| / max(1, |cpu|)
	size_t count{0};
	size_t within_1e6{0};
	bool all_finite{true};
};

using PhaseStats = FieldStats[kFieldCount];

// Accumulates |cpu - gpu| over the vehicles selected by `subset` (all vehicles if null).
void accumulate(PhaseStats stats, const Fields &cpu, const Fields &gpu, const std::vector<uint8_t> *subset)
{
	for (int field = 0; field < kFieldCount; ++field) {
		const std::vector<float> &c = fieldOf(cpu, field);
		const std::vector<float> &g = fieldOf(gpu, field);
		const int width = kFieldWidths[field];
		FieldStats &s = stats[field];

		for (int i = 0; i < kVehicles; ++i) {
			if (subset != nullptr && (*subset)[i] == 0) {
				continue;
			}

			for (int k = 0; k < width; ++k) {
				const float cv = c[width * i + k];
				const float gv = g[width * i + k];

				if (!std::isfinite(cv) || !std::isfinite(gv)) {
					s.all_finite = false;
					continue;
				}

				const double d = std::fabs(static_cast<double>(cv) - static_cast<double>(gv));
				s.max_abs = std::max(s.max_abs, d);
				s.max_rel = std::max(s.max_rel, d / std::max(1.0, std::fabs(static_cast<double>(cv))));
				++s.count;
				s.within_1e6 += d <= 1e-6 ? 1 : 0;
			}
		}
	}
}

void mergeInto(PhaseStats total, const PhaseStats part)
{
	for (int field = 0; field < kFieldCount; ++field) {
		total[field].max_abs = std::max(total[field].max_abs, part[field].max_abs);
		total[field].max_rel = std::max(total[field].max_rel, part[field].max_rel);
		total[field].count += part[field].count;
		total[field].within_1e6 += part[field].within_1e6;
		total[field].all_finite = total[field].all_finite && part[field].all_finite;
	}
}

void printPhase(const char *title, const PhaseStats stats)
{
	std::printf("%s\n  %-12s %12s %12s %14s\n", title, "field", "max|d|", "max rel", "frac |d|<=1e-6");

	for (int field = 0; field < kFieldCount; ++field) {
		const FieldStats &s = stats[field];
		std::printf("  %-12s %12.3e %12.3e %14.4f\n", kFieldNames[field], s.max_abs, s.max_rel,
			    s.count > 0 ? static_cast<double>(s.within_1e6) / static_cast<double>(s.count) : 0.0);
	}
}

// Single-step criteria for two fleets that held identical state before the step.
void expectSingleStepAgreement(const char *phase, const PhaseStats stats)
{
	EXPECT_LE(stats[kMotors].max_abs, 1e-3) << phase << " motors";
	EXPECT_LE(stats[kTorqueSp].max_abs, 1e-3) << phase << " torque_sp";
	EXPECT_LE(stats[kRatesSp].max_rel, 1e-3) << phase << " rates_sp";
	EXPECT_LE(stats[kActuator].max_abs, 1.0) << phase << " actuator";
	EXPECT_LE(stats[kThrustBody].max_abs, 1e-6) << phase << " thrust_body";

	for (int field = 0; field < kFieldCount; ++field) {
		const FieldStats &s = stats[field];
		ASSERT_GT(s.count, 0u) << phase << " " << kFieldNames[field];
		EXPECT_GE(static_cast<double>(s.within_1e6) / static_cast<double>(s.count), 0.95)
				<< phase << " " << kFieldNames[field] << ": fewer than 95% of values agree to 1e-6";
	}
}

} // namespace

TEST(Parity, CpuFleetVsCudaFleet)
{
	// Per-vehicle parameters: rate_k in [0.8, 1.2], airmode i % 3, thr_mdl_fac 0.3.
	std::vector<px4rl_params> params(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		Rng rng{static_cast<uint32_t>(i) * 2654435761u + 12345u};
		const float k = rng.range(0.8f, 1.2f);
		px4rl_params_default(&params[i]);
		params[i].rate_k[0] = params[i].rate_k[1] = params[i].rate_k[2] = k;
		params[i].airmode = i % 3;
		params[i].thr_mdl_fac = 0.3f;
	}

	std::vector<uint8_t> reset_mask(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		reset_mask[i] = (i % 7 == 0) ? 1 : 0;
	}

	px4rl::CpuFleet cpu(kVehicles);
	cpu.reset(nullptr, params.data(), kVehicles);

	px4rl_cuda_fleet *gpu = nullptr;
	API_CALL(px4rl_cuda_fleet_create(kVehicles, 0, &gpu));
	API_CALL(px4rl_cuda_fleet_reset(gpu, nullptr, params.data(), kVehicles, nullptr));

	DeviceArray<float> d_q(4 * kVehicles), d_q_sp(4 * kVehicles), d_thrust_sp(3 * kVehicles);
	DeviceArray<float> d_rates(3 * kVehicles), d_accel(3 * kVehicles);
	DeviceArray<float> d_rates_sp(3 * kVehicles), d_thrust_body(3 * kVehicles);
	DeviceArray<float> d_motors(4 * kVehicles), d_actuator(4 * kVehicles), d_torque(3 * kVehicles);
	DeviceArray<uint8_t> d_reset_mask(kVehicles);
	d_reset_mask.upload(reset_mask);

	std::vector<float> q(4 * kVehicles), q_sp(4 * kVehicles), thrust_sp(3 * kVehicles);
	std::vector<float> rates(3 * kVehicles), rates_prev(3 * kVehicles), accel(3 * kVehicles);
	Fields cpu_out(kVehicles), gpu_out(kVehicles);

	Rng rng{20261021u};

	for (float &r : rates) {
		r = rng.range(-3.f, 3.f);
	}

	PhaseStats step0{}, after_reset{}, all_steps{};

	for (int t = 0; t < kSteps; ++t) {
		if (t == kResetStep) {
			cpu.reset(reset_mask.data(), params.data(), kVehicles);
			API_CALL(px4rl_cuda_fleet_reset(gpu, d_reset_mask.get(), params.data(), kVehicles, nullptr));
		}

		// Draw into named locals: argument evaluation order is compiler dependent.
		if (t % kSetpointPeriod == 0) {
			for (int i = 0; i < kVehicles; ++i) {
				const float roll = rng.range(-0.8f, 0.8f);
				const float pitch = rng.range(-0.8f, 0.8f);
				const float yaw = rng.range(-3.1f, 3.1f);
				const matrix::Quatf sp(matrix::Eulerf(roll, pitch, yaw));
				const float thrust = rng.range(0.3f, 0.7f);

				for (int k = 0; k < 4; ++k) {
					q_sp[4 * i + k] = sp(k);
				}

				thrust_sp[3 * i + 0] = 0.f;
				thrust_sp[3 * i + 1] = 0.f;
				thrust_sp[3 * i + 2] = -thrust;
			}
		}

		rates_prev = rates;

		for (int i = 0; i < kVehicles; ++i) {
			const float roll = rng.range(-0.6f, 0.6f);
			const float pitch = rng.range(-0.6f, 0.6f);
			const float yaw = rng.range(-3.f, 3.f);
			const matrix::Quatf att(matrix::Eulerf(roll, pitch, yaw));

			for (int k = 0; k < 4; ++k) {
				q[4 * i + k] = att(k);
			}

			const float wx = rng.range(-3.f, 3.f);
			const float wy = rng.range(-3.f, 3.f);
			const float wz = rng.range(-3.f, 3.f);
			rates[3 * i + 0] = wx;
			rates[3 * i + 1] = wy;
			rates[3 * i + 2] = wz;

			for (int a = 0; a < 3; ++a) {
				accel[3 * i + a] = (rates[3 * i + a] - rates_prev[3 * i + a]) / kDt;
			}
		}

		cpu.attitudeStep(q.data(), q_sp.data(), nullptr, thrust_sp.data(), cpu_out.rates_sp.data(),
				 cpu_out.thrust_body.data());
		cpu.rateStep(cpu_out.rates_sp.data(), cpu_out.thrust_body.data(), rates.data(), accel.data(), kDt, nullptr,
			     cpu_out.motors.data(), cpu_out.actuator.data(), cpu_out.torque_sp.data());

		d_q.upload(q);
		d_q_sp.upload(q_sp);
		d_thrust_sp.upload(thrust_sp);
		d_rates.upload(rates);
		d_accel.upload(accel);
		API_CALL(px4rl_cuda_attitude_step(gpu, d_q.get(), d_q_sp.get(), nullptr, d_thrust_sp.get(),
						  d_rates_sp.get(), d_thrust_body.get(), nullptr));
		API_CALL(px4rl_cuda_rate_step(gpu, d_rates_sp.get(), d_thrust_body.get(), d_rates.get(), d_accel.get(), kDt,
					      nullptr, d_motors.get(), d_actuator.get(), d_torque.get(), nullptr));
		d_rates_sp.download(gpu_out.rates_sp);
		d_thrust_body.download(gpu_out.thrust_body);
		d_motors.download(gpu_out.motors);
		d_actuator.download(gpu_out.actuator);
		d_torque.download(gpu_out.torque_sp);

		PhaseStats this_step{};
		accumulate(this_step, cpu_out, gpu_out, nullptr);
		mergeInto(all_steps, this_step);

		if (t == 0) {
			mergeInto(step0, this_step);
		}

		if (t == kResetStep) {
			accumulate(after_reset, cpu_out, gpu_out, &reset_mask);
		}
	}

	API_CALL(px4rl_cuda_fleet_destroy(gpu));

	std::printf("\nparity_test: %d vehicles, %d steps, open loop, masked reset (i %% 7 == 0) at step %d\n",
		    kVehicles, kSteps, kResetStep);
	printPhase("step 0 (all vehicles, identical state):", step0);
	printPhase("first step after masked reset (reset vehicles only, identical state):", after_reset);
	printPhase("all 600 steps (all vehicles):", all_steps);

	for (int field = 0; field < kFieldCount; ++field) {
		EXPECT_TRUE(all_steps[field].all_finite) << kFieldNames[field] << " has non-finite values";
	}

	expectSingleStepAgreement("step 0", step0);
	expectSingleStepAgreement("first step after masked reset", after_reset);
	EXPECT_LE(all_steps[kMotors].max_abs, 0.1) << "motors over 600 steps";
}
