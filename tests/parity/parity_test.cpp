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

// Accumulates |cpu - gpu| of one field with `width` values per vehicle over the vehicles selected by
// `subset` (all vehicles if null).
void accumulateField(FieldStats &s, const std::vector<float> &c, const std::vector<float> &g, int width,
		     const std::vector<uint8_t> *subset)
{
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

void accumulate(PhaseStats stats, const Fields &cpu, const Fields &gpu, const std::vector<uint8_t> *subset)
{
	for (int field = 0; field < kFieldCount; ++field) {
		accumulateField(stats[field], fieldOf(cpu, field), fieldOf(gpu, field), kFieldWidths[field], subset);
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

// One compared output of the position chain, with CPU and GPU values for the current step.
struct ChainField {
	const char *name;
	int width;
	bool compare_relative; // single-step criterion on |d| / max(1, |v|) instead of |d|
	double single_step_limit;
	std::vector<float> cpu, gpu;
	FieldStats step0, after_reset, all_steps;
};

void mergeStats(FieldStats &total, const FieldStats &part)
{
	total.max_abs = std::max(total.max_abs, part.max_abs);
	total.max_rel = std::max(total.max_rel, part.max_rel);
	total.count += part.count;
	total.within_1e6 += part.within_1e6;
	total.all_finite = total.all_finite && part.all_finite;
}

void printChainStats(const char *title, const std::vector<ChainField> &fields, FieldStats ChainField::*phase)
{
	std::printf("%s\n  %-14s %12s %12s %14s\n", title, "field", "max|d|", "max rel", "frac |d|<=1e-6");

	for (const ChainField &f : fields) {
		const FieldStats &s = f.*phase;
		std::printf("  %-14s %12.3e %12.3e %14.4f\n", f.name, s.max_abs, s.max_rel,
			    s.count > 0 ? static_cast<double>(s.within_1e6) / static_cast<double>(s.count) : 0.0);
	}
}

void expectChainSingleStep(const char *phase, const std::vector<ChainField> &fields, FieldStats ChainField::*which)
{
	for (const ChainField &f : fields) {
		const FieldStats &s = f.*which;
		ASSERT_GT(s.count, 0u) << phase << " " << f.name;
		EXPECT_LE(f.compare_relative ? s.max_rel : s.max_abs, f.single_step_limit) << phase << " " << f.name;
		EXPECT_GE(static_cast<double>(s.within_1e6) / static_cast<double>(s.count), 0.95)
				<< phase << " " << f.name << ": fewer than 95% of values agree to 1e-6";
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

// Position loop, hover thrust estimator, stored-setpoint attitude step and rate step, fed with the same open-loop
// inputs. Each fleet feeds its own position output into its own attitude step.
TEST(Parity, PositionChainCpuFleetVsCudaFleet)
{
	constexpr float kPositionDt = 0.02f;
	constexpr uint64_t kPositionStepUs = 20000;

	std::vector<px4rl_params> params(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		Rng rng{static_cast<uint32_t>(i) * 2654435761u + 54321u};
		px4rl_params_default(&params[i]);
		const float k = rng.range(0.8f, 1.2f);
		params[i].rate_k[0] = params[i].rate_k[1] = params[i].rate_k[2] = k;
		params[i].airmode = i % 3;
		params[i].thr_mdl_fac = 0.3f;
		params[i].mpc_thr_hover = rng.range(0.35f, 0.6f);
		params[i].mpc_use_hte = (i % 4 != 3) ? 1 : 0;
		params[i].mpc_vel_lp = (i % 2 == 1) ? 4.f : 0.f;
		params[i].mpc_vel_nf_frq = (i % 3 == 0) ? 8.f : 0.f;
		params[i].mpc_acc_decouple = i % 2;
	}

	std::vector<uint8_t> reset_mask(kVehicles), in_air(kVehicles);

	for (int i = 0; i < kVehicles; ++i) {
		reset_mask[i] = (i % 7 == 0) ? 1 : 0;
		in_air[i] = (i % 5 != 0) ? 1 : 0;
	}

	px4rl::CpuFleet cpu(kVehicles);
	cpu.reset(nullptr, params.data(), kVehicles);

	px4rl_cuda_fleet *gpu = nullptr;
	API_CALL(px4rl_cuda_fleet_create(kVehicles, 0, &gpu));
	API_CALL(px4rl_cuda_fleet_reset(gpu, nullptr, params.data(), kVehicles, nullptr));

	DeviceArray<float> d_position(3 * kVehicles), d_velocity(3 * kVehicles), d_heading(kVehicles), d_az(kVehicles);
	DeviceArray<float> d_traj(11 * kVehicles), d_q(4 * kVehicles), d_rates(3 * kVehicles), d_accel(3 * kVehicles);
	DeviceArray<float> d_q_sp(4 * kVehicles), d_thrust_sp(3 * kVehicles), d_yawspeed(kVehicles), d_hover(kVehicles);
	DeviceArray<float> d_rates_sp(3 * kVehicles), d_thrust_body(3 * kVehicles), d_motors(4 * kVehicles);
	DeviceArray<uint8_t> d_outcome(kVehicles), d_in_air(kVehicles), d_reset_mask(kVehicles);
	d_in_air.upload(in_air);
	d_reset_mask.upload(reset_mask);

	std::vector<ChainField> fields = {
		{"q_sp", 4, false, 1e-3, {}, {}, {}, {}, {}},
		{"thrust_sp_body", 3, false, 1e-3, {}, {}, {}, {}, {}},
		{"yawspeed", 1, false, 1e-3, {}, {}, {}, {}, {}},
		{"hover_thrust", 1, false, 1e-3, {}, {}, {}, {}, {}},
		{"rates_sp", 3, true, 1e-3, {}, {}, {}, {}, {}},
		{"motors", 4, false, 1e-3, {}, {}, {}, {}, {}},
	};
	enum { kQSp, kThrust, kYawspeed, kHover, kRatesSp, kMotors };

	std::vector<float> position(3 * kVehicles), velocity(3 * kVehicles), heading(kVehicles), az(kVehicles);
	std::vector<float> traj(11 * kVehicles), q(4 * kVehicles), rates(3 * kVehicles), rates_prev(3 * kVehicles),
		accel(3 * kVehicles);
	std::vector<float> cpu_thrust_body(3 * kVehicles), cpu_actuator(4 * kVehicles), cpu_torque(3 * kVehicles);
	std::vector<uint8_t> cpu_outcome(kVehicles), gpu_outcome;
	size_t outcome_mismatches_step0 = 0, outcome_mismatches_after_reset = 0, outcome_mismatches_total = 0;
	size_t outcome_counts[3] = {0, 0, 0};

	Rng rng{20261022u};

	for (float &r : rates) {
		r = rng.range(-3.f, 3.f);
	}

	for (int t = 0; t < kSteps; ++t) {
		if (t == kResetStep) {
			cpu.reset(reset_mask.data(), params.data(), kVehicles);
			API_CALL(px4rl_cuda_fleet_reset(gpu, d_reset_mask.get(), params.data(), kVehicles, nullptr));
		}

		const uint64_t time_us = 1000000u + kPositionStepUs * static_cast<uint64_t>(t + 1);
		rates_prev = rates;

		for (int i = 0; i < kVehicles; ++i) {
			position[3 * i + 0] = rng.range(-5.f, 5.f);
			position[3 * i + 1] = rng.range(-5.f, 5.f);
			position[3 * i + 2] = rng.range(-10.f, -2.f);

			for (int a = 0; a < 3; ++a) {
				velocity[3 * i + a] = rng.range(-2.f, 2.f);
			}

			heading[i] = rng.range(-3.f, 3.f);
			az[i] = rng.range(-1.f, 1.f);

			float *sp = &traj[11 * i];
			std::fill(sp, sp + 11, NAN);
			sp[9] = rng.range(-3.f, 3.f);

			switch (i % 4) {
			case 0: // velocity + yaw, every third command invalid (fallback)
				if (t % 3 != 2) {
					for (int a = 0; a < 3; ++a) {
						sp[3 + a] = rng.range(-3.f, 3.f);
					}
				}

				break;

			case 1: // acceleration + yaw
				for (int a = 0; a < 3; ++a) {
					sp[6 + a] = rng.range(-2.f, 2.f);
				}

				break;

			case 2: // position + yaw
				for (int a = 0; a < 3; ++a) {
					sp[a] = position[3 * i + a] + rng.range(-2.f, 2.f);
				}

				break;

			default: // never valid (failsafe)
				sp[9] = NAN;
				break;
			}

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

		fields[kQSp].cpu.resize(4 * kVehicles);
		fields[kThrust].cpu.resize(3 * kVehicles);
		fields[kYawspeed].cpu.resize(kVehicles);
		fields[kHover].cpu.resize(kVehicles);
		fields[kRatesSp].cpu.resize(3 * kVehicles);
		fields[kMotors].cpu.resize(4 * kVehicles);

		cpu.positionStep(position.data(), velocity.data(), heading.data(), az.data(), traj.data(), in_air.data(),
				 kPositionDt, time_us, fields[kQSp].cpu.data(), fields[kThrust].cpu.data(),
				 fields[kYawspeed].cpu.data(), fields[kHover].cpu.data(), cpu_outcome.data());
		cpu.attitudeStep(q.data(), nullptr, nullptr, fields[kThrust].cpu.data(), fields[kRatesSp].cpu.data(),
				 cpu_thrust_body.data());
		cpu.rateStep(fields[kRatesSp].cpu.data(), cpu_thrust_body.data(), rates.data(), accel.data(), kDt, nullptr,
			     fields[kMotors].cpu.data(), cpu_actuator.data(), cpu_torque.data());

		d_position.upload(position);
		d_velocity.upload(velocity);
		d_heading.upload(heading);
		d_az.upload(az);
		d_traj.upload(traj);
		d_q.upload(q);
		d_rates.upload(rates);
		d_accel.upload(accel);
		API_CALL(px4rl_cuda_position_step(gpu, d_position.get(), d_velocity.get(), d_heading.get(), d_az.get(),
						  d_traj.get(), d_in_air.get(), kPositionDt, time_us, d_q_sp.get(),
						  d_thrust_sp.get(), d_yawspeed.get(), d_hover.get(), d_outcome.get(),
						  nullptr));
		API_CALL(px4rl_cuda_attitude_step(gpu, d_q.get(), nullptr, nullptr, d_thrust_sp.get(), d_rates_sp.get(),
						  d_thrust_body.get(), nullptr));
		API_CALL(px4rl_cuda_rate_step(gpu, d_rates_sp.get(), d_thrust_body.get(), d_rates.get(), d_accel.get(), kDt,
					      nullptr, d_motors.get(), nullptr, nullptr, nullptr));
		d_q_sp.download(fields[kQSp].gpu);
		d_thrust_sp.download(fields[kThrust].gpu);
		d_yawspeed.download(fields[kYawspeed].gpu);
		d_hover.download(fields[kHover].gpu);
		d_rates_sp.download(fields[kRatesSp].gpu);
		d_motors.download(fields[kMotors].gpu);
		d_outcome.download(gpu_outcome);

		size_t mismatches = 0, mismatches_reset_subset = 0;

		for (int i = 0; i < kVehicles; ++i) {
			++outcome_counts[cpu_outcome[i] <= 2 ? cpu_outcome[i] : 0];
			const bool differs = cpu_outcome[i] != gpu_outcome[i];
			mismatches += differs ? 1 : 0;
			mismatches_reset_subset += (differs && reset_mask[i]) ? 1 : 0;
		}

		outcome_mismatches_total += mismatches;

		if (t == 0) {
			outcome_mismatches_step0 = mismatches;
		}

		if (t == kResetStep) {
			outcome_mismatches_after_reset = mismatches_reset_subset;
		}

		for (ChainField &f : fields) {
			FieldStats this_step{};
			accumulateField(this_step, f.cpu, f.gpu, f.width, nullptr);
			mergeStats(f.all_steps, this_step);

			if (t == 0) {
				mergeStats(f.step0, this_step);
			}

			if (t == kResetStep) {
				accumulateField(f.after_reset, f.cpu, f.gpu, f.width, &reset_mask);
			}
		}
	}

	API_CALL(px4rl_cuda_fleet_destroy(gpu));

	std::printf("\nparity_test (position chain): %d vehicles, %d steps, open loop, masked reset (i %% 7 == 0) at step %d\n",
		    kVehicles, kSteps, kResetStep);
	std::printf("outcomes (cpu) ok/fallback/failsafe = %zu/%zu/%zu; cpu vs gpu outcome mismatches: step 0 = %zu, "
		    "after reset = %zu, all steps = %zu of %zu\n",
		    outcome_counts[0], outcome_counts[1], outcome_counts[2], outcome_mismatches_step0,
		    outcome_mismatches_after_reset, outcome_mismatches_total,
		    static_cast<size_t>(kVehicles) * static_cast<size_t>(kSteps));
	printChainStats("step 0 (all vehicles, identical state):", fields, &ChainField::step0);
	printChainStats("first step after masked reset (reset vehicles only, identical state):", fields,
			&ChainField::after_reset);
	printChainStats("all steps (all vehicles):", fields, &ChainField::all_steps);

	for (const ChainField &f : fields) {
		EXPECT_TRUE(f.all_steps.all_finite) << f.name << " has non-finite values";
	}

	// Only single-step agreement is bounded; the divergence over the whole run is printed above.

	EXPECT_EQ(outcome_mismatches_step0, 0u);
	EXPECT_EQ(outcome_mismatches_after_reset, 0u);
	expectChainSingleStep("step 0", fields, &ChainField::step0);
	expectChainSingleStep("first step after masked reset", fields, &ChainField::after_reset);
	EXPECT_GT(outcome_counts[1], 0u);
	EXPECT_GT(outcome_counts[2], 0u);
}
