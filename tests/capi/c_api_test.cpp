// Links only libpx4rl_cuda (no CpuFleet), so px4rl_params_default also checks the export.
// Error paths, non-default streams, coexisting fleets and graph capture of the CUDA C API.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "px4rl/c_api.h"

namespace {

constexpr float kDt = 0.004f;

#define CUDA_CALL(expr)                                                                                      \
	do {                                                                                                 \
		const cudaError_t cuda_call_error = (expr);                                                  \
		if (cuda_call_error != cudaSuccess) {                                                        \
			std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_call_error)); \
			std::abort();                                                                        \
		}                                                                                            \
	} while (0)

template <typename T>
class DeviceArray {
public:
	explicit DeviceArray(size_t count) : count_(count) { CUDA_CALL(cudaMalloc(&data_, count * sizeof(T))); }
	~DeviceArray() { cudaFree(data_); }
	DeviceArray(const DeviceArray &) = delete;
	DeviceArray &operator=(const DeviceArray &) = delete;

	T *get() { return data_; }

	void upload(const std::vector<T> &host)
	{
		CUDA_CALL(cudaMemcpy(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
	}

	std::vector<T> download() const
	{
		std::vector<T> host(count_);
		CUDA_CALL(cudaMemcpy(host.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
		return host;
	}

private:
	size_t count_;
	T *data_{nullptr};
};

// Identity attitude, a fixed setpoint, small rates: enough for non-trivial, finite outputs.
struct Inputs {
	int n;
	DeviceArray<float> q, q_sp, thrust_sp, rates, accel;
	DeviceArray<float> rates_sp, thrust_body, motors, actuator, torque;

	explicit Inputs(int count)
		: n(count), q(4 * count), q_sp(4 * count), thrust_sp(3 * count), rates(3 * count), accel(3 * count),
		  rates_sp(3 * count), thrust_body(3 * count), motors(4 * count), actuator(4 * count),
		  torque(3 * count)
	{
		std::vector<float> hq(4 * n), hq_sp(4 * n), ht(3 * n), hr(3 * n), ha(3 * n, 0.f);

		for (int i = 0; i < n; ++i) {
			hq[4 * i + 0] = 1.f;
			hq[4 * i + 1] = hq[4 * i + 2] = hq[4 * i + 3] = 0.f;
			// 0.3 rad about x (w, x, y, z).
			hq_sp[4 * i + 0] = std::cos(0.15f);
			hq_sp[4 * i + 1] = std::sin(0.15f);
			hq_sp[4 * i + 2] = hq_sp[4 * i + 3] = 0.f;
			ht[3 * i + 0] = ht[3 * i + 1] = 0.f;
			ht[3 * i + 2] = -0.5f;
			hr[3 * i + 0] = 0.1f * static_cast<float>(i % 5);
			hr[3 * i + 1] = -0.2f;
			hr[3 * i + 2] = 0.05f;
		}

		q.upload(hq);
		q_sp.upload(hq_sp);
		thrust_sp.upload(ht);
		rates.upload(hr);
		accel.upload(ha);
	}
};

int attitudeStep(px4rl_cuda_fleet *f, Inputs &in, void *stream)
{
	return px4rl_cuda_attitude_step(f, in.q.get(), in.q_sp.get(), nullptr, in.thrust_sp.get(), in.rates_sp.get(),
					in.thrust_body.get(), stream);
}

int rateStep(px4rl_cuda_fleet *f, Inputs &in, void *stream)
{
	return px4rl_cuda_rate_step(f, in.rates_sp.get(), in.thrust_body.get(), in.rates.get(), in.accel.get(), kDt,
				    nullptr, in.motors.get(), in.actuator.get(), in.torque.get(), stream);
}

px4rl_cuda_fleet *createFleet(int n)
{
	px4rl_cuda_fleet *f = nullptr;
	EXPECT_EQ(px4rl_cuda_fleet_create(n, 0, &f), 0) << px4rl_last_error();
	return f;
}

bool allFinite(const std::vector<float> &v)
{
	for (float x : v) {
		if (!std::isfinite(x)) {
			return false;
		}
	}

	return true;
}

} // namespace

TEST(CApi, ErrorPaths)
{
	px4rl_cuda_fleet *f = nullptr;
	px4rl_params params[2];
	px4rl_params_default(&params[0]);
	px4rl_params_default(&params[1]);

	EXPECT_NE(px4rl_cuda_fleet_create(8, 0, nullptr), 0);
	EXPECT_STRNE(px4rl_last_error(), "");
	EXPECT_NE(px4rl_cuda_fleet_create(0, 0, &f), 0);
	EXPECT_EQ(f, nullptr);
	EXPECT_NE(px4rl_cuda_fleet_create(-3, 0, &f), 0);
	EXPECT_NE(px4rl_cuda_fleet_create(8, -1, &f), 0);
	EXPECT_EQ(f, nullptr);
	EXPECT_STRNE(px4rl_last_error(), "");
	EXPECT_NE(px4rl_cuda_fleet_create(8, 1 << 20, &f), 0);
	EXPECT_EQ(f, nullptr);

	// Null fleet on every entry point.
	Inputs in(8);
	EXPECT_NE(px4rl_cuda_fleet_destroy(nullptr), 0);
	EXPECT_EQ(px4rl_cuda_fleet_size(nullptr), -1);
	EXPECT_NE(px4rl_cuda_fleet_reset(nullptr, nullptr, params, 1, nullptr), 0);
	EXPECT_NE(attitudeStep(nullptr, in, nullptr), 0);
	EXPECT_STRNE(px4rl_last_error(), "");
	EXPECT_NE(rateStep(nullptr, in, nullptr), 0);

	f = createFleet(8);
	ASSERT_NE(f, nullptr);
	EXPECT_EQ(px4rl_cuda_fleet_size(f), 8);

	// n_params must be 1 or n.
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, params, 2, nullptr), 0);
	EXPECT_STRNE(px4rl_last_error(), "");
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, params, 0, nullptr), 0);
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, params, 9, nullptr), 0);
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, nullptr, 1, nullptr), 0);

	// Parameters that cannot be configured.
	px4rl_params bad = params[0];
	bad.num_rotors = 0;
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, &bad, 1, nullptr), 0);
	bad.num_rotors = PX4RL_MAX_ROTORS + 1;
	EXPECT_NE(px4rl_cuda_fleet_reset(f, nullptr, &bad, 1, nullptr), 0);

	// Required buffers.
	EXPECT_NE(px4rl_cuda_attitude_step(f, nullptr, in.q_sp.get(), nullptr, in.thrust_sp.get(), in.rates_sp.get(),
					   in.thrust_body.get(), nullptr), 0);
	EXPECT_NE(px4rl_cuda_rate_step(f, in.rates_sp.get(), in.thrust_body.get(), in.rates.get(), in.accel.get(), kDt,
				       nullptr, nullptr, nullptr, nullptr, nullptr), 0);

	// The fleet is still usable after rejected calls.
	EXPECT_EQ(px4rl_cuda_fleet_reset(f, nullptr, params, 1, nullptr), 0) << px4rl_last_error();
	EXPECT_EQ(attitudeStep(f, in, nullptr), 0) << px4rl_last_error();
	EXPECT_EQ(rateStep(f, in, nullptr), 0) << px4rl_last_error();
	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_TRUE(allFinite(in.motors.download()));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, OptionalOutputsMayBeNull)
{
	px4rl_cuda_fleet *f = createFleet(16);
	ASSERT_NE(f, nullptr);
	Inputs in(16);
	ASSERT_EQ(attitudeStep(f, in, nullptr), 0);
	EXPECT_EQ(px4rl_cuda_rate_step(f, in.rates_sp.get(), in.thrust_body.get(), in.rates.get(), in.accel.get(), kDt,
				       nullptr, in.motors.get(), nullptr, nullptr, nullptr), 0) << px4rl_last_error();
	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_TRUE(allFinite(in.motors.download()));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, NonDefaultStreamMatchesDefaultStream)
{
	constexpr int n = 256;
	px4rl_params params;
	px4rl_params_default(&params);
	params.airmode = 2;

	const auto run = [&](void *stream, std::vector<float> &motors, std::vector<float> &torque) {
		px4rl_cuda_fleet *f = createFleet(n);
		ASSERT_NE(f, nullptr);
		Inputs in(n);
		CUDA_CALL(cudaDeviceSynchronize());
		ASSERT_EQ(px4rl_cuda_fleet_reset(f, nullptr, &params, 1, stream), 0) << px4rl_last_error();

		for (int step = 0; step < 20; ++step) {
			ASSERT_EQ(attitudeStep(f, in, stream), 0) << px4rl_last_error();
			ASSERT_EQ(rateStep(f, in, stream), 0) << px4rl_last_error();
		}

		CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
		motors = in.motors.download();
		torque = in.torque.download();
		ASSERT_EQ(px4rl_cuda_fleet_destroy(f), 0);
	};

	std::vector<float> motors_default, torque_default, motors_stream, torque_stream;
	run(nullptr, motors_default, torque_default);

	cudaStream_t stream = nullptr;
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	run(stream, motors_stream, torque_stream);
	CUDA_CALL(cudaStreamDestroy(stream));

	ASSERT_TRUE(allFinite(motors_stream));
	ASSERT_EQ(motors_default.size(), motors_stream.size());
	EXPECT_EQ(std::memcmp(motors_default.data(), motors_stream.data(), motors_default.size() * sizeof(float)), 0);
	EXPECT_EQ(std::memcmp(torque_default.data(), torque_stream.data(), torque_default.size() * sizeof(float)), 0);
}

TEST(CApi, TwoFleetsCoexistIndependently)
{
	px4rl_cuda_fleet *a = createFleet(64);
	px4rl_cuda_fleet *b = createFleet(128);
	px4rl_cuda_fleet *reference = createFleet(64);
	ASSERT_TRUE(a && b && reference);
	EXPECT_EQ(px4rl_cuda_fleet_size(a), 64);
	EXPECT_EQ(px4rl_cuda_fleet_size(b), 128);

	Inputs in_a(64), in_b(128), in_ref(64);

	// Reconfigure b between steps of a; a must keep tracking the untouched reference fleet.
	px4rl_params other;
	px4rl_params_default(&other);
	other.airmode = 0;
	other.rate_k[0] = other.rate_k[1] = other.rate_k[2] = 1.5f;

	for (int step = 0; step < 30; ++step) {
		if (step == 10) {
			ASSERT_EQ(px4rl_cuda_fleet_reset(b, nullptr, &other, 1, nullptr), 0) << px4rl_last_error();
		}

		ASSERT_EQ(attitudeStep(a, in_a, nullptr), 0);
		ASSERT_EQ(rateStep(a, in_a, nullptr), 0);
		ASSERT_EQ(attitudeStep(b, in_b, nullptr), 0);
		ASSERT_EQ(rateStep(b, in_b, nullptr), 0);
		ASSERT_EQ(attitudeStep(reference, in_ref, nullptr), 0);
		ASSERT_EQ(rateStep(reference, in_ref, nullptr), 0);
	}

	CUDA_CALL(cudaDeviceSynchronize());
	const std::vector<float> motors_a = in_a.motors.download();
	const std::vector<float> motors_ref = in_ref.motors.download();
	EXPECT_EQ(std::memcmp(motors_a.data(), motors_ref.data(), motors_a.size() * sizeof(float)), 0);
	EXPECT_TRUE(allFinite(in_b.motors.download()));

	EXPECT_EQ(px4rl_cuda_fleet_destroy(a), 0);
	EXPECT_EQ(px4rl_cuda_fleet_destroy(b), 0);
	EXPECT_EQ(px4rl_cuda_fleet_destroy(reference), 0);
}

TEST(CApi, StepCallsAreGraphCapturable)
{
	constexpr int n = 512;
	px4rl_cuda_fleet *f = createFleet(n);
	ASSERT_NE(f, nullptr);
	Inputs in(n);

	cudaStream_t stream = nullptr;
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// Warm up so lazy module loading happens outside the capture.
	ASSERT_EQ(attitudeStep(f, in, stream), 0);
	ASSERT_EQ(rateStep(f, in, stream), 0);
	CUDA_CALL(cudaStreamSynchronize(stream));

	// Any allocation or synchronisation inside a step call would invalidate the capture.
	CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	EXPECT_EQ(attitudeStep(f, in, stream), 0) << px4rl_last_error();
	EXPECT_EQ(rateStep(f, in, stream), 0) << px4rl_last_error();
	cudaGraph_t graph = nullptr;
	CUDA_CALL(cudaStreamEndCapture(stream, &graph));

	cudaGraphExec_t exec = nullptr;
	CUDA_CALL(cudaGraphInstantiate(&exec, graph, 0));

	for (int i = 0; i < 5; ++i) {
		CUDA_CALL(cudaGraphLaunch(exec, stream));
	}

	CUDA_CALL(cudaStreamSynchronize(stream));
	EXPECT_TRUE(allFinite(in.motors.download()));

	CUDA_CALL(cudaGraphExecDestroy(exec));
	CUDA_CALL(cudaGraphDestroy(graph));
	CUDA_CALL(cudaStreamDestroy(stream));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}
