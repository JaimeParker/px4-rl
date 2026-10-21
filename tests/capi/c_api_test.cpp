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

#include "px4rl/airframes.h"
#include "px4rl/c_api.h"
#include "px4rl/plant.h"

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

// Hover at 5 m with a velocity command, so the position step produces a non-trivial setpoint.
struct PositionInputs {
	int n;
	DeviceArray<float> position, velocity, heading, az, trajectory_sp;
	DeviceArray<float> q_sp, thrust_sp, yawspeed, hover_thrust;
	DeviceArray<uint8_t> outcome;

	explicit PositionInputs(int count)
		: n(count), position(3 * count), velocity(3 * count), heading(count), az(count), trajectory_sp(11 * count),
		  q_sp(4 * count), thrust_sp(3 * count), yawspeed(count), hover_thrust(count), outcome(count)
	{
		std::vector<float> hp(3 * n, 0.f), hv(3 * n, 0.f), hh(n, 0.3f), ha(n, 0.f), hs(11 * n, NAN);

		for (int i = 0; i < n; ++i) {
			hp[3 * i + 2] = -5.f;
			hs[11 * i + 3] = 1.f + 0.1f * static_cast<float>(i % 7);
			hs[11 * i + 4] = hs[11 * i + 5] = 0.f;
			hs[11 * i + 9] = 0.5f;
		}

		position.upload(hp);
		velocity.upload(hv);
		heading.upload(hh);
		az.upload(ha);
		trajectory_sp.upload(hs);
	}
};

int positionStep(px4rl_cuda_fleet *f, PositionInputs &in, uint64_t time_us, void *stream, bool optional_outputs = true)
{
	return px4rl_cuda_position_step(f, in.position.get(), in.velocity.get(), in.heading.get(), in.az.get(),
					in.trajectory_sp.get(), nullptr, 0.02f, time_us, in.q_sp.get(), in.thrust_sp.get(),
					in.yawspeed.get(), optional_outputs ? in.hover_thrust.get() : nullptr,
					optional_outputs ? in.outcome.get() : nullptr, stream);
}

// Rotor step inputs at the Iris hover command (rounded PWM), body at rest.
struct RotorInputs {
	int n;
	DeviceArray<float> actuator, velocity, rates, force, torque, rotor_force;
	DeviceArray<uint8_t> armed;

	explicit RotorInputs(int count)
		: n(count), actuator(4 * count), velocity(3 * count), rates(3 * count), force(3 * count), torque(3 * count),
		  rotor_force(12 * count), armed(count)
	{
		actuator.upload(std::vector<float>(4 * n, 1694.f));
		velocity.upload(std::vector<float>(3 * n, 0.f));
		rates.upload(std::vector<float>(3 * n, 0.f));
		armed.upload(std::vector<uint8_t>(n, 1));
	}
};

int rotorStep(px4rl_cuda_fleet *f, RotorInputs &in, void *stream)
{
	return px4rl_cuda_rotor_step(f, in.actuator.get(), in.armed.get(), in.velocity.get(), in.rates.get(), kDt,
				     in.force.get(), in.torque.get(), in.rotor_force.get(), stream);
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

	// Position step: null fleet, required buffers.
	PositionInputs position_in(8);
	EXPECT_NE(positionStep(nullptr, position_in, 1000000u, nullptr), 0);
	EXPECT_STRNE(px4rl_last_error(), "");

	// Required buffers.
	EXPECT_NE(px4rl_cuda_attitude_step(f, nullptr, in.q_sp.get(), nullptr, in.thrust_sp.get(), in.rates_sp.get(),
					   in.thrust_body.get(), nullptr), 0);
	EXPECT_NE(px4rl_cuda_rate_step(f, in.rates_sp.get(), in.thrust_body.get(), in.rates.get(), in.accel.get(), kDt,
				       nullptr, nullptr, nullptr, nullptr, nullptr), 0);

	EXPECT_NE(px4rl_cuda_position_step(f, nullptr, position_in.velocity.get(), position_in.heading.get(),
					   position_in.az.get(), position_in.trajectory_sp.get(), nullptr, 0.02f, 1000000u,
					   position_in.q_sp.get(), position_in.thrust_sp.get(), position_in.yawspeed.get(),
					   nullptr, nullptr, nullptr), 0);
	EXPECT_NE(px4rl_cuda_position_step(f, position_in.position.get(), position_in.velocity.get(),
					   position_in.heading.get(), position_in.az.get(), position_in.trajectory_sp.get(),
					   nullptr, 0.02f, 1000000u, position_in.q_sp.get(), position_in.thrust_sp.get(), nullptr,
					   nullptr, nullptr, nullptr), 0);
	EXPECT_NE(px4rl_cuda_attitude_step(f, in.q.get(), in.q_sp.get(), nullptr, nullptr, in.rates_sp.get(),
					   in.thrust_body.get(), nullptr), 0);

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

TEST(CApi, IrisFunctionsAreExportedByTheCudaLibrary)
{
	px4rl_params params;
	px4rl_params_iris(&params);
	EXPECT_EQ(params.num_rotors, 4);
	EXPECT_EQ(params.rotor_ct[0], 6.5f);
	EXPECT_EQ(params.rotor_position[0][0], 0.1515f);
	EXPECT_EQ(params.rotor_position[1][1], -0.1875f);
	EXPECT_EQ(params.rotor_km[2], -0.05f);

	const px4rl_plant_params plant = px4rl_plant_params_iris();
	EXPECT_EQ(plant.num_rotors, 4);
	EXPECT_EQ(plant.rotor[0].motor_constant, 5.84e-06f);
	EXPECT_EQ(px4rl_rigid_body_params_iris().mass, 1.5f);
}

TEST(CApi, RotorStepOfTheDefaultIrisPlantBalancesWeightAtHoverCommand)
{
	px4rl_cuda_fleet *f = createFleet(16);
	ASSERT_NE(f, nullptr);
	RotorInputs in(16);

	for (int t = 0; t < 250; ++t) {
		ASSERT_EQ(rotorStep(f, in, nullptr), 0) << px4rl_last_error();
	}

	CUDA_CALL(cudaDeviceSynchronize());
	const std::vector<float> force = in.force.download();
	const std::vector<float> rotor_force = in.rotor_force.download();
	const float weight = px4rl_rigid_body_params_iris().mass * 9.80665f;

	for (int i = 0; i < 16; ++i) {
		EXPECT_NEAR(force[3 * i + 2], weight, 0.01f * weight);
		EXPECT_NEAR(rotor_force[12 * i + 2], weight / 4.f, 0.01f * weight / 4.f);
	}

	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, RotorStepOptionalInputsMayBeNullAndPlantResetRestsTheRotors)
{
	px4rl_cuda_fleet *f = createFleet(16);
	ASSERT_NE(f, nullptr);
	RotorInputs in(16);

	for (int t = 0; t < 100; ++t) {
		ASSERT_EQ(px4rl_cuda_rotor_step(f, in.actuator.get(), nullptr, in.velocity.get(), nullptr, kDt,
						in.force.get(), in.torque.get(), nullptr, nullptr), 0) << px4rl_last_error();
	}

	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_GT(in.force.download()[2], 1.f);

	// A controller reset keeps the rotors spinning, a plant reset stops them.
	px4rl_params params;
	px4rl_params_iris(&params);
	ASSERT_EQ(px4rl_cuda_fleet_reset(f, nullptr, &params, 1, nullptr), 0) << px4rl_last_error();
	ASSERT_EQ(rotorStep(f, in, nullptr), 0);
	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_GT(in.force.download()[2], 1.f);

	const px4rl_plant_params iris = px4rl_plant_params_iris();
	ASSERT_EQ(px4rl_cuda_fleet_reset_plant(f, nullptr, &iris, 1, nullptr), 0) << px4rl_last_error();
	ASSERT_EQ(rotorStep(f, in, nullptr), 0);
	CUDA_CALL(cudaDeviceSynchronize());
	const std::vector<float> force = in.force.download();

	for (int i = 0; i < 16; ++i) {
		EXPECT_EQ(force[3 * i + 2], 0.f) << "vehicle " << i;
	}

	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, RotorStepAndPlantResetErrorPaths)
{
	RotorInputs in(8);
	const px4rl_plant_params iris = px4rl_plant_params_iris();

	EXPECT_NE(rotorStep(nullptr, in, nullptr), 0);
	EXPECT_STRNE(px4rl_last_error(), "");
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(nullptr, nullptr, &iris, 1, nullptr), 0);

	px4rl_cuda_fleet *f = createFleet(8);
	ASSERT_NE(f, nullptr);

	// Required buffers and dt.
	EXPECT_NE(px4rl_cuda_rotor_step(f, nullptr, nullptr, in.velocity.get(), nullptr, kDt, in.force.get(),
					in.torque.get(), nullptr, nullptr), 0);
	EXPECT_NE(px4rl_cuda_rotor_step(f, in.actuator.get(), nullptr, nullptr, nullptr, kDt, in.force.get(),
					in.torque.get(), nullptr, nullptr), 0);
	EXPECT_NE(px4rl_cuda_rotor_step(f, in.actuator.get(), nullptr, in.velocity.get(), nullptr, kDt, nullptr,
					in.torque.get(), nullptr, nullptr), 0);
	EXPECT_NE(px4rl_cuda_rotor_step(f, in.actuator.get(), nullptr, in.velocity.get(), nullptr, kDt, in.force.get(),
					nullptr, nullptr, nullptr), 0);

	for (float bad_dt : {0.f, -kDt, static_cast<float>(NAN), static_cast<float>(INFINITY)}) {
		EXPECT_NE(px4rl_cuda_rotor_step(f, in.actuator.get(), nullptr, in.velocity.get(), nullptr, bad_dt,
						in.force.get(), in.torque.get(), nullptr, nullptr), 0) << bad_dt;
	}

	// n_plant must be 1 or n; the plant must be simulatable.
	px4rl_plant_params two[2] = {iris, iris};
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, two, 2, nullptr), 0);
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, two, 0, nullptr), 0);
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, nullptr, 1, nullptr), 0);
	px4rl_plant_params bad = iris;
	bad.num_rotors = 0;
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, &bad, 1, nullptr), 0);
	bad = iris;
	bad.rotor[0].px4_output_index = 4;
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, &bad, 1, nullptr), 0);
	bad = iris;
	bad.rotor[3].time_constant_up = 0.f;
	EXPECT_NE(px4rl_cuda_fleet_reset_plant(f, nullptr, &bad, 1, nullptr), 0);

	// The fleet is still usable after rejected calls.
	EXPECT_EQ(px4rl_cuda_fleet_reset_plant(f, nullptr, &iris, 1, nullptr), 0) << px4rl_last_error();
	EXPECT_EQ(rotorStep(f, in, nullptr), 0) << px4rl_last_error();
	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_TRUE(allFinite(in.force.download()));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, PositionStepOptionalOutputsMayBeNull)
{
	px4rl_cuda_fleet *f = createFleet(16);
	ASSERT_NE(f, nullptr);
	PositionInputs in(16);
	EXPECT_EQ(positionStep(f, in, 1000000u, nullptr, false), 0) << px4rl_last_error();
	CUDA_CALL(cudaDeviceSynchronize());
	EXPECT_TRUE(allFinite(in.q_sp.download()));
	EXPECT_TRUE(allFinite(in.thrust_sp.download()));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}

TEST(CApi, AttitudeStepWithoutSetpointUsesPositionStepOutput)
{
	constexpr int n = 64;
	px4rl_cuda_fleet *stored = createFleet(n);
	px4rl_cuda_fleet *explicit_fleet = createFleet(n);
	ASSERT_TRUE(stored && explicit_fleet);
	PositionInputs position_a(n), position_b(n);
	Inputs in_a(n), in_b(n);

	ASSERT_EQ(positionStep(stored, position_a, 1000000u, nullptr), 0) << px4rl_last_error();
	ASSERT_EQ(positionStep(explicit_fleet, position_b, 1000000u, nullptr), 0) << px4rl_last_error();

	// Stored setpoint (yaw rate included) versus the same setpoint passed explicitly.
	ASSERT_EQ(px4rl_cuda_attitude_step(stored, in_a.q.get(), nullptr, nullptr, position_a.thrust_sp.get(),
					   in_a.rates_sp.get(), in_a.thrust_body.get(), nullptr), 0) << px4rl_last_error();
	ASSERT_EQ(px4rl_cuda_attitude_step(explicit_fleet, in_b.q.get(), position_b.q_sp.get(), position_b.yawspeed.get(),
					   position_b.thrust_sp.get(), in_b.rates_sp.get(), in_b.thrust_body.get(), nullptr),
		  0) << px4rl_last_error();
	CUDA_CALL(cudaDeviceSynchronize());

	const std::vector<float> rates_stored = in_a.rates_sp.download();
	const std::vector<float> rates_explicit = in_b.rates_sp.download();
	ASSERT_TRUE(allFinite(rates_stored));
	EXPECT_EQ(std::memcmp(rates_stored.data(), rates_explicit.data(), rates_stored.size() * sizeof(float)), 0);

	// The command is a velocity step, so the setpoint is not the level one that a fresh fleet would use.
	px4rl_cuda_fleet *fresh = createFleet(n);
	Inputs in_fresh(n);
	ASSERT_EQ(px4rl_cuda_attitude_step(fresh, in_fresh.q.get(), nullptr, nullptr, position_a.thrust_sp.get(),
					   in_fresh.rates_sp.get(), in_fresh.thrust_body.get(), nullptr), 0);
	CUDA_CALL(cudaDeviceSynchronize());
	const std::vector<float> rates_fresh = in_fresh.rates_sp.download();
	EXPECT_NE(std::memcmp(rates_stored.data(), rates_fresh.data(), rates_stored.size() * sizeof(float)), 0);

	const std::vector<uint8_t> outcome = position_a.outcome.download();

	for (uint8_t o : outcome) {
		EXPECT_EQ(o, PX4RL_OUTCOME_OK);
	}

	EXPECT_EQ(px4rl_cuda_fleet_destroy(stored), 0);
	EXPECT_EQ(px4rl_cuda_fleet_destroy(explicit_fleet), 0);
	EXPECT_EQ(px4rl_cuda_fleet_destroy(fresh), 0);
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
	PositionInputs position_in(n);
	RotorInputs rotor_in(n);

	cudaStream_t stream = nullptr;
	CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	// Warm up so lazy module loading happens outside the capture.
	ASSERT_EQ(positionStep(f, position_in, 1000000u, stream), 0);
	ASSERT_EQ(attitudeStep(f, in, stream), 0);
	ASSERT_EQ(rateStep(f, in, stream), 0);
	ASSERT_EQ(rotorStep(f, rotor_in, stream), 0);
	CUDA_CALL(cudaStreamSynchronize(stream));

	// Any allocation or synchronisation inside a step call would invalidate the capture.
	CUDA_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
	EXPECT_EQ(positionStep(f, position_in, 1020000u, stream), 0) << px4rl_last_error();
	EXPECT_EQ(attitudeStep(f, in, stream), 0) << px4rl_last_error();
	EXPECT_EQ(rateStep(f, in, stream), 0) << px4rl_last_error();
	EXPECT_EQ(rotorStep(f, rotor_in, stream), 0) << px4rl_last_error();
	cudaGraph_t graph = nullptr;
	CUDA_CALL(cudaStreamEndCapture(stream, &graph));

	cudaGraphExec_t exec = nullptr;
	CUDA_CALL(cudaGraphInstantiate(&exec, graph, 0));

	for (int i = 0; i < 5; ++i) {
		CUDA_CALL(cudaGraphLaunch(exec, stream));
	}

	CUDA_CALL(cudaStreamSynchronize(stream));
	EXPECT_TRUE(allFinite(in.motors.download()));
	EXPECT_TRUE(allFinite(position_in.q_sp.download()));
	EXPECT_TRUE(allFinite(rotor_in.force.download()));
	EXPECT_TRUE(allFinite(rotor_in.torque.download()));

	CUDA_CALL(cudaGraphExecDestroy(exec));
	CUDA_CALL(cudaGraphDestroy(graph));
	CUDA_CALL(cudaStreamDestroy(stream));
	EXPECT_EQ(px4rl_cuda_fleet_destroy(f), 0);
}
