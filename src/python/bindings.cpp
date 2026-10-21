// pybind11 module px4rl._cpu: the batched CpuFleet on numpy arrays and the single-vehicle controller classes.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "px4rl/airframes.h"
#include "px4rl/c_api.h"
#include "px4rl/cpu_fleet.hpp"
#include "px4rl/params.hpp"
#include "px4rl/plant.h"
#include "px4rl/vehicle.hpp"

namespace py = pybind11;

namespace {

// ---- parameter structs from a Python buffer (a ctypes struct or a ctypes array of them) ----------------------

template <typename T>
struct StructBuffer {
	py::buffer_info info;
	int count;

	T *data() const { return static_cast<T *>(info.ptr); }
};

template <typename T>
StructBuffer<T> structBuffer(const py::object &object, const char *name, bool writable)
{
	if (!PyObject_CheckBuffer(object.ptr())) {
		throw py::type_error(std::string(name) + " must be a ctypes struct or a ctypes array of them");
	}

	py::buffer_info info = py::reinterpret_borrow<py::buffer>(object).request(writable);

	if (info.itemsize != static_cast<py::ssize_t>(sizeof(T))) {
		throw py::value_error(std::string(name) + " has an item size of " + std::to_string(info.itemsize) +
				      " bytes, expected " + std::to_string(sizeof(T)) +
				      " (a struct from another px4rl version?)");
	}

	if (info.ndim > 1 || (info.ndim == 1 && info.strides[0] != info.itemsize)) {
		throw py::value_error(std::string(name) + " must be one struct or a contiguous 1-D array of them");
	}

	const int count = info.ndim == 0 ? 1 : static_cast<int>(info.shape[0]);
	return {std::move(info), count};
}

template <typename T>
const T &singleStruct(const py::object &object, const char *name)
{
	// The buffer_info is released when it goes out of scope, but the object keeps the memory alive.
	const StructBuffer<T> buffer = structBuffer<T>(object, name, false);

	if (buffer.count != 1) {
		throw py::value_error(std::string(name) + " must hold exactly one struct");
	}

	return *buffer.data();
}

px4rl_params controllerParams(const py::object &object)
{
	const px4rl_params &params = singleStruct<px4rl_params>(object, "params");

	if (const char *error = px4rl::validateParams(params)) {
		throw py::value_error(error);
	}

	return params;
}

template <typename T>
void fillStruct(const py::object &object, const T &value)
{
	const StructBuffer<T> buffer = structBuffer<T>(object, "buffer", true);

	if (buffer.count != 1) {
		throw py::value_error("buffer must hold exactly one struct");
	}

	*buffer.data() = value;
}

// ---- numpy arrays -------------------------------------------------------------------------------------------

std::string describe(const std::vector<py::ssize_t> &shape)
{
	std::string text = "(";

	for (size_t i = 0; i < shape.size(); ++i) {
		text += (i > 0 ? ", " : "") + std::to_string(shape[i]);
	}

	return text + (shape.size() == 1 ? ",)" : ")");
}

// Validates a float32 / uint8 C-contiguous numpy array of the given shape and returns its data pointer.
template <typename T>
T *arrayData(const py::object &object, const std::string &name, const std::vector<py::ssize_t> &shape, bool writable)
{
	if (!py::isinstance<py::array>(object)) {
		throw py::type_error(name + " must be a numpy array");
	}

	if (!py::array_t<T>::check_(object)) {
		throw py::value_error(name + " must have dtype " + py::str(py::dtype::of<T>()).cast<std::string>());
	}

	const auto array = py::reinterpret_borrow<py::array>(object);

	if (!(array.flags() & py::array::c_style)) {
		throw py::value_error(name + " must be C-contiguous");
	}

	bool same_shape = static_cast<size_t>(array.ndim()) == shape.size();

	for (size_t i = 0; same_shape && i < shape.size(); ++i) {
		same_shape = array.shape(static_cast<py::ssize_t>(i)) == shape[i];
	}

	if (!same_shape) {
		std::vector<py::ssize_t> actual(array.shape(), array.shape() + array.ndim());
		throw py::value_error(name + " must have shape " + describe(shape) + ", got " + describe(actual));
	}

	if (writable && !array.writeable()) {
		throw py::value_error(name + " must be writeable");
	}

	return static_cast<T *>(const_cast<void *>(array.data()));
}

template <typename T>
T *optionalArrayData(const py::object &object, const std::string &name, const std::vector<py::ssize_t> &shape,
		     bool writable)
{
	return object.is_none() ? nullptr : arrayData<T>(object, name, shape, writable);
}

py::array_t<float> zeros(std::vector<py::ssize_t> shape)
{
	py::array_t<float> array(shape);
	std::memset(array.mutable_data(), 0, static_cast<size_t>(array.size()) * sizeof(float));
	return array;
}

py::array_t<uint8_t> zerosUint8(std::vector<py::ssize_t> shape)
{
	py::array_t<uint8_t> array(shape);
	std::memset(array.mutable_data(), 0, static_cast<size_t>(array.size()));
	return array;
}

// Array-like (tuple, list, numpy array) of exactly N numbers.
template <size_t N>
std::array<float, N> readVector(const py::handle &object, const char *name)
{
	const auto array = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(object);

	if (!array) {
		throw py::type_error(std::string(name) + " must be array-like numbers");
	}

	if (array.ndim() != 1 || array.shape(0) != static_cast<py::ssize_t>(N)) {
		throw py::value_error(std::string(name) + " must have length " + std::to_string(N));
	}

	std::array<float, N> values;
	std::memcpy(values.data(), array.data(), N * sizeof(float));
	return values;
}

template <size_t N>
py::array_t<float> toArray(const float *values)
{
	py::array_t<float> array(static_cast<py::ssize_t>(N));
	std::memcpy(array.mutable_data(), values, N * sizeof(float));
	return array;
}

matrix::Vector3f toVector3(const py::handle &object, const char *name)
{
	const auto values = readVector<3>(object, name);
	return matrix::Vector3f(values[0], values[1], values[2]);
}

py::array_t<float> toArray(const matrix::Vector3f &v)
{
	const float values[3] = {v(0), v(1), v(2)};
	return toArray<3>(values);
}

py::object resultType(const char *name)
{
	return py::module_::import("px4rl._results").attr(name);
}

// ---- CpuFleet -----------------------------------------------------------------------------------------------

class CpuFleetBinding {
public:
	CpuFleetBinding(int n, int num_threads) : fleet_(n, num_threads), n_(n), last_thrust_sp_body_(py::none())
	{
		position_out_ = resultType("PositionResult")(zeros({n, 4}), zeros({n, 3}), zeros({n}), zeros({n}),
							      zerosUint8({n}));
		attitude_out_ = resultType("AttitudeResult")(zeros({n, 3}), zeros({n, 3}));
		rate_out_ = resultType("RateResult")(zeros({n, 4}), zeros({n, 4}), zeros({n, 3}));
		rotor_out_ = resultType("RotorResult")(zeros({n, 3}), zeros({n, 3}), zeros({n, 4, 3}));
	}

	int size() const { return n_; }
	int numThreads() const { return fleet_.numThreads(); }

	void reset(const py::object &params, const py::object &mask)
	{
		const StructBuffer<px4rl_params> buffer = structBuffer<px4rl_params>(params, "params", false);
		const uint8_t *mask_data = optionalArrayData<uint8_t>(mask, "mask", {n_}, false);
		const py::gil_scoped_release release;
		fleet_.reset(mask_data, buffer.data(), buffer.count);
	}

	void resetPlant(const py::object &plants, const py::object &mask)
	{
		const StructBuffer<px4rl_plant_params> buffer = structBuffer<px4rl_plant_params>(plants, "plants", false);
		const uint8_t *mask_data = optionalArrayData<uint8_t>(mask, "mask", {n_}, false);
		const py::gil_scoped_release release;
		fleet_.resetPlant(mask_data, buffer.data(), buffer.count);
	}

	py::object positionStep(const py::object &position, const py::object &velocity, const py::object &heading,
				const py::object &acceleration_z, const py::object &trajectory_sp, float dt,
				uint64_t time_us, const py::object &in_air, const py::object &out)
	{
		const py::object &result = out.is_none() ? position_out_ : out;
		const float *position_data = arrayData<float>(position, "position", {n_, 3}, false);
		const float *velocity_data = arrayData<float>(velocity, "velocity", {n_, 3}, false);
		const float *heading_data = arrayData<float>(heading, "heading", {n_}, false);
		const float *acceleration_data = arrayData<float>(acceleration_z, "acceleration_z", {n_}, false);
		const float *trajectory_data = arrayData<float>(trajectory_sp, "trajectory_sp", {n_, 11}, false);
		const uint8_t *in_air_data = optionalArrayData<uint8_t>(in_air, "in_air", {n_}, false);

		const py::object thrust = field(result, "thrust_sp_body");
		float *q_sp_data = arrayData<float>(field(result, "q_sp"), "out.q_sp", {n_, 4}, true);
		float *thrust_data = arrayData<float>(thrust, "out.thrust_sp_body", {n_, 3}, true);
		float *yawspeed_data = arrayData<float>(field(result, "yawspeed"), "out.yawspeed", {n_}, true);
		float *hover_data = optionalArrayData<float>(field(result, "hover_thrust"), "out.hover_thrust", {n_}, true);
		uint8_t *outcome_data = optionalArrayData<uint8_t>(field(result, "outcome"), "out.outcome", {n_}, true);

		{
			const py::gil_scoped_release release;
			fleet_.positionStep(position_data, velocity_data, heading_data, acceleration_data, trajectory_data,
					    in_air_data, dt, time_us, q_sp_data, thrust_data, yawspeed_data, hover_data,
					    outcome_data);
		}

		last_thrust_sp_body_ = thrust;
		return result;
	}

	py::object attitudeStep(const py::object &q, const py::object &q_sp, const py::object &thrust_sp_body,
				const py::object &yawspeed_ff, const py::object &out)
	{
		const py::object &result = out.is_none() ? attitude_out_ : out;
		const py::object thrust = thrust_sp_body.is_none() ? last_thrust_sp_body_ : thrust_sp_body;

		if (thrust.is_none()) {
			throw py::value_error("thrust_sp_body is required before the first position_step");
		}

		const float *q_data = arrayData<float>(q, "q", {n_, 4}, false);
		const float *q_sp_data = optionalArrayData<float>(q_sp, "q_sp", {n_, 4}, false);
		const float *yawspeed_data = optionalArrayData<float>(yawspeed_ff, "yawspeed_ff", {n_}, false);
		const float *thrust_data = arrayData<float>(thrust, "thrust_sp_body", {n_, 3}, false);
		float *rates_sp_data = arrayData<float>(field(result, "rates_sp"), "out.rates_sp", {n_, 3}, true);
		float *thrust_body_data = arrayData<float>(field(result, "thrust_body"), "out.thrust_body", {n_, 3}, true);

		{
			const py::gil_scoped_release release;
			fleet_.attitudeStep(q_data, q_sp_data, yawspeed_data, thrust_data, rates_sp_data, thrust_body_data);
		}

		return result;
	}

	py::object rateStep(const py::object &rates_sp, const py::object &thrust_body, const py::object &rates,
			    const py::object &angular_accel, float dt, const py::object &landed, const py::object &out)
	{
		const py::object &result = out.is_none() ? rate_out_ : out;
		const float *rates_sp_data = arrayData<float>(rates_sp, "rates_sp", {n_, 3}, false);
		const float *thrust_data = arrayData<float>(thrust_body, "thrust_body", {n_, 3}, false);
		const float *rates_data = arrayData<float>(rates, "rates", {n_, 3}, false);
		const float *accel_data = arrayData<float>(angular_accel, "angular_accel", {n_, 3}, false);
		const uint8_t *landed_data = optionalArrayData<uint8_t>(landed, "landed", {n_}, false);
		float *motors_data = arrayData<float>(field(result, "motors"), "out.motors", {n_, 4}, true);
		float *actuator_data = optionalArrayData<float>(field(result, "actuator"), "out.actuator", {n_, 4}, true);
		float *torque_data = optionalArrayData<float>(field(result, "torque_sp"), "out.torque_sp", {n_, 3}, true);

		{
			const py::gil_scoped_release release;
			fleet_.rateStep(rates_sp_data, thrust_data, rates_data, accel_data, dt, landed_data, motors_data,
					actuator_data, torque_data);
		}

		return result;
	}

	py::object rotorStep(const py::object &actuator, const py::object &body_velocity_flu, float dt,
			     const py::object &body_rates_flu, const py::object &armed, const py::object &out)
	{
		const py::object &result = out.is_none() ? rotor_out_ : out;
		const float *actuator_data = arrayData<float>(actuator, "actuator", {n_, 4}, false);
		const float *velocity_data = arrayData<float>(body_velocity_flu, "body_velocity_flu", {n_, 3}, false);
		const float *rates_data = optionalArrayData<float>(body_rates_flu, "body_rates_flu", {n_, 3}, false);
		const uint8_t *armed_data = optionalArrayData<uint8_t>(armed, "armed", {n_}, false);
		float *force_data = arrayData<float>(field(result, "force"), "out.force", {n_, 3}, true);
		float *torque_data = arrayData<float>(field(result, "torque"), "out.torque", {n_, 3}, true);
		float *rotor_data = optionalArrayData<float>(field(result, "rotor_force"), "out.rotor_force", {n_, 4, 3}, true);

		{
			const py::gil_scoped_release release;
			fleet_.rotorStep(actuator_data, armed_data, velocity_data, rates_data, dt, force_data, torque_data,
					 rotor_data);
		}

		return result;
	}

private:
	static py::object field(const py::object &result, const char *name) { return py::getattr(result, name, py::none()); }

	px4rl::CpuFleet fleet_;
	py::ssize_t n_;
	py::object position_out_, attitude_out_, rate_out_, rotor_out_;
	py::object last_thrust_sp_body_; // None until the first position_step
};

// ---- single-vehicle classes ---------------------------------------------------------------------------------

constexpr float kNaN = NAN;

trajectory_setpoint_s toSetpoint(const py::handle &trajectory_sp, uint64_t time_us)
{
	const auto sp = readVector<11>(trajectory_sp, "trajectory_sp");
	return {time_us, {sp[0], sp[1], sp[2]}, {sp[3], sp[4], sp[5]}, {sp[6], sp[7], sp[8]}, {kNaN, kNaN, kNaN},
		sp[9], sp[10]};
}

matrix::Quatf toQuaternion(const py::handle &object, const char *name)
{
	const auto q = readVector<4>(object, name);
	return matrix::Quatf(q[0], q[1], q[2], q[3]);
}

py::array_t<float> toArray(const matrix::Quatf &q)
{
	const float values[4] = {q(0), q(1), q(2), q(3)};
	return toArray<4>(values);
}

py::array_t<bool> toArray(const matrix::Vector3<bool> &v)
{
	py::array_t<bool> array(3);
	bool *data = array.mutable_data();

	for (int i = 0; i < 3; ++i) {
		data[i] = v(i);
	}

	return array;
}

// The full controller chain of one vehicle, with the same method set as CpuFleet.
class VehicleBinding {
public:
	explicit VehicleBinding(const py::object &params) { vehicle_.configure(controllerParams(params)); }

	py::object positionStep(const py::object &position, const py::object &velocity, float heading,
				float acceleration_z, const py::object &trajectory_sp, float dt, uint64_t time_us,
				bool in_air)
	{
		px4rl::PositionStepResult result;
		vehicle_.positionStep(toVector3(position, "position"), toVector3(velocity, "velocity"), heading,
				      acceleration_z, toSetpoint(trajectory_sp, time_us), in_air, dt, time_us, result);
		last_thrust_sp_body_ = result.thrust_sp_body;
		has_thrust_ = true;
		return resultType("PositionResult")(toArray(result.q_sp), toArray(result.thrust_sp_body), result.yawspeed,
						    result.hover_thrust, static_cast<int>(result.outcome));
	}

	py::object attitudeStep(const py::object &q, const py::object &q_sp, const py::object &thrust_sp_body,
				float yawspeed_ff)
	{
		matrix::Vector3f thrust;

		if (!thrust_sp_body.is_none()) {
			thrust = toVector3(thrust_sp_body, "thrust_sp_body");

		} else if (has_thrust_) {
			thrust = last_thrust_sp_body_;

		} else {
			throw py::value_error("thrust_sp_body is required before the first position_step");
		}

		matrix::Vector3f rates_sp;
		matrix::Vector3f thrust_body;
		const matrix::Quatf attitude = toQuaternion(q, "q");

		if (q_sp.is_none()) {
			vehicle_.attitudeStepFromPosition(attitude, thrust, rates_sp, thrust_body);

		} else {
			vehicle_.attitudeStep(attitude, toQuaternion(q_sp, "q_sp"), yawspeed_ff, thrust, rates_sp,
					      thrust_body);
		}

		return resultType("AttitudeResult")(toArray(rates_sp), toArray(thrust_body));
	}

	py::object rateStep(const py::object &rates_sp, const py::object &thrust_body, const py::object &rates,
			    const py::object &angular_accel, float dt, bool landed)
	{
		px4rl::RateStepResult result;
		vehicle_.rateStep(toVector3(rates_sp, "rates_sp"), toVector3(thrust_body, "thrust_body"),
				  toVector3(rates, "rates"), toVector3(angular_accel, "angular_accel"), dt, landed, result);
		return resultType("RateResult")(toArray<4>(result.motors), toArray<4>(result.actuator),
						toArray(result.torque_sp));
	}

private:
	px4rl::Vehicle vehicle_;
	matrix::Vector3f last_thrust_sp_body_;
	bool has_thrust_{false};
};

} // namespace

PYBIND11_MODULE(_cpu, m)
{
	m.doc() = "px4rl CPU backend: CpuFleet and the single-vehicle controller classes (numpy).";

	// ---- parameter factories and struct sizes ----
	m.def("params_default", [](const py::object &buffer) { fillStruct(buffer, px4rl::defaultParams()); },
	      py::arg("buffer"), "Writes the default controller parameters into a Params.");
	m.def("params_iris", [](const py::object &buffer) { fillStruct(buffer, px4rl::irisParams()); }, py::arg("buffer"),
	      "Writes the Iris controller parameters into a Params.");
	m.def("plant_params_iris", [](const py::object &buffer) { fillStruct(buffer, px4rl_plant_params_iris()); },
	      py::arg("buffer"), "Writes the Iris rotor plant into a PlantParams.");
	m.def("rigid_body_params_iris", [](const py::object &buffer) { fillStruct(buffer, px4rl_rigid_body_params_iris()); },
	      py::arg("buffer"), "Writes the Iris mass and inertia into a RigidBodyParams.");
	m.def("params_size", []() { return px4rl_params_size(); }, "sizeof(px4rl_params) in this build.");
	m.def("plant_params_size", []() { return px4rl_plant_params_size(); }, "sizeof(px4rl_plant_params).");
	m.def("rigid_body_params_size", []() { return px4rl_rigid_body_params_size(); },
	      "sizeof(px4rl_rigid_body_params).");

	// ---- CpuFleet ----
	py::class_<CpuFleetBinding>(m, "CpuFleet",
				    "N vehicles stepped on the CPU with OpenMP. Arrays are float32 and C-contiguous (uint8 for "
				    "masks). Outputs go into buffers the fleet owns and are returned (the next call overwrites "
				    "them), or into `out=`.")
		.def(py::init<int, int>(), py::arg("n"), py::arg("num_threads") = 0,
		     "num_threads: OpenMP threads per call; 0 is the OpenMP default, 1 runs serially. "
		     "The results do not depend on it.")
		.def_property_readonly("n", &CpuFleetBinding::size)
		.def_property_readonly("num_threads", &CpuFleetBinding::numThreads)
		.def("__len__", &CpuFleetBinding::size)
		.def("reset", &CpuFleetBinding::reset, py::arg("params"), py::arg("mask") = py::none(),
		     "Reconstructs the controllers of the vehicles selected by mask (uint8 (n,), None = all). params is "
		     "a Params (broadcast) or a ctypes array of n Params. The rotors are not touched.")
		.def("reset_plant", &CpuFleetBinding::resetPlant, py::arg("plants"), py::arg("mask") = py::none(),
		     "Replaces the rotor plant of the selected vehicles and puts their rotors at rest.")
		.def("position_step", &CpuFleetBinding::positionStep, py::arg("position"), py::arg("velocity"),
		     py::arg("heading"), py::arg("acceleration_z"), py::arg("trajectory_sp"), py::arg("dt"),
		     py::arg("time_us"), py::arg("in_air") = py::none(), py::kw_only(), py::arg("out") = py::none(),
		     "Position and velocity loop with the hover thrust estimator. Returns a PositionResult. Its thrust "
		     "is the default thrust_sp_body of attitude_step.")
		.def("attitude_step", &CpuFleetBinding::attitudeStep, py::arg("q"), py::arg("q_sp") = py::none(),
		     py::arg("thrust_sp_body") = py::none(), py::arg("yawspeed_ff") = py::none(), py::kw_only(),
		     py::arg("out") = py::none(),
		     "Attitude loop. q_sp=None uses the setpoint of the last position_step. Returns an AttitudeResult.")
		.def("rate_step", &CpuFleetBinding::rateStep, py::arg("rates_sp"), py::arg("thrust_body"),
		     py::arg("rates"), py::arg("angular_accel"), py::arg("dt"), py::arg("landed") = py::none(),
		     py::kw_only(), py::arg("out") = py::none(),
		     "Rate controller, allocator and actuator output. Returns a RateResult.")
		.def("rotor_step", &CpuFleetBinding::rotorStep, py::arg("actuator"), py::arg("body_velocity_flu"),
		     py::arg("dt"), py::arg("body_rates_flu") = py::none(), py::arg("armed") = py::none(),
		     py::kw_only(), py::arg("out") = py::none(),
		     "Gazebo rotor model: PWM to the wrench on the body, body FLU. Returns a RotorResult.");

	// ---- single-vehicle controllers ----
	py::class_<px4_stack::AttitudeControllerWrapper>(m, "AttitudeController",
							 "PX4 attitude controller of one vehicle.")
		.def(py::init([](const py::object &params) {
			     auto controller = std::make_unique<px4_stack::AttitudeControllerWrapper>();
			     controller->setGains(px4rl::toAttitudeConfig(controllerParams(params)));
			     return controller;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4_stack::AttitudeControllerWrapper &controller, const py::object &q, const py::object &q_sp,
			   const py::object &thrust_sp_body, float yawspeed_ff) {
				px4_stack::AttitudeState state;
				state.q = toQuaternion(q, "q");
				px4_stack::AttitudeSetpoint setpoint;
				setpoint.q_d = toQuaternion(q_sp, "q_sp");
				setpoint.yaw_sp_move_rate = yawspeed_ff;
				setpoint.thrust_body = toVector3(thrust_sp_body, "thrust_sp_body");
				const px4_stack::RateSetpoint rate_setpoint = controller.update(state, setpoint);
				return resultType("AttitudeResult")(toArray(rate_setpoint.rates),
								    toArray(rate_setpoint.thrust_body));
			},
			py::arg("q"), py::arg("q_sp"), py::arg("thrust_sp_body"), py::arg("yawspeed_ff") = 0.f,
			"Returns an AttitudeResult (rates_sp, thrust_body).");

	py::class_<px4_stack::RateControllerWrapper>(m, "RateController", "PX4 rate controller of one vehicle.")
		.def(py::init([](const py::object &params) {
			     auto controller = std::make_unique<px4_stack::RateControllerWrapper>();
			     controller->setParams(px4rl::toRateParams(controllerParams(params)));
			     return controller;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4_stack::RateControllerWrapper &controller, const py::object &rates_sp,
			   const py::object &thrust_body, const py::object &rates, const py::object &angular_accel,
			   float dt, bool landed) {
				px4_stack::RateControlInput input;
				input.rates_sp = toVector3(rates_sp, "rates_sp");
				input.rates_actual = toVector3(rates, "rates");
				input.angular_accel = toVector3(angular_accel, "angular_accel");
				input.thrust_body = toVector3(thrust_body, "thrust_body");
				input.dt = dt;
				input.landed = landed;
				px4_stack::RateControlOutput output;
				controller.update(input, output);
				return py::make_tuple(toArray(output.torque_sp), toArray(output.thrust_body));
			},
			py::arg("rates_sp"), py::arg("thrust_body"), py::arg("rates"), py::arg("angular_accel"),
			py::arg("dt"), py::arg("landed") = false, "Returns (torque_sp, thrust_body).")
		.def(
			"set_saturation_status",
			[](px4_stack::RateControllerWrapper &controller, const py::object &positive,
			   const py::object &negative) {
				const auto p = readVector<3>(positive, "positive");
				const auto n = readVector<3>(negative, "negative");
				controller.setSaturationStatus(matrix::Vector3<bool>(p[0] != 0.f, p[1] != 0.f, p[2] != 0.f),
							       matrix::Vector3<bool>(n[0] != 0.f, n[1] != 0.f, n[2] != 0.f));
			},
			py::arg("positive"), py::arg("negative"),
			"Allocator saturation per axis (roll, pitch, yaw), for the integrator anti-windup.")
		.def("reset_integral", &px4_stack::RateControllerWrapper::resetIntegral)
		.def_property_readonly("integrator", [](px4_stack::RateControllerWrapper &controller) {
			px4_stack::RateCtrlStatus status;
			controller.getStatus(status);
			const float values[3] = {status.rollspeed_integ, status.pitchspeed_integ, status.yawspeed_integ};
			return toArray<3>(values);
		});

	py::class_<px4_stack::ControlAllocatorWrapper>(m, "ControlAllocator", "PX4 control allocator of one vehicle.")
		.def(py::init([](const py::object &params) {
			     auto allocator = std::make_unique<px4_stack::ControlAllocatorWrapper>();
			     allocator->setParams(px4rl::toAllocatorParams(controllerParams(params)));
			     return allocator;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4_stack::ControlAllocatorWrapper &allocator, const py::object &torque_sp,
			   const py::object &thrust_sp, float dt) {
				px4_stack::ControlAllocatorInput input;
				input.torque_sp = toVector3(torque_sp, "torque_sp");
				input.thrust_sp = toVector3(thrust_sp, "thrust_sp");
				input.dt = dt;
				px4_stack::ControlAllocatorOutput output;
				allocator.update(input, output);
				return resultType("AllocatorResult")(toArray<4>(output.motors),
								     toArray(output.saturation_positive),
								     toArray(output.saturation_negative),
								     toArray(output.unallocated_torque),
								     toArray(output.unallocated_thrust));
			},
			py::arg("torque_sp"), py::arg("thrust_sp"), py::arg("dt"), "Returns an AllocatorResult.");

	py::class_<px4_stack::ActuatorOutputWrapper>(m, "ActuatorOutput",
						     "Thrust-curve inverse and PWM / DShot mapping of one vehicle.")
		.def(py::init([](const py::object &params) {
			     auto output = std::make_unique<px4_stack::ActuatorOutputWrapper>();
			     output->setParams(px4rl::toOutputParams(controllerParams(params)));
			     return output;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4_stack::ActuatorOutputWrapper &output, const py::object &motors) {
				const auto values = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(motors);

				if (!values || values.ndim() != 1 || values.shape(0) < 1 || values.shape(0) > 4) {
					throw py::value_error("motors must be 1 to 4 numbers");
				}

				px4_stack::ActuatorOutputInput input;
				input.num_motors = static_cast<int>(values.shape(0));

				for (int i = 0; i < input.num_motors; ++i) {
					input.motors[i] = values.data()[i];
				}

				px4_stack::ActuatorOutputOutput result;
				output.update(input, result);
				py::array_t<uint16_t> array(4);

				for (int i = 0; i < 4; ++i) {
					array.mutable_data()[i] = result.values[i];
				}

				return array;
			},
			py::arg("motors"), "Motor commands in [0, 1] to PWM (microseconds) or DShot values, uint16 (4,).");

	py::class_<px4rl::PositionModule>(m, "PositionController",
					  "PX4 position / velocity controller of one vehicle (the in-flight path of "
					  "MulticopterPositionControl), starting at the configured tilt limit.")
		.def(py::init([](const py::object &params) {
			     const px4rl_params p = controllerParams(params);
			     auto module = std::make_unique<px4rl::PositionModule>();
			     module->configure(p, p.position_dt);
			     module->setTiltLimitStateDeg(p.mpc_tiltmax_air);
			     return module;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4rl::PositionModule &module, const py::object &position, const py::object &velocity,
			   float heading, float dt, const py::object &trajectory_sp, uint64_t time_us) {
				vehicle_attitude_setpoint_s attitude_setpoint;
				const auto outcome = module.update(toVector3(position, "position"),
								   toVector3(velocity, "velocity"), heading, dt,
								   toSetpoint(trajectory_sp, time_us), time_us,
								   attitude_setpoint);
				const float q[4] = {attitude_setpoint.q_d[0], attitude_setpoint.q_d[1],
						    attitude_setpoint.q_d[2], attitude_setpoint.q_d[3]};
				const float thrust[3] = {attitude_setpoint.thrust_body[0], attitude_setpoint.thrust_body[1],
							 attitude_setpoint.thrust_body[2]};
				return resultType("PositionControlResult")(toArray<4>(q), toArray<3>(thrust),
									   attitude_setpoint.yaw_sp_move_rate,
									   static_cast<int>(outcome));
			},
			py::arg("position"), py::arg("velocity"), py::arg("heading"), py::arg("dt"),
			py::arg("trajectory_sp"), py::arg("time_us"),
			"One position-loop cycle. trajectory_sp is (pos[3], vel[3], acc[3], yaw, yawspeed) with NaN for "
			"unused fields. Returns a PositionControlResult.")
		.def("apply_hover_thrust_estimate", &px4rl::PositionModule::applyHoverThrustEstimate,
		     py::arg("hover_thrust"), py::arg("valid"),
		     "Hands the estimate of a HoverThrustEstimator to the controller (used if MPC_USE_HTE is set).");

	py::class_<px4rl::HoverThrustModule>(m, "HoverThrustEstimator",
					     "PX4 hover thrust estimator of one vehicle (MulticopterHoverThrustEstimator).")
		.def(py::init([](const py::object &params) {
			     auto module = std::make_unique<px4rl::HoverThrustModule>();
			     module->configure(controllerParams(params));
			     return module;
		     }),
		     py::arg("params"))
		.def(
			"update",
			[](px4rl::HoverThrustModule &module, bool armed, bool in_air, float dt, float acceleration_z,
			   const py::object &velocity, const py::object &q, const py::object &thrust_sp_body,
			   const py::object &unallocated_thrust, uint64_t time_us) {
				module.update(armed, in_air, dt, acceleration_z, toVector3(velocity, "velocity"),
					      toQuaternion(q, "q"), toVector3(thrust_sp_body, "thrust_sp_body"),
					      toVector3(unallocated_thrust, "unallocated_thrust"), time_us);
				return module.hoverThrust();
			},
			py::arg("armed"), py::arg("in_air"), py::arg("dt"), py::arg("acceleration_z"),
			py::arg("velocity"), py::arg("q"), py::arg("thrust_sp_body"), py::arg("unallocated_thrust"),
			py::arg("time_us"), "One estimator cycle. Returns the hover thrust estimate.")
		.def("reset", &px4rl::HoverThrustModule::reset)
		.def_property_readonly("hover_thrust", &px4rl::HoverThrustModule::hoverThrust)
		.def_property_readonly("valid", &px4rl::HoverThrustModule::valid);

	py::class_<VehicleBinding>(m, "Vehicle",
				   "The whole controller chain of one vehicle, with the method set of CpuFleet.")
		.def(py::init<const py::object &>(), py::arg("params"))
		.def("position_step", &VehicleBinding::positionStep, py::arg("position"), py::arg("velocity"),
		     py::arg("heading"), py::arg("acceleration_z"), py::arg("trajectory_sp"), py::arg("dt"),
		     py::arg("time_us"), py::arg("in_air") = true, "Returns a PositionResult.")
		.def("attitude_step", &VehicleBinding::attitudeStep, py::arg("q"), py::arg("q_sp") = py::none(),
		     py::arg("thrust_sp_body") = py::none(), py::arg("yawspeed_ff") = 0.f,
		     "q_sp=None uses the setpoint of the last position_step. Returns an AttitudeResult.")
		.def("rate_step", &VehicleBinding::rateStep, py::arg("rates_sp"), py::arg("thrust_body"),
		     py::arg("rates"), py::arg("angular_accel"), py::arg("dt"), py::arg("landed") = false,
		     "Returns a RateResult.");
}
