// Runs rate control -> control allocation -> actuator output once through the
// installed px4rl package and checks that every output is finite.
#include <cmath>
#include <cstdio>

#include <ActuatorOutputWrapper.hpp>
#include <ControlAllocatorWrapper.hpp>
#include <RateControllerWrapper.hpp>

int main()
{
	px4_stack::RateControllerWrapper rate_controller;
	px4_stack::ControlAllocatorWrapper allocator;
	px4_stack::ActuatorOutputWrapper actuator_output;

	px4_stack::RateControlInput rate_input;
	rate_input.rates_sp = matrix::Vector3f(0.1f, -0.1f, 0.05f);
	rate_input.rates_actual = matrix::Vector3f(0.f, 0.f, 0.f);
	rate_input.thrust_body = matrix::Vector3f(0.f, 0.f, -0.5f);
	rate_input.dt = 0.004f;
	px4_stack::RateControlOutput rate_output;
	rate_controller.update(rate_input, rate_output);

	px4_stack::ControlAllocatorInput allocator_input;
	allocator_input.torque_sp = rate_output.torque_sp;
	allocator_input.thrust_sp = rate_output.thrust_body;
	allocator_input.dt = rate_input.dt;
	px4_stack::ControlAllocatorOutput allocator_output;
	allocator.update(allocator_input, allocator_output);

	px4_stack::ActuatorOutputInput actuator_input;
	for (int i = 0; i < 4; ++i) {
		actuator_input.motors[i] = allocator_output.motors[i];
	}
	actuator_input.num_motors = allocator_output.num_motors;
	px4_stack::ActuatorOutputOutput pwm_output;
	actuator_output.update(actuator_input, pwm_output);

	bool all_finite = true;
	for (int i = 0; i < 4; ++i) {
		all_finite = all_finite && std::isfinite(allocator_output.motors[i]);
		std::printf("motor[%d]=%f pwm=%u\n", i, static_cast<double>(allocator_output.motors[i]),
			    static_cast<unsigned>(pwm_output.values[i]));
	}
	for (int i = 0; i < 3; ++i) {
		all_finite = all_finite && std::isfinite(rate_output.torque_sp(i));
	}

	if (!all_finite) {
		std::fprintf(stderr, "non-finite output\n");
		return 1;
	}
	return 0;
}
