// One simulated vehicle's controller chain. This is the single source of the per-vehicle
// algorithm: CpuFleet and the CUDA kernels both call configure/attitudeStep/rateStep.
#pragma once

#include <ActuatorOutputWrapper.hpp>
#include <AttitudeControllerWrapper.hpp>
#include <ControlAllocatorWrapper.hpp>
#include <RateControllerWrapper.hpp>

#include "px4rl/params.hpp"

namespace px4rl {

struct RateStepResult {
	float motors[4]{0.f, 0.f, 0.f, 0.f}; ///< normalized [0, 1], before the thrust-curve inverse
	float actuator[4]{0.f, 0.f, 0.f, 0.f}; ///< PWM / DShot value
	matrix::Vector3f torque_sp{0.f, 0.f, 0.f};
};

class Vehicle {
public:
	PX4RL_HD void configure(const px4rl_params &p)
	{
		attitude_.setGains(toAttitudeConfig(p));
		rate_.setParams(toRateParams(p));
		allocator_.setParams(toAllocatorParams(p));
		output_.setParams(toOutputParams(p));
	}

	PX4RL_HD void attitudeStep(const matrix::Quatf &q, const matrix::Quatf &q_sp, float yawspeed_ff,
				   const matrix::Vector3f &thrust_sp_body, matrix::Vector3f &rates_sp_out,
				   matrix::Vector3f &thrust_body_out)
	{
		px4_stack::AttitudeState state;
		state.q = q;
		px4_stack::AttitudeSetpoint setpoint;
		setpoint.q_d = q_sp;
		setpoint.yaw_sp_move_rate = yawspeed_ff;
		setpoint.thrust_body = thrust_sp_body;
		const px4_stack::RateSetpoint rate_setpoint = attitude_.update(state, setpoint);
		rates_sp_out = rate_setpoint.rates;
		thrust_body_out = rate_setpoint.thrust_body;
	}

	PX4RL_HD void rateStep(const matrix::Vector3f &rates_sp, const matrix::Vector3f &thrust_body,
			       const matrix::Vector3f &rates, const matrix::Vector3f &angular_accel, float dt,
			       bool landed, RateStepResult &result)
	{
		px4_stack::RateControlInput rate_input;
		rate_input.rates_sp = rates_sp;
		rate_input.rates_actual = rates;
		rate_input.angular_accel = angular_accel;
		rate_input.thrust_body = thrust_body;
		rate_input.dt = dt;
		rate_input.landed = landed;
		px4_stack::RateControlOutput rate_output;
		rate_.update(rate_input, rate_output);

		px4_stack::ControlAllocatorInput allocator_input;
		allocator_input.torque_sp = rate_output.torque_sp;
		allocator_input.thrust_sp = rate_output.thrust_body;
		allocator_input.dt = dt;
		px4_stack::ControlAllocatorOutput allocator_output;
		allocator_.update(allocator_input, allocator_output);
		rate_.setSaturationStatus(allocator_output.saturation_positive, allocator_output.saturation_negative);

		px4_stack::ActuatorOutputInput output_input;
		output_input.num_motors = allocator_output.num_motors;
		for (int i = 0; i < 4; ++i) {
			output_input.motors[i] = allocator_output.motors[i];
		}
		px4_stack::ActuatorOutputOutput output_output;
		output_.update(output_input, output_output);

		for (int i = 0; i < 4; ++i) {
			result.motors[i] = allocator_output.motors[i];
			result.actuator[i] = static_cast<float>(output_output.values[i]);
		}
		result.torque_sp = rate_output.torque_sp;
	}

private:
	px4_stack::AttitudeControllerWrapper attitude_;
	px4_stack::RateControllerWrapper rate_;
	px4_stack::ControlAllocatorWrapper allocator_;
	px4_stack::ActuatorOutputWrapper output_;
};

} // namespace px4rl
