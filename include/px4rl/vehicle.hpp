// One simulated vehicle's controller chain. This is the single source of the per-vehicle
// algorithm: CpuFleet and the CUDA kernels both call configure/positionStep/attitudeStep/rateStep.
#pragma once

#include <stdint.h>

#include <ActuatorOutputWrapper.hpp>
#include <AttitudeControllerWrapper.hpp>
#include <ControlAllocatorWrapper.hpp>
#include <RateControllerWrapper.hpp>

#include "px4rl/hover_thrust_module.hpp"
#include "px4rl/params.hpp"
#include "px4rl/position_module.hpp"

namespace px4rl {

struct RateStepResult {
	float motors[4]{0.f, 0.f, 0.f, 0.f}; ///< normalized [0, 1], before the thrust-curve inverse
	float actuator[4]{0.f, 0.f, 0.f, 0.f}; ///< PWM / DShot value
	matrix::Vector3f torque_sp{0.f, 0.f, 0.f};
};

struct PositionStepResult {
	matrix::Quatf q_sp;                ///< attitude setpoint, body FRD to NED
	matrix::Vector3f thrust_sp_body;   ///< normalized thrust, body FRD
	float yawspeed{0.f};               ///< yaw rate feed-forward, rad/s
	float hover_thrust{0.f};           ///< current hover thrust estimate
	PositionModule::Outcome outcome{PositionModule::Outcome::Ok};
};

class Vehicle {
public:
	/// Configures a freshly constructed vehicle. The vehicle models a flight that is already under way,
	/// so the tilt limit starts at MPC_TILTMAX_AIR instead of PX4's slew from the ground.
	PX4RL_HD void configure(const px4rl_params &p)
	{
		attitude_.setGains(toAttitudeConfig(p));
		rate_.setParams(toRateParams(p));
		allocator_.setParams(toAllocatorParams(p));
		output_.setParams(toOutputParams(p));
		position_.configure(p, p.position_dt);
		position_.setTiltLimitStateDeg(p.mpc_tiltmax_air);
		hover_thrust_.configure(p);
	}

	/// Position loop: hover thrust estimate, then the position/velocity controller.
	/// - `acceleration_z`: vertical acceleration in NED, positive down.
	/// - `in_air`: gates the hover thrust estimator; the position controller always runs. The estimator also
	///   waits for the first rateStep, because it needs the thrust that was actually commanded.
	/// - `setpoint`: NaN means unused; its timestamp is overwritten with `time_us` so that PX4's 200 ms
	///   fallback to the last valid setpoint applies on the caller's clock.
	/// The attitude setpoint is stored for attitudeStepFromPosition.
	PX4RL_HD void positionStep(const matrix::Vector3f &position, const matrix::Vector3f &velocity, float heading,
				   float acceleration_z, trajectory_setpoint_s setpoint, bool in_air, float dt,
				   uint64_t time_us, PositionStepResult &result)
	{
		hover_thrust_.update(true, in_air && thrust_recorded_, dt, acceleration_z, velocity, q_, thrust_sp_body_,
				     unallocated_thrust_, time_us);
		position_.applyHoverThrustEstimate(hover_thrust_.hoverThrust(), hover_thrust_.valid());

		setpoint.timestamp = time_us;
		vehicle_attitude_setpoint_s attitude_setpoint;
		result.outcome = position_.update(position, velocity, heading, dt, setpoint, time_us, attitude_setpoint);

		q_sp_ = matrix::Quatf(attitude_setpoint.q_d[0], attitude_setpoint.q_d[1], attitude_setpoint.q_d[2],
				      attitude_setpoint.q_d[3]);
		yawspeed_sp_ = attitude_setpoint.yaw_sp_move_rate;

		result.q_sp = q_sp_;
		result.thrust_sp_body = detail::load(attitude_setpoint.thrust_body);
		result.yawspeed = yawspeed_sp_;
		result.hover_thrust = hover_thrust_.hoverThrust();
	}

	/// Attitude loop with the setpoint of the last positionStep (level, zero yaw rate before the first one).
	PX4RL_HD void attitudeStepFromPosition(const matrix::Quatf &q, const matrix::Vector3f &thrust_sp_body,
					       matrix::Vector3f &rates_sp_out, matrix::Vector3f &thrust_body_out)
	{
		attitudeStep(q, q_sp_, yawspeed_sp_, thrust_sp_body, rates_sp_out, thrust_body_out);
	}

	PX4RL_HD void attitudeStep(const matrix::Quatf &q, const matrix::Quatf &q_sp, float yawspeed_ff,
				   const matrix::Vector3f &thrust_sp_body, matrix::Vector3f &rates_sp_out,
				   matrix::Vector3f &thrust_body_out)
	{
		q_ = q; // the hover thrust estimator rotates the commanded thrust with it
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
		thrust_sp_body_ = allocator_input.thrust_sp;
		unallocated_thrust_ = allocator_output.unallocated_thrust;
		thrust_recorded_ = true;

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
	PositionModule position_;
	HoverThrustModule hover_thrust_;

	// Inputs of the hover thrust estimator, recorded by attitudeStep and rateStep.
	matrix::Quatf q_{1.f, 0.f, 0.f, 0.f};
	matrix::Vector3f thrust_sp_body_{0.f, 0.f, 0.f};
	matrix::Vector3f unallocated_thrust_{0.f, 0.f, 0.f};
	bool thrust_recorded_{false};

	// Attitude setpoint of the last positionStep.
	matrix::Quatf q_sp_{1.f, 0.f, 0.f, 0.f};
	float yawspeed_sp_{0.f};
};

} // namespace px4rl
