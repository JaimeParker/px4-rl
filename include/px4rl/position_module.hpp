// px4rl::PositionModule: the in-flight path of PX4's MulticopterPositionControl module.
//
// Reproduces modules/mc_pos_control/MulticopterPositionControl.cpp (PX4 v1.16.2) around the vendored
// PositionControl class. The vehicle is always "flying": no takeoff or land state machine, land detector,
// goto control, EKF-reset setpoint adjustment, z_deriv blending, or offboard want_takeoff handling
// (see px4/PROVENANCE.md). Every block below names the PX4 lines it mirrors.
#pragma once

#include <stdint.h>

#include <px4_defines.h> // float math constants, needed by NotchFilter.hpp

#include <PositionControl.hpp>
#include <mathlib/math/filter/AlphaFilter.hpp>
#include <mathlib/math/filter/NotchFilter.hpp>
#include <mathlib/mathlib.h>
#include <slew_rate/SlewRate.hpp>

#include "px4rl/params.hpp"

namespace px4rl {

// PositionControl::empty_trajectory_setpoint is a host-only static data member, so device code uses this.
// Mirrors PositionControl.cpp:45.
PX4RL_HD inline trajectory_setpoint_s emptyTrajectorySetpoint()
{
	return {0, {NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN, NAN, NAN}, {NAN, NAN, NAN}, NAN, NAN};
}

class PositionModule {
public:
	enum class Outcome : uint8_t {
		Ok,       ///< the input setpoint was executed
		Fallback, ///< the input setpoint was invalid, the last valid one (< 200 ms old) was executed
		Failsafe, ///< neither was executable, the failsafe setpoint was executed
	};

	/// Mirrors the constructor (MulticopterPositionControl.cpp:46-52) and parameters_update (:73-304),
	/// restricted to the _control setters and the filter setup. `sample_interval_s` is the nominal
	/// interval between update() calls; PX4 uses the running mean of dt, seeded with 0.01 (:49).
	PX4RL_HD void configure(const px4rl_params &p, float sample_interval_s = 0.01f)
	{
		_tilt_limit_slew_rate.setSlewRate(.2f); // :51

		// :84-116 filter setup
		const float sample_freq_hz = 1.f / sample_interval_s;

		if ((p.mpc_vel_nf_frq > 0.f) && (p.mpc_vel_nf_bw > 0.f)) {
			_vel_xy_notch_filter.setParameters(sample_freq_hz, p.mpc_vel_nf_frq, p.mpc_vel_nf_bw);
			_vel_z_notch_filter.setParameters(sample_freq_hz, p.mpc_vel_nf_frq, p.mpc_vel_nf_bw);

		} else {
			_vel_xy_notch_filter.disable();
			_vel_z_notch_filter.disable();
		}

		if (p.mpc_vel_lp > 0.f) {
			_vel_xy_lp_filter.setCutoffFreq(sample_freq_hz, p.mpc_vel_lp);
			_vel_z_lp_filter.setCutoffFreq(sample_freq_hz, p.mpc_vel_lp);

		} else {
			_vel_xy_lp_filter.setAlpha(1.f);
			_vel_z_lp_filter.setAlpha(1.f);
		}

		if (p.mpc_veld_lp > 0.f) {
			_vel_deriv_xy_lp_filter.setCutoffFreq(sample_freq_hz, p.mpc_veld_lp);
			_vel_deriv_z_lp_filter.setCutoffFreq(sample_freq_hz, p.mpc_veld_lp);

		} else {
			_vel_deriv_xy_lp_filter.setAlpha(1.f);
			_vel_deriv_z_lp_filter.setAlpha(1.f);
		}

		// :196-202 _control setters
		_control.setPositionGains(matrix::Vector3f(p.mpc_xy_p, p.mpc_xy_p, p.mpc_z_p));
		_control.setVelocityGains(
			matrix::Vector3f(p.mpc_xy_vel_p_acc, p.mpc_xy_vel_p_acc, p.mpc_z_vel_p_acc),
			matrix::Vector3f(p.mpc_xy_vel_i_acc, p.mpc_xy_vel_i_acc, p.mpc_z_vel_i_acc),
			matrix::Vector3f(p.mpc_xy_vel_d_acc, p.mpc_xy_vel_d_acc, p.mpc_z_vel_d_acc));
		_control.setHorizontalThrustMargin(p.mpc_thr_xy_marg);
		_control.decoupleHorizontalAndVecticalAcceleration(p.mpc_acc_decouple != 0);

		// :295-298 hover thrust initialisation
		if (!p.mpc_use_hte || !_hover_thrust_initialized) {
			_control.setHoverThrust(p.mpc_thr_hover);
			_hover_thrust_initialized = true;
		}

		// Parameters read directly by Run()
		_use_hte = p.mpc_use_hte != 0;
		_xy_vel_max = p.mpc_xy_vel_max;
		_z_vel_max_up = p.mpc_z_vel_max_up;
		_z_vel_max_dn = p.mpc_z_vel_max_dn;
		_land_speed = math::min(p.mpc_land_speed, p.mpc_z_vel_max_dn); // :302
		_tiltmax_air_deg = p.mpc_tiltmax_air;
		_thr_min = p.mpc_thr_min;
		_thr_max = p.mpc_thr_max;
	}

	/// Mirrors Run():416-424 (hover thrust estimate subscription), taking the estimate from the caller.
	PX4RL_HD void applyHoverThrustEstimate(float hover_thrust, bool valid)
	{
		if (_use_hte && valid) {
			_control.updateHoverThrust(hover_thrust);
		}
	}

	/// PX4 slews the tilt limit from 0 at 0.2 rad/s after construction (:51, :520-522), because it
	/// starts on the ground. Forcing the state lets a caller that models a vehicle already in flight
	/// start at the configured limit.
	PX4RL_HD void setTiltLimitStateDeg(float tilt_deg)
	{
		_tilt_limit_slew_rate.setForcedValue(math::radians(tilt_deg));
	}

	/// One Run() cycle (:378-612, in-flight branch). `time_us` is the caller's microsecond counter and
	/// is compared with setpoint timestamps in the fallback logic; it also stamps the output.
	/// `setpoint` follows PX4 semantics: NaN means unused.
	PX4RL_HD Outcome update(const matrix::Vector3f &position, const matrix::Vector3f &velocity, float heading,
				float dt_s, const trajectory_setpoint_s &setpoint, uint64_t time_us,
				vehicle_attitude_setpoint_s &attitude_setpoint)
	{
		// :395-396 (the 0.002 / 0.04 clamp)
		const float dt = math::constrain(dt_s, 0.002f, 0.04f);

		PositionControlStates states{setVehicleStates(position, velocity, heading, dt)};

		_setpoint = setpoint;

		// Velocity limits of the flying state: the takeoff ramp returns the constraint, which is the
		// parameter default (:457-458, :485-487, :524-528).
		const float speed_up = _z_vel_max_up;
		const float speed_down = _z_vel_max_dn;

		// :520-522 tilt limit
		_control.setTiltLimit(_tilt_limit_slew_rate.update(math::radians(_tiltmax_air_deg), dt));

		// :530-531 (flying => MPC_THR_MIN)
		_control.setThrustLimits(_thr_min, _thr_max);

		// :533-542 (vehicle_local_position.vxy_max is NaN)
		_control.setVelocityLimits(
			_xy_vel_max,
			math::min(speed_up, _z_vel_max_up),
			math::max(speed_down, 0.f));

		_control.setInputSetpoint(_setpoint);

		// :544 setInputSetpoint, :558-564 resetIntegralXY rule
		if ((!PX4_ISFINITE(_setpoint.velocity[0]) || !PX4_ISFINITE(_setpoint.velocity[1]))
		    && (!PX4_ISFINITE(_setpoint.position[0]) || !PX4_ISFINITE(_setpoint.position[1]))) {
			// Horizontal velocity is not controlled, reset the integrators to avoid
			// over-corrections when starting again.
			_control.resetIntegralXY();
		}

		_control.setState(states);

		// :566-594 setState, update, fallback, failsafe
		Outcome outcome = Outcome::Ok;

		if (_control.update(dt)) {
			// Valid control update - store for fallback
			_last_valid_setpoint = _setpoint;

		} else {
			outcome = Outcome::Fallback;

			// Initial update failed - Try fallback if within timeout
			if (time_us < _last_valid_setpoint.timestamp + kFallbackTimeoutUs) {
				_control.setInputSetpoint(_last_valid_setpoint);
			}

			// Still failing / not within timeout - Go to failsafe
			if (!_control.update(dt)) {
				outcome = Outcome::Failsafe;

				_control.setInputSetpoint(generateFailsafeSetpoint(time_us, states));
				_control.setVelocityLimits(_xy_vel_max, _z_vel_max_up, _z_vel_max_dn);

				_control.update(dt);
			}
		}

		// :606-609 getAttitudeSetpoint
		attitude_setpoint = {};
		_control.getAttitudeSetpoint(attitude_setpoint);
		attitude_setpoint.timestamp = time_us;

		return outcome;
	}

	/// Local position setpoint of the last update() (PID corrections included); mirrors :599-603.
	PX4RL_HD void getLocalPositionSetpoint(vehicle_local_position_setpoint_s &local_position_setpoint) const
	{
		_control.getLocalPositionSetpoint(local_position_setpoint);
	}

private:
	static constexpr uint64_t kFallbackTimeoutUs = 200000; // 200_ms

	/// Mirrors set_vehicle_states (:310-375). All validity flags are true; a non-finite value is
	/// treated as invalid, as PX4 does for xy.
	PX4RL_HD PositionControlStates setVehicleStates(const matrix::Vector3f &position, const matrix::Vector3f &velocity,
							float heading, float dt_s)
	{
		PositionControlStates states;

		const matrix::Vector2f position_xy(position(0), position(1));

		if (position_xy.isAllFinite()) {
			states.position.xy() = position_xy;

		} else {
			states.position(0) = states.position(1) = NAN;
		}

		if (PX4_ISFINITE(position(2))) {
			states.position(2) = position(2);

		} else {
			states.position(2) = NAN;
		}

		const matrix::Vector2f velocity_xy(velocity(0), velocity(1));

		if (velocity_xy.isAllFinite()) {
			const matrix::Vector2f vel_xy_prev = _vel_xy_lp_filter.getState();

			// vel xy notch filter, then low pass filter
			states.velocity.xy() = _vel_xy_lp_filter.update(_vel_xy_notch_filter.apply(velocity_xy));

			// vel xy derivative low pass filter
			states.acceleration.xy() = _vel_deriv_xy_lp_filter.update((_vel_xy_lp_filter.getState() - vel_xy_prev) / dt_s);

		} else {
			states.velocity(0) = states.velocity(1) = NAN;
			states.acceleration(0) = states.acceleration(1) = NAN;

			// reset filters to prevent acceleration spikes when regaining velocity
			_vel_xy_lp_filter.reset({});
			_vel_xy_notch_filter.reset();
			_vel_deriv_xy_lp_filter.reset({});
		}

		if (PX4_ISFINITE(velocity(2))) {
			const float vel_z_prev = _vel_z_lp_filter.getState();

			// vel z notch filter, then low pass filter
			states.velocity(2) = _vel_z_lp_filter.update(_vel_z_notch_filter.apply(velocity(2)));

			// vel z derivative low pass filter
			states.acceleration(2) = _vel_deriv_z_lp_filter.update((_vel_z_lp_filter.getState() - vel_z_prev) / dt_s);

		} else {
			states.velocity(2) = NAN;
			states.acceleration(2) = NAN;

			// reset filters to prevent acceleration spikes when regaining velocity
			_vel_z_lp_filter.reset({});
			_vel_z_notch_filter.reset();
			_vel_deriv_z_lp_filter.reset({});
		}

		states.yaw = heading;

		return states;
	}

	/// Mirrors generateFailsafeSetpoint (:633-681) without the warnings.
	PX4RL_HD trajectory_setpoint_s generateFailsafeSetpoint(uint64_t now, const PositionControlStates &states) const
	{
		trajectory_setpoint_s failsafe_setpoint = emptyTrajectorySetpoint();
		failsafe_setpoint.timestamp = now;

		if (matrix::Vector2f(states.velocity).isAllFinite()) {
			// don't move along xy
			failsafe_setpoint.velocity[0] = failsafe_setpoint.velocity[1] = 0.f;

		} else {
			// descend with land speed since we can't stop
			failsafe_setpoint.acceleration[0] = failsafe_setpoint.acceleration[1] = 0.f;
			failsafe_setpoint.velocity[2] = _land_speed;
		}

		if (PX4_ISFINITE(states.velocity(2))) {
			// don't move along z if we can stop in all dimensions
			if (!PX4_ISFINITE(failsafe_setpoint.velocity[2])) {
				failsafe_setpoint.velocity[2] = 0.f;
			}

		} else {
			// emergency descend with a bit below hover thrust
			failsafe_setpoint.velocity[2] = NAN;
			failsafe_setpoint.acceleration[2] = .3f;
		}

		return failsafe_setpoint;
	}

	PositionControl _control;

	trajectory_setpoint_s _setpoint{emptyTrajectorySetpoint()};
	trajectory_setpoint_s _last_valid_setpoint{emptyTrajectorySetpoint()};

	AlphaFilter<matrix::Vector2f> _vel_xy_lp_filter{};
	AlphaFilter<float> _vel_z_lp_filter{};

	math::NotchFilter<matrix::Vector2f> _vel_xy_notch_filter{};
	math::NotchFilter<float> _vel_z_notch_filter{};

	AlphaFilter<matrix::Vector2f> _vel_deriv_xy_lp_filter{};
	AlphaFilter<float> _vel_deriv_z_lp_filter{};

	SlewRate<float> _tilt_limit_slew_rate;

	bool _hover_thrust_initialized{false};
	bool _use_hte{true};
	float _xy_vel_max{0.f};
	float _z_vel_max_up{0.f};
	float _z_vel_max_dn{0.f};
	float _land_speed{0.f};
	float _tiltmax_air_deg{0.f};
	float _thr_min{0.f};
	float _thr_max{0.f};
};

} // namespace px4rl
