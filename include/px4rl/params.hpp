// Conversion between the flat C parameter struct and the px4_stack parameter structs.
// Defaults are read from the px4_stack struct initialisers; no numbers are repeated here.
#pragma once

#include <AttitudeControlTypes.hpp>
#include <ActuatorOutputTypes.hpp>
#include <ControlAllocatorTypes.hpp>
#include <RateControlTypes.hpp>

#include "px4rl/c_api.h"

namespace px4rl {

namespace detail {

PX4RL_HD inline void store(float *dst, const matrix::Vector3f &v)
{
	dst[0] = v(0);
	dst[1] = v(1);
	dst[2] = v(2);
}

PX4RL_HD inline matrix::Vector3f load(const float *src)
{
	return matrix::Vector3f(src[0], src[1], src[2]);
}

} // namespace detail

PX4RL_HD inline px4_stack::AttitudeControlConfig toAttitudeConfig(const px4rl_params &p)
{
	px4_stack::AttitudeControlConfig c;
	c.roll_p = p.att_roll_p;
	c.pitch_p = p.att_pitch_p;
	c.yaw_p = p.att_yaw_p;
	c.yaw_weight = p.att_yaw_weight;
	c.max_roll_rate = p.att_max_roll_rate;
	c.max_pitch_rate = p.att_max_pitch_rate;
	c.max_yaw_rate = p.att_max_yaw_rate;
	return c;
}

PX4RL_HD inline px4_stack::RateControlParams toRateParams(const px4rl_params &p)
{
	px4_stack::RateControlParams r;
	r.gain_p = detail::load(p.rate_p);
	r.gain_i = detail::load(p.rate_i);
	r.gain_d = detail::load(p.rate_d);
	r.gain_ff = detail::load(p.rate_ff);
	r.gain_k = detail::load(p.rate_k);
	r.integrator_limit = detail::load(p.rate_integrator_limit);
	r.yaw_torque_lpf_cutoff_freq = p.rate_yaw_torque_cutoff_hz;
	return r;
}

PX4RL_HD inline px4_stack::ControlAllocatorParams toAllocatorParams(const px4rl_params &p)
{
	px4_stack::ControlAllocatorParams a;
	px4_stack::Geometry &g = a.geometry;
	g.num_rotors = p.num_rotors;
	for (int i = 0; i < PX4RL_MAX_ROTORS; ++i) {
		g.rotors[i].position = detail::load(p.rotor_position[i]);
		g.rotors[i].axis = detail::load(p.rotor_axis[i]);
		g.rotors[i].thrust_coef = p.rotor_ct[i];
		g.rotors[i].moment_ratio = p.rotor_km[i];
		a.actuator_min[i] = p.actuator_min[i];
		a.actuator_max[i] = p.actuator_max[i];
		a.slew_rate_limit[i] = p.actuator_slew_rate[i];
	}
	g.propeller_torque_disabled = p.propeller_torque_disabled != 0;
	g.yaw_by_differential_thrust_disabled = p.yaw_by_differential_thrust_disabled != 0;
	g.propeller_torque_disabled_non_upwards = p.propeller_torque_disabled_non_upwards != 0;
	g.three_dimensional_thrust_disabled = p.three_dimensional_thrust_disabled != 0;
	a.airmode = p.airmode;
	a.normalize_rpy = p.normalize_rpy != 0;
	return a;
}

PX4RL_HD inline px4_stack::ActuatorOutputParams toOutputParams(const px4rl_params &p)
{
	px4_stack::ActuatorOutputParams o;
	o.thrust_model_factor = p.thr_mdl_fac;
	o.mode = static_cast<px4_stack::OutputMode>(p.output_mode);
	o.pwm_min = static_cast<uint16_t>(p.pwm_min);
	o.pwm_max = static_cast<uint16_t>(p.pwm_max);
	o.pwm_disarmed = static_cast<uint16_t>(p.pwm_disarmed);
	o.dshot_min_throttle = static_cast<uint16_t>(p.dshot_min_throttle);
	o.dshot_max_throttle = static_cast<uint16_t>(p.dshot_max_throttle);
	o.dshot_disarmed = static_cast<uint16_t>(p.dshot_disarmed);
	o.dshot_min_fraction = p.dshot_min_fraction;
	return o;
}

// Every field is filled from the corresponding px4_stack struct default.
PX4RL_HD inline px4rl_params defaultParams()
{
	px4rl_params p{};

	const px4_stack::AttitudeControlConfig att;
	p.att_roll_p = att.roll_p;
	p.att_pitch_p = att.pitch_p;
	p.att_yaw_p = att.yaw_p;
	p.att_yaw_weight = att.yaw_weight;
	p.att_max_roll_rate = att.max_roll_rate;
	p.att_max_pitch_rate = att.max_pitch_rate;
	p.att_max_yaw_rate = att.max_yaw_rate;

	const px4_stack::RateControlParams rate;
	detail::store(p.rate_p, rate.gain_p);
	detail::store(p.rate_i, rate.gain_i);
	detail::store(p.rate_d, rate.gain_d);
	detail::store(p.rate_ff, rate.gain_ff);
	detail::store(p.rate_k, rate.gain_k);
	detail::store(p.rate_integrator_limit, rate.integrator_limit);
	p.rate_yaw_torque_cutoff_hz = rate.yaw_torque_lpf_cutoff_freq;

	const px4_stack::ControlAllocatorParams alloc;
	const px4_stack::Geometry &g = alloc.geometry;
	p.num_rotors = g.num_rotors;
	for (int i = 0; i < PX4RL_MAX_ROTORS; ++i) {
		detail::store(p.rotor_position[i], g.rotors[i].position);
		detail::store(p.rotor_axis[i], g.rotors[i].axis);
		p.rotor_ct[i] = g.rotors[i].thrust_coef;
		p.rotor_km[i] = g.rotors[i].moment_ratio;
		p.actuator_min[i] = alloc.actuator_min[i];
		p.actuator_max[i] = alloc.actuator_max[i];
		p.actuator_slew_rate[i] = alloc.slew_rate_limit[i];
	}
	p.propeller_torque_disabled = g.propeller_torque_disabled;
	p.yaw_by_differential_thrust_disabled = g.yaw_by_differential_thrust_disabled;
	p.propeller_torque_disabled_non_upwards = g.propeller_torque_disabled_non_upwards;
	p.three_dimensional_thrust_disabled = g.three_dimensional_thrust_disabled;
	// PX4's MC_AIRMODE default is 0 (src/lib/mixer_module/params.c:18, docs/iris-params-sources.md section 4).
	// px4_stack's wrapper defaults to 1 (roll/pitch airmode), so the PX4 value is set here explicitly.
	p.airmode = 0;
	p.normalize_rpy = alloc.normalize_rpy;

	const px4_stack::ActuatorOutputParams out;
	p.thr_mdl_fac = out.thrust_model_factor;
	p.output_mode = static_cast<int32_t>(out.mode);
	p.pwm_min = static_cast<float>(out.pwm_min);
	p.pwm_max = static_cast<float>(out.pwm_max);
	p.pwm_disarmed = static_cast<float>(out.pwm_disarmed);
	p.dshot_min_throttle = static_cast<float>(out.dshot_min_throttle);
	p.dshot_max_throttle = static_cast<float>(out.dshot_max_throttle);
	p.dshot_disarmed = static_cast<float>(out.dshot_disarmed);
	p.dshot_min_fraction = out.dshot_min_fraction;

	p.position_dt = 0.02f; // 50 Hz; PX4 seeds its running mean of the interval with 0.01 s

	// PX4 v1.16.2 parameter defaults, docs/iris-params-sources.md section 4 (source file:line there).
	// They are not held in a px4_stack struct, so this is the one place they are written down.
	p.mpc_xy_p = 0.95f;
	p.mpc_z_p = 1.f;
	p.mpc_xy_vel_p_acc = 1.8f;
	p.mpc_xy_vel_i_acc = 0.4f;
	p.mpc_xy_vel_d_acc = 0.2f;
	p.mpc_z_vel_p_acc = 4.f;
	p.mpc_z_vel_i_acc = 2.f;
	p.mpc_z_vel_d_acc = 0.f;
	p.mpc_xy_vel_max = 12.f;
	p.mpc_z_vel_max_up = 3.f;
	p.mpc_z_vel_max_dn = 1.5f;
	// MPC_LAND_SPEED is not in the doc table; value from
	// src/modules/mc_pos_control/multicopter_takeoff_land_params.c:111.
	p.mpc_land_speed = 0.7f;
	p.mpc_tiltmax_air = 45.f;
	p.mpc_thr_min = 0.12f;
	p.mpc_thr_max = 1.f;
	p.mpc_thr_hover = 0.5f;
	p.mpc_thr_xy_marg = 0.3f;
	p.mpc_use_hte = 1;
	// MPC_ACC_DECOUPLE is not in the doc table; value from
	// src/modules/mc_pos_control/multicopter_position_control_limits_params.c:152.
	p.mpc_acc_decouple = 1;
	p.mpc_vel_lp = 0.f;
	p.mpc_veld_lp = 5.f;
	p.mpc_vel_nf_frq = 0.f;
	p.mpc_vel_nf_bw = 5.f;
	p.hte_ht_noise = 0.0036f;
	p.hte_ht_err_init = 0.1f;
	p.hte_acc_gate = 3.f;
	p.hte_thr_range = 0.2f;
	p.hte_vxy_thr = 10.f;
	p.hte_vz_thr = 2.f;
	return p;
}

// Controller parameters of PX4 SITL's Gazebo Classic Iris: defaultParams() plus what airframe
// ROMFS/px4fmu_common/init.d-posix/airframes/10015_gazebo-classic_iris (and the rc.mc_defaults it sources)
// sets for the controller chain; all lines are tabulated in docs/iris-params-sources.md, sections 3 and 5.
//
// Applied:
//   CA_ROTOR0..3_PX / PY / KM (airframe lines 16-27) -> rotor_position x / y, rotor_km. The airframe leaves
//   PZ, AX, AY, AZ and CT at the control allocator's defaults (module.yaml:160, 172, 183, 194, 209): PZ = 0,
//   axis (0, 0, -1) (up in FRD), CT = 6.5.
// Same as createQuadXIris(), which defaultParams() uses: every position, KM, axis and num_rotors. The only
// difference is CT, 1.0 there against 6.5 here. The allocator normalises its mix matrix, so a uniform CT
// cancels out (tests/fleet checks that the motor outputs agree).
// Implied by the above: CA_AIRFRAME 0 (multirotor), CA_ROTOR_COUNT 4, and PWM_MAIN_FUNC1-4 = 101-104, which
// connect output i to motor i, the order of rotor_position[].
// Ignored, with no effect on the controller chain: MAV_TYPE, IMU_GYRO_RATEMAX (the caller sets the loop
// rates), NAV_ACC_RAD, RTL_RETURN_ALT, RTL_DESCEND_ALT, GPS_UBX_DYNMODEL, EKF2_RNG_FOG (rc.mc_defaults) and
// MC_AT_EN (init.d-posix/rcS:178, autotune).
// No MC_* / MPC_* / HTE_* / THR_MDL_FAC override exists for this airframe: they keep PX4's defaults, as held
// in defaultParams().
PX4RL_HD inline px4rl_params irisParams()
{
	px4rl_params p = defaultParams();

	const float position_x[4] = {0.1515f, -0.1515f, 0.1515f, -0.1515f};  // CA_ROTOR{0..3}_PX
	const float position_y[4] = {0.245f, -0.1875f, -0.245f, 0.1875f};    // CA_ROTOR{0..3}_PY
	const float moment_ratio[4] = {0.05f, 0.05f, -0.05f, -0.05f};        // CA_ROTOR{0..3}_KM
	constexpr float kThrustCoefficient = 6.5f;                           // CA_ROTOR${i}_CT, module.yaml:209

	p.num_rotors = 4;

	for (int i = 0; i < 4; ++i) {
		p.rotor_position[i][0] = position_x[i];
		p.rotor_position[i][1] = position_y[i];
		p.rotor_position[i][2] = 0.f;
		p.rotor_axis[i][0] = 0.f;
		p.rotor_axis[i][1] = 0.f;
		p.rotor_axis[i][2] = -1.f;
		p.rotor_ct[i] = kThrustCoefficient;
		p.rotor_km[i] = moment_ratio[i];
	}

	return p;
}

// Returns nullptr when `p` can be configured, otherwise a static message.
inline const char *validateParams(const px4rl_params &p)
{
	if (p.num_rotors < 1 || p.num_rotors > PX4RL_MAX_ROTORS) {
		return "num_rotors must be in [1, PX4RL_MAX_ROTORS]";
	}

	if (p.airmode < 0 || p.airmode > 2) {
		return "airmode must be 0, 1 or 2";
	}

	if (p.output_mode != 0 && p.output_mode != 1) {
		return "output_mode must be 0 (PWM) or 1 (DShot)";
	}

	if (!(p.position_dt > 0.f)) {
		return "position_dt must be > 0";
	}

	return nullptr;
}

} // namespace px4rl
