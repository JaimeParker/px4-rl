/* px4rl C API: parameter struct shared by all backends and the CUDA fleet entry points.
 * Plain C; no C++ types appear in the public signatures. */
#ifndef PX4RL_C_API_H
#define PX4RL_C_API_H

#include <stdint.h>

#if defined(__GNUC__)
#define PX4RL_API __attribute__((visibility("default")))
#else
#define PX4RL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PX4RL_MAX_ROTORS 4

/* Flat parameter set for one vehicle. Units follow the px4_stack structs. */
typedef struct px4rl_params {
	/* AttitudeControlConfig */
	float att_roll_p;
	float att_pitch_p;
	float att_yaw_p;
	float att_yaw_weight;
	float att_max_roll_rate;   /* rad/s */
	float att_max_pitch_rate;  /* rad/s */
	float att_max_yaw_rate;    /* rad/s */

	/* RateControlParams; arrays are [roll, pitch, yaw] */
	float rate_p[3];
	float rate_i[3];
	float rate_d[3];
	float rate_ff[3];
	float rate_k[3];
	float rate_integrator_limit[3];
	float rate_yaw_torque_cutoff_hz;

	/* ControlAllocatorParams: geometry */
	int32_t num_rotors; /* 1..PX4RL_MAX_ROTORS */
	float rotor_position[PX4RL_MAX_ROTORS][3];
	float rotor_axis[PX4RL_MAX_ROTORS][3];
	float rotor_ct[PX4RL_MAX_ROTORS];
	float rotor_km[PX4RL_MAX_ROTORS];
	int32_t propeller_torque_disabled;
	int32_t yaw_by_differential_thrust_disabled;
	int32_t propeller_torque_disabled_non_upwards;
	int32_t three_dimensional_thrust_disabled;
	/* ControlAllocatorParams: allocation */
	int32_t airmode; /* 0 disabled, 1 RP, 2 RPY */
	int32_t normalize_rpy;
	float actuator_min[PX4RL_MAX_ROTORS];
	float actuator_max[PX4RL_MAX_ROTORS];
	float actuator_slew_rate[PX4RL_MAX_ROTORS]; /* seconds from 0 to full, 0 = unlimited */

	/* ActuatorOutputParams */
	float thr_mdl_fac;
	int32_t output_mode; /* 0 = PWM, 1 = DShot (px4_stack::OutputMode) */
	float pwm_min;
	float pwm_max;
	float pwm_disarmed;
	float dshot_min_throttle;
	float dshot_max_throttle;
	float dshot_disarmed;
	float dshot_min_fraction;

	/* PositionModule: PX4 MPC_* parameters, in PX4's units. */
	float mpc_xy_p;
	float mpc_z_p;
	float mpc_xy_vel_p_acc;
	float mpc_xy_vel_i_acc;
	float mpc_xy_vel_d_acc;
	float mpc_z_vel_p_acc;
	float mpc_z_vel_i_acc;
	float mpc_z_vel_d_acc;
	float mpc_xy_vel_max;    /* m/s */
	float mpc_z_vel_max_up;  /* m/s */
	float mpc_z_vel_max_dn;  /* m/s */
	float mpc_land_speed;    /* m/s, descent speed of the blind-land failsafe */
	float mpc_tiltmax_air;   /* degrees */
	float mpc_thr_min;
	float mpc_thr_max;
	float mpc_thr_hover;
	float mpc_thr_xy_marg;
	int32_t mpc_use_hte;     /* 1 = apply the hover thrust estimate */
	int32_t mpc_acc_decouple; /* 1 = tilt setpoint ignores the vertical acceleration setpoint */
	float mpc_vel_lp;        /* Hz, <= 0 disables */
	float mpc_veld_lp;       /* Hz, <= 0 disables */
	float mpc_vel_nf_frq;    /* Hz, <= 0 disables the notch */
	float mpc_vel_nf_bw;     /* Hz */

	/* HoverThrustModule: PX4 HTE_* parameters. */
	float hte_ht_noise;
	float hte_ht_err_init;
	float hte_acc_gate;
	float hte_thr_range;
	float hte_vxy_thr;       /* m/s */
	float hte_vz_thr;        /* m/s */
} px4rl_params;

/* Fills `p` from the px4_stack struct defaults, geometry createQuadXIris(). */
PX4RL_API void px4rl_params_default(px4rl_params *p);

/* ---- CUDA fleet (libpx4rl_cuda.so) ------------------------------------
 * Every call returns 0 on success. px4rl_last_error() describes the last
 * failure of the calling thread. `stream` is a cudaStream_t (NULL = default
 * stream). Array arguments are device pointers to contiguous row-major
 * float32 (or uint8) buffers, except params_host. */

typedef struct px4rl_cuda_fleet px4rl_cuda_fleet;

PX4RL_API const char *px4rl_last_error(void);

PX4RL_API int px4rl_cuda_fleet_create(int n, int device, px4rl_cuda_fleet **out);
PX4RL_API int px4rl_cuda_fleet_destroy(px4rl_cuda_fleet *f);
/* Number of vehicles, or -1 on error. */
PX4RL_API int px4rl_cuda_fleet_size(const px4rl_cuda_fleet *f);

/* params_host: host array of n_params structs, n_params == 1 (broadcast) or n.
 * mask: device uint8[n] or NULL (= all). Re-constructs the selected vehicles. */
PX4RL_API int px4rl_cuda_fleet_reset(px4rl_cuda_fleet *f, const uint8_t *mask,
				     const px4rl_params *params_host, int n_params, void *stream);

PX4RL_API int px4rl_cuda_attitude_step(px4rl_cuda_fleet *f,
				       const float *q /*n,4*/, const float *q_sp /*n,4*/,
				       const float *yawspeed_ff /*n or NULL*/,
				       const float *thrust_sp_body /*n,3*/,
				       float *rates_sp_out /*n,3*/, float *thrust_body_out /*n,3*/,
				       void *stream);

PX4RL_API int px4rl_cuda_rate_step(px4rl_cuda_fleet *f,
				   const float *rates_sp /*n,3*/, const float *thrust_body /*n,3*/,
				   const float *rates /*n,3*/, const float *angular_accel /*n,3*/,
				   float dt, const uint8_t *landed /*n or NULL*/,
				   float *motors_out /*n,4 normalized [0,1]*/,
				   float *actuator_out /*n,4 PWM/DShot value or NULL*/,
				   float *torque_sp_out /*n,3 or NULL*/, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* PX4RL_C_API_H */
