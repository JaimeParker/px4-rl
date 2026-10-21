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
	float position_dt;       /* s, nominal interval between position_step calls (50 Hz = 0.02). It fixes the
	                          * velocity filter coefficients at reset, as PX4's measured mean sample interval
	                          * does; the dt passed to each position_step call only scales that step. */
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

/* Per-vehicle result of position_step (px4rl_cuda_position_step and CpuFleet::positionStep). */
#define PX4RL_OUTCOME_OK 0       /* the commanded setpoint was executed */
#define PX4RL_OUTCOME_FALLBACK 1 /* it was invalid; the last valid one (less than 200 ms old) was executed */
#define PX4RL_OUTCOME_FAILSAFE 2 /* neither was usable; PX4's failsafe setpoint was executed */

/* Fills `p` from the px4_stack struct defaults, geometry createQuadXIris(). */
PX4RL_API void px4rl_params_default(px4rl_params *p);

/* Fills `p` with the controller parameters PX4 SITL flies the Gazebo Classic Iris with: the defaults of
 * px4rl_params_default plus the CA_ROTOR* geometry of PX4's 10015_gazebo-classic_iris airframe. See
 * docs/iris-params-sources.md and docs/architecture.md ("Iris airframe"). The plant side is
 * px4rl_plant_params_iris() in airframes.h. */
PX4RL_API void px4rl_params_iris(px4rl_params *p);

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

/* Position / velocity loop (PX4 MulticopterPositionControl and the hover thrust estimator).
 *
 * Inputs, per vehicle, in PX4's frames (NED world, FRD body):
 *   pos, vel      local position [m] and velocity [m/s]
 *   heading       yaw angle [rad]
 *   az            vertical acceleration [m/s^2], NED (positive down)
 *   traj_sp       n x 11 floats: position[3] velocity[3] acceleration[3] yaw yawspeed.
 *                 NaN means "unused", with PX4 semantics. Command modes by convention:
 *                   vel + yaw:  (NaN x3, vx vy vz, NaN x3, yaw, NaN)
 *                   acc + yaw:  (NaN x3, NaN x3, ax ay az, yaw, NaN)
 *                   pos + yaw:  (px py pz, NaN x3, NaN x3, yaw, NaN)
 *   in_air        uint8[n] or NULL (= all in air). Gates the hover thrust estimator only.
 *   dt            seconds since the previous position_step call; clamped to [0.002, 0.04] by the controller.
 *   time_us       the caller's monotonically increasing microsecond clock; it stamps the setpoint, so the
 *                 200 ms fallback to the last valid setpoint is measured on it.
 * The nominal call interval is px4rl_params.position_dt, set through reset.
 *
 * Outputs, per vehicle:
 *   q_sp_out            attitude setpoint (w, x, y, z), body FRD to NED
 *   thrust_sp_body_out  normalized thrust, body FRD (0, 0, -t)
 *   yawspeed_out        yaw rate feed-forward [rad/s]
 *   hover_thrust_out    current hover thrust estimate, or NULL
 *   outcome_out         PX4RL_OUTCOME_* per vehicle, or NULL
 *
 * The estimator uses the attitude seen by the last attitude_step and the thrust of the last rate_step, so it
 * starts after the first rate_step. The attitude setpoint is also stored: attitude_step(q_sp = NULL)
 * consumes it. Command modes that bypass this call: CTA calls attitude_step with an explicit q_sp, CTBR calls
 * rate_step directly, and direct motor commands do not involve px4-rl. */
PX4RL_API int px4rl_cuda_position_step(px4rl_cuda_fleet *f,
				       const float *pos /*n,3*/, const float *vel /*n,3*/,
				       const float *heading /*n*/, const float *az /*n*/,
				       const float *traj_sp /*n,11*/, const uint8_t *in_air /*n or NULL*/,
				       float dt, uint64_t time_us,
				       float *q_sp_out /*n,4*/, float *thrust_sp_body_out /*n,3*/,
				       float *yawspeed_out /*n*/, float *hover_thrust_out /*n or NULL*/,
				       uint8_t *outcome_out /*n or NULL*/, void *stream);

/* q_sp = NULL selects the attitude setpoint stored by the last position_step (level, zero yaw rate before
 * the first one) together with its yaw rate; yawspeed_ff is then ignored. With an explicit q_sp,
 * yawspeed_ff (n or NULL = 0) is the yaw rate feed-forward. thrust_sp_body is always required. */
PX4RL_API int px4rl_cuda_attitude_step(px4rl_cuda_fleet *f,
				       const float *q /*n,4*/, const float *q_sp /*n,4 or NULL*/,
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

/* ---- Rotor plant (Gazebo Classic motor model) ----------------------------
 * Every vehicle of the fleet also owns a plant: px4rl_plant_params (plant.h) and the speed state of its
 * rotors. A new fleet holds the Iris plant (px4rl_plant_params_iris()) with the rotors at rest. The
 * controller reset above leaves the plant untouched; px4rl_cuda_fleet_reset_plant replaces it. */

struct px4rl_plant_params; /* defined in plant.h */

/* plant_host: host array of n_plant structs, n_plant == 1 (broadcast) or n. mask: device uint8[n] or NULL
 * (= all). Replaces the plant of the selected vehicles and puts their rotors at rest. */
PX4RL_API int px4rl_cuda_fleet_reset_plant(px4rl_cuda_fleet *f, const uint8_t *mask,
					   const struct px4rl_plant_params *plant_host, int n_plant, void *stream);

/* Advances every vehicle's rotors by dt and returns the wrench on the body. FLU is the Gazebo / Isaac Lab
 * body frame (x forward, y left, z up).
 *
 *   actuator            n x 4 PWM values by PX4 output index, as produced by rate_step's actuator_out. They
 *                       are scaled to the actuator value u = (pwm - 1000) / (2000 - 1000), the scaling of PX4's
 *                       simulation output driver, fixed at 1000 and 2000 whatever px4rl_params.pwm_min and
 *                       pwm_max are. Each plant rotor reads the output named by its px4_output_index.
 *   armed               uint8[n] or NULL (= all armed). A disarmed vehicle commands the rotors'
 *                       zero_position_disarmed speed; PX4's driver does not scale its disarmed PWM (900).
 *   body_velocity_flu   n x 3 velocity of the body relative to the air, body FLU (wind 0).
 *   body_rates_flu      n x 3 body angular velocity, body FLU, or NULL. If given, each rotor's air velocity
 *                       includes omega x r; if NULL the rotors see the body velocity.
 *   dt                  seconds, > 0.
 * Outputs, body FLU:
 *   force_flu_out       n x 3 total force on the body, N
 *   torque_flu_out      n x 3 total moment about the body origin (the centre of mass), N m
 *   rotor_force_flu_out n x 4 x 3 force of each plant rotor (zero beyond num_rotors), or NULL
 * Gravity and rigid-body integration are the caller's. Forces are computed from the rotor speeds before the
 * step, then the speeds advance, as in Gazebo's UpdateForcesAndMoments. */
PX4RL_API int px4rl_cuda_rotor_step(px4rl_cuda_fleet *f,
				    const float *actuator /*n,4*/, const uint8_t *armed /*n or NULL*/,
				    const float *body_velocity_flu /*n,3*/, const float *body_rates_flu /*n,3 or NULL*/,
				    float dt,
				    float *force_flu_out /*n,3*/, float *torque_flu_out /*n,3*/,
				    float *rotor_force_flu_out /*n,4,3 or NULL*/, void *stream);

#ifdef __cplusplus
}
#endif

#endif /* PX4RL_C_API_H */
