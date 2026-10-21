/* px4rl plant parameters: C POD description of a multirotor's rotors for the Gazebo-faithful
 * rotor model (gazebo_motor_model.hpp). Plain C; floats, as everywhere in px4-rl (Gazebo itself
 * uses double). Frames: body FLU (x forward, y left, z up), the Gazebo / Isaac Lab convention. */
#ifndef PX4RL_PLANT_H
#define PX4RL_PLANT_H

#include <stdint.h>

#include "px4rl/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One rotor: the motor-plugin parameters (libgazebo_motor_model.so) plus the MAVLink channel that
 * feeds it (libgazebo_mavlink_interface). Field names follow the SDF element names. */
typedef struct px4rl_rotor_params {
	float position[3];  /* rotor link origin in the body FLU frame, m (SDF <link name='rotor_i'><pose>) */
	float axis[3];      /* unit rotor axis in the body FLU frame (SDF joint <axis>, rotor link z) */
	int32_t turning_direction; /* +1 ccw, -1 cw (<turningDirection>) */
	int32_t reversible;        /* nonzero: negative thrust allowed (<reversible>) */

	float motor_constant;      /* k_f, N/(rad/s)^2 (<motorConstant>) */
	float moment_constant;     /* k_m, m (<momentConstant>) */
	float time_constant_up;    /* s (<timeConstantUp>) */
	float time_constant_down;  /* s (<timeConstantDown>) */
	float max_rot_velocity;    /* rad/s (<maxRotVelocity>) */
	float rotor_drag_coefficient;       /* <rotorDragCoefficient> */
	float rolling_moment_coefficient;   /* <rollingMomentCoefficient> */
	float max_relative_airspeed;        /* m/s, default 25 (gazebo_motor_model.h) */

	/* MAVLink channel (<control_channels><channel>): omega_ref = (u + input_offset) * input_scaling
	 * + zero_position_armed when armed, zero_position_disarmed otherwise. */
	float input_offset;
	float input_scaling;
	float zero_position_armed;
	float zero_position_disarmed;
	/* PX4 output (actuator_outputs index) that drives this rotor. */
	int32_t px4_output_index;
} px4rl_rotor_params;

typedef struct px4rl_plant_params {
	int32_t num_rotors; /* 1..PX4RL_MAX_ROTORS */
	px4rl_rotor_params rotor[PX4RL_MAX_ROTORS];
} px4rl_plant_params;

/* Rigid-body data of the airframe, about the body-frame origin (= centre of mass), body FLU. */
typedef struct px4rl_rigid_body_params {
	float mass;        /* kg */
	float inertia[9];  /* row-major 3x3, kg m^2 */
} px4rl_rigid_body_params;

#ifdef __cplusplus
}
#endif

#endif /* PX4RL_PLANT_H */
