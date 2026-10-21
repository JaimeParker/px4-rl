/* px4rl airframe library. Each function returns the plant of one airframe as C PODs. */
#ifndef PX4RL_AIRFRAMES_H
#define PX4RL_AIRFRAMES_H

#include "px4rl/c_api.h"
#include "px4rl/plant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gazebo Classic Iris (PX4-SITL_gazebo-classic @ 70683dc, models/iris/iris.sdf.jinja).
 * rotor[i] is SDF rotor_i and is driven by PX4 output i. */
PX4RL_API px4rl_plant_params px4rl_plant_params_iris(void);
PX4RL_API px4rl_rigid_body_params px4rl_rigid_body_params_iris(void);

/* Hover thrust in PX4's MPC_THR_HOVER units (normalized collective thrust) of an upright vehicle in steady
 * flight: armed, zero airspeed, every rotor at its reference speed. The model is documented in
 * hover_equilibrium.hpp. Use it to seed px4rl_params.mpc_thr_hover so an episode can start in steady flight.
 * Rotor thrust is k_f w^2 along the rotor axis, with w from the actuator value u as in the Gazebo motor model;
 * the u that balances body->mass * gravity is mapped through the forward thrust curve with params->thr_mdl_fac.
 * PWM quantization is ignored.
 * Returns NAN if the vehicle cannot hover (no u in [0, 1] balances the weight), an input is invalid, or a
 * pointer is NULL. The CPU library only returns NAN; libpx4rl_cuda.so also describes the reason with
 * px4rl_last_error(). */
PX4RL_API float px4rl_hover_thrust(const px4rl_params *params, const px4rl_plant_params *plant,
				   const px4rl_rigid_body_params *body, float gravity);

#ifdef __cplusplus
}
#endif

#endif /* PX4RL_AIRFRAMES_H */
