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

#ifdef __cplusplus
}
#endif

#endif /* PX4RL_AIRFRAMES_H */
