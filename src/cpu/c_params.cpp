#include "px4rl/c_api.h"
#include "px4rl/params.hpp"
#include "px4rl/plant.h"

extern "C" void px4rl_params_default(px4rl_params *p)
{
	*p = px4rl::defaultParams();
}

extern "C" size_t px4rl_params_size(void)
{
	return sizeof(px4rl_params);
}

extern "C" size_t px4rl_plant_params_size(void)
{
	return sizeof(px4rl_plant_params);
}

extern "C" size_t px4rl_rigid_body_params_size(void)
{
	return sizeof(px4rl_rigid_body_params);
}
