#include "px4rl/airframes.h"
#include "px4rl/hover_equilibrium.hpp"

extern "C" float px4rl_hover_thrust(const px4rl_params *params, const px4rl_plant_params *plant,
				    const px4rl_rigid_body_params *body, float gravity)
{
	if (params == nullptr || plant == nullptr || body == nullptr) {
		return std::nanf("");
	}

	return px4rl::hoverThrust(*params, *plant, *body, gravity);
}
