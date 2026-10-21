#include "px4rl/c_api.h"
#include "px4rl/params.hpp"

extern "C" void px4rl_params_default(px4rl_params *p)
{
	*p = px4rl::defaultParams();
}
