/* px4-rl shim for PX4 v1.16.2 src/drivers/drv_hrt.h: only the time type, used by lib/hysteresis.
 * Copied verbatim from drv_hrt.h:62. The hrt_* clock functions are not provided; callers pass time
 * in microseconds explicitly. */
#pragma once

#include <stdint.h>

typedef uint64_t	hrt_abstime;
