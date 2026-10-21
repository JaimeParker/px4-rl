/* px4-rl shim for PX4's platforms/common/include/px4_defines.h (transitional header used by the
 * upstream tests). Forwards to the px4-rl defines.h shim and adds the float math constants that the
 * upstream defines.h provides and the px4-rl defines.h shim lacks (needed by NotchFilter.hpp and the
 * upstream ControlMath/PositionControl tests). Values copied verbatim from
 * platforms/common/include/px4_platform_common/defines.h:119-146. Each macro is guarded so that a
 * future defines.h that defines it is not redefined. */
#pragma once

#include <px4_platform_common/defines.h>

#ifndef M_E_F
#define M_E_F			2.71828183f
#endif
#ifndef M_LOG2E_F
#define M_LOG2E_F		1.44269504f
#endif
#ifndef M_LOG10E_F
#define M_LOG10E_F		0.43429448f
#endif
#ifndef M_LN2_F
#define M_LN2_F			0.69314718f
#endif
#ifndef M_LN10_F
#define M_LN10_F		2.30258509f
#endif
#ifndef M_PI_F
#define M_PI_F			3.14159265f
#endif
#ifndef M_TWOPI_F
#define M_TWOPI_F		6.28318531f
#endif
#ifndef M_PI_2_F
#define M_PI_2_F		1.57079632f
#endif
#ifndef M_PI_4_F
#define M_PI_4_F		0.78539816f
#endif
#ifndef M_3PI_4_F
#define M_3PI_4_F		2.35619449f
#endif
#ifndef M_SQRTPI_F
#define M_SQRTPI_F		1.77245385f
#endif
#ifndef M_1_PI_F
#define M_1_PI_F		0.31830989f
#endif
#ifndef M_2_PI_F
#define M_2_PI_F		0.63661977f
#endif
#ifndef M_2_SQRTPI_F
#define M_2_SQRTPI_F		1.12837917f
#endif
#ifndef M_DEG_TO_RAD_F
#define M_DEG_TO_RAD_F		0.0174532925f
#endif
#ifndef M_RAD_TO_DEG_F
#define M_RAD_TO_DEG_F		57.2957795f
#endif
#ifndef M_SQRT2_F
#define M_SQRT2_F		1.41421356f
#endif
#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F		0.70710678f
#endif
#ifndef M_LN2LO_F
#define M_LN2LO_F		1.90821484E-10f
#endif
#ifndef M_LN2HI_F
#define M_LN2HI_F		0.69314718f
#endif
#ifndef M_SQRT3_F
#define M_SQRT3_F		1.73205081f
#endif
#ifndef M_IVLN10_F
#define M_IVLN10_F		0.43429448f	// 1 / log(10)
#endif
#ifndef M_LOG2_E_F
#define M_LOG2_E_F		0.69314718f
#endif
#ifndef M_INVLN2_F
#define M_INVLN2_F		1.44269504f	// 1 / log(2)
#endif
#ifndef M_DEG_TO_RAD
#define M_DEG_TO_RAD 		0.017453292519943295
#endif
#ifndef M_RAD_TO_DEG
#define M_RAD_TO_DEG 		57.295779513082323
#endif
