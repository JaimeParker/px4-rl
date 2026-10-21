/****************************************************************************
 *
 *   Copyright (c) 2024-2026 Zhaohong Liu. All rights reserved.
 *
 * Minimal replacement for px4_platform_common/defines.h
 * Contains only the defines needed by matrix library and mathlib
 *
 ****************************************************************************/

#pragma once

#include <cmath>

// High precision Pi constant
#define M_PI_PRECISE        3.141592653589793238462643383279502884

// Two-pi constant (used by AlphaFilter)
#ifndef M_TWOPI_F
#define M_TWOPI_F           6.28318531f
#endif

// Finite check macros
PX4RL_HD static inline constexpr bool PX4_ISFINITE(float x) { return std::isfinite(x); }
PX4RL_HD static inline constexpr bool PX4_ISFINITE(double x) { return std::isfinite(x); }

// Error print usable from host and device code (px4-rl probe addition).
#include <cstdio>
#if defined(__CUDA_ARCH__)
#define PX4_ERR(...) printf(__VA_ARGS__)
#else
#define PX4_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif
