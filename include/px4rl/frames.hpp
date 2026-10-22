// Conversions between the Gazebo / Isaac Lab convention (world ENU, body FLU) and PX4's (world NED,
// body FRD). Mirrors the frame notes of PX4-SITL_gazebo-classic@70683dc include/common.h:213-237
// (q_ENU_to_NED = (0, 0.70711, 0.70711, 0), q_FLU_to_FRD = (0, 1, 0, 0), both symmetric), with the
// exact sqrt(1/2) instead of the rounded 0.70711.
//
// Quaternions are Hamilton, stored (w, x, y, z), as matrix::Quatf. "world_body" means the
// quaternion that rotates body-frame vectors into the world frame (v_world = q * v_body * q^-1),
// "body_world" the inverse rotation.
#pragma once

#include <cmath>

#include <matrix/matrix/math.hpp>

namespace px4rl {

/// ENU -> NED, the map is its own inverse: (x, y, z) -> (y, x, -z). Use for world-frame vectors
/// and world-frame angular velocity.
PX4RL_HD inline matrix::Vector3f enuToNed(const matrix::Vector3f &v)
{
	return matrix::Vector3f(v(1), v(0), -v(2));
}

/// NED -> ENU, same map as enuToNed.
PX4RL_HD inline matrix::Vector3f nedToEnu(const matrix::Vector3f &v)
{
	return enuToNed(v);
}

/// FLU -> FRD, the map is its own inverse: (x, y, z) -> (x, -y, -z). Use for body-frame vectors
/// and body angular rates.
PX4RL_HD inline matrix::Vector3f fluToFrd(const matrix::Vector3f &v)
{
	return matrix::Vector3f(v(0), -v(1), -v(2));
}

/// FRD -> FLU, same map as fluToFrd.
PX4RL_HD inline matrix::Vector3f frdToFlu(const matrix::Vector3f &v)
{
	return fluToFrd(v);
}

namespace detail {

PX4RL_HD inline matrix::Quatf conjugate(const matrix::Quatf &q)
{
	return matrix::Quatf(q(0), -q(1), -q(2), -q(3));
}

// Rotation between ENU and NED world frames: pi about (1, 1, 0) / sqrt(2). Symmetric.
PX4RL_HD inline matrix::Quatf quatEnuNed()
{
	return matrix::Quatf(0.f, 0.70710678118654752f, 0.70710678118654752f, 0.f);
}

// Rotation between FLU and FRD body frames: pi about x. Symmetric.
PX4RL_HD inline matrix::Quatf quatFluFrd()
{
	return matrix::Quatf(0.f, 1.f, 0.f, 0.f);
}

} // namespace detail

/// q_enu_flu (body FLU -> world ENU) to q_ned_frd (body FRD -> world NED).
PX4RL_HD inline matrix::Quatf quatWorldBodyEnuFluToNedFrd(const matrix::Quatf &q_enu_flu)
{
	return detail::quatEnuNed() * q_enu_flu * detail::quatFluFrd();
}

/// q_ned_frd to q_enu_flu. Inverse of quatWorldBodyEnuFluToNedFrd.
PX4RL_HD inline matrix::Quatf quatWorldBodyNedFrdToEnuFlu(const matrix::Quatf &q_ned_frd)
{
	return detail::quatEnuNed() * q_ned_frd * detail::quatFluFrd();
}

/// q_flu_enu (world ENU -> body FLU) to q_frd_ned (world NED -> body FRD).
PX4RL_HD inline matrix::Quatf quatBodyWorldEnuFluToNedFrd(const matrix::Quatf &q_flu_enu)
{
	return detail::conjugate(quatWorldBodyEnuFluToNedFrd(detail::conjugate(q_flu_enu)));
}

/// q_frd_ned to q_flu_enu. Inverse of quatBodyWorldEnuFluToNedFrd.
PX4RL_HD inline matrix::Quatf quatBodyWorldNedFrdToEnuFlu(const matrix::Quatf &q_frd_ned)
{
	return detail::conjugate(quatWorldBodyNedFrdToEnuFlu(detail::conjugate(q_frd_ned)));
}

/// PX4 heading (yaw about NED down, 0 = north, positive clockwise seen from above, rad in (-pi, pi])
/// of a body FLU -> world ENU quaternion. Same as the yaw of Euler(321) of the converted quaternion.
PX4RL_HD inline float headingNedFromEnuFlu(const matrix::Quatf &q_enu_flu)
{
	const matrix::Quatf q = quatWorldBodyEnuFluToNedFrd(q_enu_flu);
	return atan2f(2.f * (q(0) * q(3) + q(1) * q(2)), 1.f - 2.f * (q(2) * q(2) + q(3) * q(3)));
}

} // namespace px4rl
