/* px4-rl shim: plain struct replacing the uORB-generated trajectory_setpoint_s.
 * Mirrors PX4 v1.16.2 msg/versioned/TrajectorySetpoint.msg: same field names, types and declaration
 * order, so that aggregate initialisers written for PX4 (e.g. PositionControl::empty_trajectory_setpoint)
 * keep working. The generator's trailing alignment padding member is omitted. */
#pragma once

#include <stdint.h>

struct trajectory_setpoint_s {
	uint64_t timestamp; // time since system start (microseconds)
	float position[3]; // NED local world frame, meters
	float velocity[3]; // meters/second
	float acceleration[3]; // meters/second^2
	float jerk[3]; // meters/second^3 (for logging only)
	float yaw; // euler angle of desired attitude in radians -PI..+PI
	float yawspeed; // angular velocity around NED frame z-axis in radians/second

	static constexpr uint32_t MESSAGE_VERSION = 0;
};
