/* px4-rl shim: plain struct replacing the uORB-generated vehicle_local_position_setpoint_s.
 * Mirrors PX4 v1.16.2 msg/VehicleLocalPositionSetpoint.msg: same field names, types and declaration
 * order. The generator's trailing alignment padding member is omitted. */
#pragma once

#include <stdint.h>

struct vehicle_local_position_setpoint_s {
	uint64_t timestamp; // time since system start (microseconds)
	float x; // in meters NED
	float y; // in meters NED
	float z; // in meters NED
	float vx; // in meters/sec
	float vy; // in meters/sec
	float vz; // in meters/sec
	float acceleration[3]; // in meters/sec^2
	float thrust[3]; // normalized thrust vector in NED
	float yaw; // in radians NED -PI..+PI
	float yawspeed; // in radians/sec
};
