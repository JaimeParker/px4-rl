/* px4-rl shim: plain struct replacing the uORB-generated vehicle_attitude_setpoint_s.
 * Mirrors PX4 v1.16.2 msg/versioned/VehicleAttitudeSetpoint.msg: same field names, types and
 * declaration order. The generator's trailing alignment padding member is omitted. */
#pragma once

#include <stdint.h>

struct vehicle_attitude_setpoint_s {
	uint64_t timestamp; // time since system start (microseconds)
	float yaw_sp_move_rate; // rad/s (commanded by user)
	float q_d[4]; // desired quaternion for quaternion control
	float thrust_body[3]; // normalized thrust command in body FRD frame [-1,1]
	bool reset_integral; // reset roll/pitch/yaw integrals (navigation logic change)
	bool fw_control_yaw_wheel; // control heading with steering wheel (used for auto takeoff on runway)

	static constexpr uint32_t MESSAGE_VERSION = 0;
};
