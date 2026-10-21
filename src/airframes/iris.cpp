// Gazebo Classic Iris plant: reference/gazebo_classic_iris/iris.sdf.jinja (PX4-SITL_gazebo-classic
// @ 70683dc). Raw values with their sources are tabulated in docs/iris-params-sources.md (section
// numbers below refer to it). Values are float; the SDF values are decimal and double.
//
// Rotor index mapping (PX4 output index -> SDF rotor), derived end to end:
//   PX4 actuator_outputs.output[i] -> HIL_ACTUATOR_CONTROLS controls[i]
//     (src/modules/mavlink/streams/HIL_ACTUATOR_CONTROLS.hpp:76)
//   -> <channel> with <input_index> i (iris.sdf.jinja:471 / 479 / 487 / 495 = 0 / 1 / 2 / 3), which
//      fills input_reference_[i] (gazebo_mavlink_interface.cpp:235-243, 1190-1194)
//   -> motor_speed message entry i (gazebo_mavlink_interface.cpp:618-620)
//   -> the motor plugin whose <motorNumber> is i (iris.sdf.jinja:364 / 381 / 398 / 415 = 0 / 1 / 2 / 3;
//      read at gazebo_motor_model.cpp:189)
//   -> its <jointName> / <linkName> rotor_i_joint / rotor_i (iris.sdf.jinja:355 / 372 / 389 / 406).
// So the mapping is the identity: PX4 output i drives SDF rotor_i. By turning direction and
// position (iris.sdf.jinja:89 / 154 / 219 / 284, 357 / 374 / 391 / 408):
//   output 0 -> rotor_0 front-right (+0.13, -0.22), ccw   [plugin 'front_right_motor_model']
//   output 1 -> rotor_1 back-left   (-0.13, +0.20), ccw   [plugin 'back_left_motor_model']
//   output 2 -> rotor_2 front-left  (+0.13, +0.22), cw    [plugin 'front_left_motor_model']
//   output 3 -> rotor_3 back-right  (-0.13, -0.20), cw    [plugin 'back_right_motor_model']
// This is the plant side only. It is intentionally not reconciled with the controller's CA_ROTOR*
// geometry (docs/architecture.md, "Iris airframe"): PX4 SITL runs the same mismatch.
#include "px4rl/airframes.h"
#include "px4rl/params.hpp"

extern "C" {

void px4rl_params_iris(px4rl_params *p)
{
	*p = px4rl::irisParams();
}

px4rl_plant_params px4rl_plant_params_iris(void)
{
	// Positions (iris.sdf.jinja:89, 154, 219, 284), docs/iris-params-sources.md section 1b. Body FLU.
	static const float kPosition[4][3] = {
		{0.13f, -0.22f, 0.023f},
		{-0.13f, 0.2f, 0.023f},
		{0.13f, 0.22f, 0.023f},
		{-0.13f, -0.2f, 0.023f},
	};
	// <turningDirection> (iris.sdf.jinja:357, 374, 391, 408): ccw = +1, cw = -1 (gazebo_motor_model.cpp:108-116).
	static const int kTurningDirection[4] = {+1, +1, -1, -1};

	px4rl_plant_params plant = {};
	plant.num_rotors = 4;

	for (int i = 0; i < 4; ++i) {
		px4rl_rotor_params &r = plant.rotor[i];

		for (int k = 0; k < 3; ++k) {
			r.position[k] = kPosition[i][k];
		}

		// Joint axis 0 0 1 in the model frame (iris.sdf.jinja:141, use_parent_model_frame = 1).
		r.axis[0] = 0.f;
		r.axis[1] = 0.f;
		r.axis[2] = 1.f;
		r.turning_direction = kTurningDirection[i];
		r.reversible = 0; // no <reversible> element; default false (gazebo_motor_model.h:124)

		// Motor plugin, same for all four rotors (iris.sdf.jinja:358-366 and the equivalent lines of the others).
		r.motor_constant = 5.84e-06f;            // :361
		r.moment_constant = 0.06f;               // :362
		r.time_constant_up = 0.0125f;            // :358
		r.time_constant_down = 0.025f;           // :359
		r.max_rot_velocity = 1100.f;             // :360
		r.rotor_drag_coefficient = 0.000175f;    // :365
		r.rolling_moment_coefficient = 1e-06f;   // :366
		r.max_relative_airspeed = 25.f;          // not in the SDF: kDefaultMaxRelativeAirspeed, gazebo_motor_model.h:74

		// MAVLink channels rotor1..rotor4 (iris.sdf.jinja:470-501): input_index i, offset 0, scaling 1000,
		// zero_position_disarmed 0, zero_position_armed 100.
		r.input_offset = 0.f;
		r.input_scaling = 1000.f;
		r.zero_position_armed = 100.f;
		r.zero_position_disarmed = 0.f;
		r.px4_output_index = i; // identity mapping, see the header comment
	}

	return plant;
}

px4rl_rigid_body_params px4rl_rigid_body_params_iris(void)
{
	// base_link inertial (iris.sdf.jinja:7-15), docs/iris-params-sources.md section 1a.
	// Inertial pose is the origin, off-diagonal terms are 0.
	px4rl_rigid_body_params body = {};
	body.mass = 1.5f;            // :8
	body.inertia[0] = 0.029125f; // ixx :10
	body.inertia[4] = 0.029125f; // iyy :13
	body.inertia[8] = 0.055225f; // izz :15
	return body;
}

} // extern "C"
