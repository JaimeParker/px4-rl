"""The single-vehicle classes against CpuFleet with n = 1, bit for bit."""
import numpy as np
import pytest

import px4rl
from scenario import DT_POSITION, DT_RATE, row, same_bits, snapshots, varied_params

STEPS = 20


def f32(value):
    return np.asarray(value, np.float32)


def test_vehicle_matches_a_fleet_of_one_over_20_steps_of_every_loop():
    for seed in range(3):
        params = varied_params(1, seed)[0]
        fleet = px4rl.CpuFleet(1, num_threads=1)
        fleet.reset(params)
        vehicle = px4rl.Vehicle(params)

        for step, snapshot in enumerate(snapshots(3, STEPS, seed=seed)):
            s = row(snapshot, seed % 3)
            fp = fleet.position_step(s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp,
                                     DT_POSITION, s.time_us)
            vp = vehicle.position_step(s.position[0], s.velocity[0], float(s.heading[0]),
                                       float(s.acceleration_z[0]), s.trajectory_sp[0], DT_POSITION, s.time_us)
            assert same_bits(vp.q_sp, fp.q_sp[0]) and same_bits(vp.thrust_sp_body, fp.thrust_sp_body[0]), step
            assert (vp.yawspeed, vp.hover_thrust, vp.outcome) == (
                fp.yawspeed[0], fp.hover_thrust[0], fp.outcome[0]), step

            fa = fleet.attitude_step(s.q)
            va = vehicle.attitude_step(s.q[0])
            assert same_bits(va.rates_sp, fa.rates_sp[0]) and same_bits(va.thrust_body, fa.thrust_body[0]), step

            fr = fleet.rate_step(fa.rates_sp, fa.thrust_body, s.rates, s.angular_accel, DT_RATE)
            vr = vehicle.rate_step(va.rates_sp, va.thrust_body, s.rates[0], s.angular_accel[0], DT_RATE)
            for name in ("motors", "actuator", "torque_sp"):
                assert same_bits(getattr(vr, name), getattr(fr, name)[0]), (step, name)


def test_vehicle_explicit_attitude_setpoint_matches_the_fleet():
    params = px4rl.iris_params()
    fleet, vehicle = px4rl.CpuFleet(1, num_threads=1), px4rl.Vehicle(params)
    fleet.reset(params)
    s = row(snapshots(1, 1)[0], 0)
    q_sp = f32([[0.99, 0.05, -0.05, 0.1]])
    thrust = f32([[0.0, 0.0, -0.6]])
    f = fleet.attitude_step(s.q, q_sp, thrust, f32([0.3]))
    v = vehicle.attitude_step(s.q[0], q_sp[0], thrust[0], 0.3)
    assert same_bits(v.rates_sp, f.rates_sp[0]) and same_bits(v.thrust_body, f.thrust_body[0])
    with pytest.raises(ValueError, match="position_step"):
        px4rl.Vehicle(params).attitude_step(s.q[0])


def test_the_separate_controller_classes_compose_to_the_fleet_bit_for_bit():
    """Wires the chain (hover thrust estimator, position, attitude, rate, allocator, output) from the individual
    classes, the way a user building a custom chain would."""
    params = varied_params(1, seed=7)[0]
    fleet = px4rl.CpuFleet(1, num_threads=1)
    fleet.reset(params)

    hover = px4rl.HoverThrustEstimator(params)
    position = px4rl.PositionController(params)
    attitude = px4rl.AttitudeController(params)
    rate = px4rl.RateController(params)
    allocator = px4rl.ControlAllocator(params)
    output = px4rl.ActuatorOutput(params)

    q = f32([1, 0, 0, 0])
    thrust_allocated_sp = np.zeros(3, np.float32)
    unallocated = np.zeros(3, np.float32)
    rate_recorded = False

    for step, snapshot in enumerate(snapshots(1, STEPS, seed=11)):
        s = row(snapshot, 0)

        # Position loop; the estimator needs the attitude and thrust of the previous cycle.
        hover.update(True, rate_recorded, DT_POSITION, float(s.acceleration_z[0]), s.velocity[0], q,
                     thrust_allocated_sp, unallocated, s.time_us)
        position.apply_hover_thrust_estimate(hover.hover_thrust, hover.valid)
        p = position.update(s.position[0], s.velocity[0], float(s.heading[0]), DT_POSITION, s.trajectory_sp[0],
                            s.time_us)
        f = fleet.position_step(s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp,
                                DT_POSITION, s.time_us)
        assert same_bits(p.q_sp, f.q_sp[0]) and same_bits(p.thrust_sp_body, f.thrust_sp_body[0]), step
        assert p.yawspeed == f.yawspeed[0] and hover.hover_thrust == f.hover_thrust[0], step

        # Attitude loop.
        q = s.q[0]
        a = attitude.update(q, p.q_sp, p.thrust_sp_body, p.yawspeed)
        fa = fleet.attitude_step(s.q)
        assert same_bits(a.rates_sp, fa.rates_sp[0]) and same_bits(a.thrust_body, fa.thrust_body[0]), step

        # Rate loop: controller, allocator, saturation feedback, output mapping.
        torque_sp, thrust_body = rate.update(a.rates_sp, a.thrust_body, s.rates[0], s.angular_accel[0], DT_RATE)
        allocated = allocator.update(torque_sp, thrust_body, DT_RATE)
        rate.set_saturation_status(allocated.saturation_positive, allocated.saturation_negative)
        thrust_allocated_sp, unallocated, rate_recorded = thrust_body, allocated.unallocated_thrust, True
        pwm = output.update(allocated.motors)
        fr = fleet.rate_step(fa.rates_sp, fa.thrust_body, s.rates, s.angular_accel, DT_RATE)
        assert same_bits(allocated.motors, fr.motors[0]) and same_bits(torque_sp, fr.torque_sp[0]), step
        assert same_bits(pwm.astype(np.float32), fr.actuator[0]), step


def test_single_vehicle_classes_validate_their_inputs():
    params = px4rl.iris_params()
    with pytest.raises(ValueError):
        px4rl.AttitudeController(px4rl.params_array(2))
    bad = px4rl.iris_params()
    bad.num_rotors = 9
    with pytest.raises(ValueError, match="num_rotors"):
        px4rl.Vehicle(bad)
    with pytest.raises(TypeError):
        px4rl.RateController("not params")

    attitude = px4rl.AttitudeController(params)
    with pytest.raises(ValueError, match="length 4"):
        attitude.update([1, 0, 0], [1, 0, 0, 0], [0, 0, -0.5])
    with pytest.raises(ValueError, match="1 to 4"):
        px4rl.ActuatorOutput(params).update(np.zeros(5))
    # Tuples and lists are accepted.
    result = attitude.update((1, 0, 0, 0), [1, 0, 0, 0], (0, 0, -0.5))
    assert result.rates_sp.shape == (3,) and result.thrust_body.shape == (3,)
