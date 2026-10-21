"""CpuFleet through the bindings: determinism, per-vehicle independence, validation and physical sanity."""
import numpy as np
import pytest

import px4rl
from scenario import DT_POSITION, DT_RATE, row, run_chain, same_bits, snapshots, varied_params

N = 8
STEPS = 20


def test_each_vehicle_of_a_fleet_matches_a_fleet_of_one_bit_for_bit_over_20_steps_of_every_loop():
    params = varied_params(N)
    history = snapshots(N, STEPS)

    fleet = px4rl.CpuFleet(N, num_threads=1)
    fleet.reset(params)
    batched = run_chain(fleet, history)

    for i in range(N):
        single = px4rl.CpuFleet(1, num_threads=1)
        single.reset(params[i])
        alone = run_chain(single, [row(snapshot, i) for snapshot in history])
        for step, (all_vehicles, one_vehicle) in enumerate(zip(batched, alone)):
            for field, (a, b) in enumerate(zip(all_vehicles, one_vehicle)):
                assert same_bits(a[i:i + 1], b), f"vehicle {i}, step {step}, output {field}"


def test_thread_count_does_not_change_a_bit():
    n = 128  # above the size at which CpuFleet parallelises
    params = varied_params(n)
    history = snapshots(n, 5)
    results = []
    for threads in (1, 4):
        fleet = px4rl.CpuFleet(n, num_threads=threads)
        assert fleet.num_threads == threads
        fleet.reset(params)
        results.append(run_chain(fleet, history))
    for step, (a, b) in enumerate(zip(*results)):
        assert all(same_bits(x, y) for x, y in zip(a, b)), f"step {step}"


def test_a_masked_reset_reconstructs_only_the_selected_vehicles():
    history = snapshots(4, 6)
    reference = px4rl.CpuFleet(4, num_threads=1)
    reference.reset(px4rl.iris_params())
    fresh = px4rl.CpuFleet(4, num_threads=1)
    fresh.reset(px4rl.iris_params())

    run_chain(reference, history[:3])
    run_chain(fresh, history[:3])
    mask = np.array([0, 1, 0, 1], np.uint8)
    fresh.reset(px4rl.iris_params(), mask)

    after_reset = run_chain(fresh, history[3:])
    continued = run_chain(reference, history[3:])
    restarted = px4rl.CpuFleet(4, num_threads=1)
    restarted.reset(px4rl.iris_params())
    from_scratch = run_chain(restarted, history[3:])

    # The first step after the reset: unmasked vehicles continue, masked ones start from scratch. The rotor
    # outputs (fields 10 and up) are left out: a controller reset does not touch the rotors.
    for field in range(10):
        a, c, s = after_reset[0][field], continued[0][field], from_scratch[0][field]
        assert same_bits(a[[0, 2]], c[[0, 2]]) and same_bits(a[[1, 3]], s[[1, 3]]), field
    force_after, force_continued = after_reset[0][10], continued[0][10]
    assert same_bits(force_after, force_continued)  # the spun-up rotors were kept


def test_reset_plant_replaces_only_the_selected_plants_and_leaves_the_controllers_alone():
    n = 3
    history = snapshots(n, 4)
    fleets = [px4rl.CpuFleet(n, num_threads=1) for _ in range(2)]
    for fleet in fleets:
        fleet.reset(px4rl.iris_params())
        run_chain(fleet, history[:2])

    plants = px4rl.params_array(n, px4rl.iris_plant())
    for rotor in plants[1].rotor:
        rotor.motor_constant *= 2.0
    fleets[0].reset_plant(plants, np.array([0, 1, 0], np.uint8))
    changed = run_chain(fleets[0], history[2:])
    untouched = run_chain(fleets[1], history[2:])

    for step_changed, step_untouched in zip(changed, untouched):
        controller_fields = range(10)  # position, attitude and rate results
        assert all(same_bits(step_changed[k], step_untouched[k]) for k in controller_fields)
        force_changed, force_untouched = step_changed[10], step_untouched[10]
        assert same_bits(force_changed[[0, 2]], force_untouched[[0, 2]])
        assert not same_bits(force_changed[[1]], force_untouched[[1]])


def test_position_hold_at_rest_commands_the_hover_thrust_and_a_level_attitude():
    n = 4
    fleet = px4rl.CpuFleet(n, num_threads=1)
    fleet.reset(px4rl.iris_params())
    zeros = lambda *shape: np.zeros(shape, np.float32)
    setpoint = px4rl.setpoints.position_yaw(zeros(n, 3), 0.0)
    result = fleet.position_step(zeros(n, 3), zeros(n, 3), zeros(n), zeros(n), setpoint, DT_POSITION, 20000)
    np.testing.assert_allclose(result.thrust_sp_body[:, 2], -0.5, atol=1e-6)
    np.testing.assert_allclose(result.q_sp, np.tile([1, 0, 0, 0], (n, 1)), atol=1e-6)
    assert (result.outcome == px4rl.OUTCOME_OK).all()


def test_rotor_step_reaches_the_weight_of_the_iris_at_the_hover_command():
    fleet = px4rl.CpuFleet(2, num_threads=1)
    mass, gravity = px4rl.iris_rigid_body().mass, 9.80665
    pwm = np.full((2, 4), 1000 + 1000 * 0.6935, np.float32)
    zero = np.zeros((2, 3), np.float32)
    for _ in range(500):  # 2 s, many time constants
        result = fleet.rotor_step(pwm, zero, 0.004)
    np.testing.assert_allclose(result.force[:, 2], mass * gravity, rtol=2e-3)
    np.testing.assert_allclose(result.force[:, :2], 0, atol=1e-5)
    assert result.rotor_force.shape == (2, 4, 3)
    np.testing.assert_allclose(result.rotor_force.sum(axis=1), result.force, atol=1e-4)
    # A disarmed vehicle commands the disarmed rotor speed (zero), so its rotors spin down.
    armed = np.array([1, 0], np.uint8)
    for _ in range(500):
        result = fleet.rotor_step(pwm, zero, 0.004, armed=armed)
    assert result.force[0, 2] == pytest.approx(mass * gravity, rel=2e-3)
    assert result.force[1, 2] < 0.01 * mass * gravity


def test_results_are_the_fleets_own_buffers_unless_out_is_given():
    n = 3
    s = snapshots(n, 1)[0]
    args = (s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp, DT_POSITION, s.time_us)
    fleet, fleet_with_out = (px4rl.CpuFleet(n, num_threads=1) for _ in range(2))
    for f in (fleet, fleet_with_out):
        f.reset(px4rl.iris_params())

    own = fleet.position_step(*args)
    assert all(x is y for x, y in zip(own, fleet.position_step(*args)))

    mine = px4rl.PositionResult(*(np.zeros_like(a) for a in own))
    returned = fleet_with_out.position_step(*args, out=mine)
    assert returned is mine
    fresh = px4rl.CpuFleet(n, num_threads=1)
    fresh.reset(px4rl.iris_params())
    assert all(same_bits(x, y) for x, y in zip(mine, fresh.position_step(*args)))

    skipped = mine._replace(hover_thrust=None, outcome=None)
    assert fleet_with_out.position_step(*args, out=skipped) is skipped


def test_attitude_step_defaults_to_the_last_position_step():
    n = 5
    history = snapshots(n, 2)
    s = history[0]
    fleet_a, fleet_b = (px4rl.CpuFleet(n, num_threads=1) for _ in range(2))
    for fleet in (fleet_a, fleet_b):
        fleet.reset(px4rl.iris_params())
    args = (s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp, DT_POSITION, s.time_us)
    position = fleet_a.position_step(*args)
    fleet_b.position_step(*args)
    implicit = fleet_a.attitude_step(s.q)
    explicit = fleet_b.attitude_step(s.q, position.q_sp.copy(), position.thrust_sp_body.copy(),
                                     position.yawspeed.copy())
    assert same_bits(implicit.rates_sp, explicit.rates_sp) and same_bits(implicit.thrust_body, explicit.thrust_body)

    fresh = px4rl.CpuFleet(n, num_threads=1)
    with pytest.raises(ValueError, match="position_step"):
        fresh.attitude_step(s.q)


def test_invalid_arguments_raise():
    n = 4
    fleet = px4rl.CpuFleet(n, num_threads=1)
    s = snapshots(n, 1)[0]
    args = [s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp, DT_POSITION, s.time_us]

    def call(**override):
        names = ["position", "velocity", "heading", "acceleration_z", "trajectory_sp", "dt", "time_us"]
        values = dict(zip(names, args))
        values.update(override)
        return fleet.position_step(**values)

    with pytest.raises(ValueError, match="float32"):
        call(position=s.position.astype(np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        call(position=np.asfortranarray(s.position))
    with pytest.raises(ValueError, match="shape"):
        call(heading=s.heading[:2])
    read_only = px4rl.PositionResult(*(np.zeros_like(a) for a in call()))
    for a in read_only:
        a.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        fleet.position_step(*args, out=read_only)
    with pytest.raises(TypeError):
        call(velocity=[[0.0] * 3] * n)
    with pytest.raises(ValueError):
        fleet.rotor_step(np.zeros((n, 4), np.float32), np.zeros((n, 3), np.float32), 0.0)
    with pytest.raises(ValueError):
        px4rl.CpuFleet(0)
    with pytest.raises(ValueError):
        px4rl.CpuFleet(4, num_threads=-1)


def test_reset_validates_the_parameter_buffer():
    fleet = px4rl.CpuFleet(4, num_threads=1)
    bad = px4rl.iris_params()
    bad.num_rotors = 0
    with pytest.raises(ValueError, match="num_rotors"):
        fleet.reset(bad)
    with pytest.raises(ValueError, match="n_params"):
        fleet.reset(px4rl.params_array(3))
    with pytest.raises(ValueError, match="item size"):
        fleet.reset(px4rl.iris_plant())
    with pytest.raises(TypeError):
        fleet.reset(object())
    with pytest.raises(ValueError, match="shape"):
        fleet.reset(px4rl.iris_params(), np.zeros(3, np.uint8))
    with pytest.raises(ValueError):
        fleet.reset_plant(px4rl.params_array(2, px4rl.iris_plant()))
