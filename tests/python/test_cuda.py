"""CudaFleet: parity with CpuFleet, streams, CUDA graphs, input validation and the no-allocation property."""
import numpy as np
import pytest

import px4rl
from scenario import DT_POSITION, DT_RATE, snapshots, varied_params

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda

N = 256


@pytest.fixture(scope="module")
def px4rl_cuda():
    return pytest.importorskip("px4rl.cuda")


def to_gpu(snapshot):
    """Snapshot of numpy arrays as torch CUDA tensors; time_us stays an int."""
    return type(snapshot)(**{k: torch.from_numpy(v).cuda() if isinstance(v, np.ndarray) else v
                             for k, v in vars(snapshot).items()})


def run_gpu_chain(fleet, s):
    position = fleet.position_step(s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp,
                                   DT_POSITION, s.time_us)
    attitude = fleet.attitude_step(s.q)
    rate = fleet.rate_step(attitude.rates_sp, attitude.thrust_body, s.rates, s.angular_accel, DT_RATE)
    rotor = fleet.rotor_step(rate.actuator, s.body_velocity, DT_RATE, s.body_rates)
    return position, attitude, rate, rotor


def max_abs(a, b, relative=False):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    difference = np.abs(a - b)
    return (difference / np.maximum(1.0, np.abs(b))).max() if relative else difference.max()


def fraction_within(a, b, tolerance):
    return (np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)) <= tolerance).mean()


def test_single_step_agrees_with_cpu_fleet_within_the_parity_thresholds(px4rl_cuda):
    params = varied_params(N, seed=3)
    snapshot = snapshots(N, 1, seed=5)[0]
    gpu_in = to_gpu(snapshot)

    cpu = px4rl.CpuFleet(N, num_threads=1)
    cpu.reset(params)
    gpu = px4rl_cuda.CudaFleet(N)
    gpu.reset(params)

    s = snapshot
    c_position = cpu.position_step(s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp,
                                   DT_POSITION, s.time_us)
    g_position = gpu.position_step(gpu_in.position, gpu_in.velocity, gpu_in.heading, gpu_in.acceleration_z,
                                   gpu_in.trajectory_sp, DT_POSITION, s.time_us)
    for name in ("q_sp", "thrust_sp_body", "yawspeed", "hover_thrust"):
        assert max_abs(getattr(g_position, name).cpu().numpy(), getattr(c_position, name)) <= 1e-3, name
    assert (g_position.outcome.cpu().numpy() == c_position.outcome).all()

    # The next stages are fed the CPU outputs, so each stage is compared from identical inputs.
    c_attitude = cpu.attitude_step(s.q)
    g_attitude = gpu.attitude_step(gpu_in.q, torch.from_numpy(c_position.q_sp).cuda(),
                                   torch.from_numpy(c_position.thrust_sp_body).cuda(),
                                   torch.from_numpy(c_position.yawspeed).cuda())
    assert max_abs(g_attitude.rates_sp.cpu().numpy(), c_attitude.rates_sp, relative=True) <= 1e-3
    assert max_abs(g_attitude.thrust_body.cpu().numpy(), c_attitude.thrust_body) <= 1e-6

    c_rate = cpu.rate_step(c_attitude.rates_sp, c_attitude.thrust_body, s.rates, s.angular_accel, DT_RATE)
    g_rate = gpu.rate_step(torch.from_numpy(c_attitude.rates_sp).cuda(),
                           torch.from_numpy(c_attitude.thrust_body).cuda(), gpu_in.rates, gpu_in.angular_accel,
                           DT_RATE)
    assert max_abs(g_rate.motors.cpu().numpy(), c_rate.motors) <= 1e-3
    assert max_abs(g_rate.torque_sp.cpu().numpy(), c_rate.torque_sp) <= 1e-3
    assert max_abs(g_rate.actuator.cpu().numpy(), c_rate.actuator) <= 1.0  # microseconds
    assert fraction_within(g_rate.motors.cpu().numpy(), c_rate.motors, 1e-6) >= 0.95

    c_rotor = cpu.rotor_step(c_rate.actuator, s.body_velocity, DT_RATE, s.body_rates)
    g_rotor = gpu.rotor_step(torch.from_numpy(c_rate.actuator).cuda(), gpu_in.body_velocity, DT_RATE,
                             gpu_in.body_rates)
    for name in ("force", "torque", "rotor_force"):
        assert max_abs(getattr(g_rotor, name).cpu().numpy(), getattr(c_rotor, name), relative=True) <= 1e-4, name


def test_masked_reset_and_per_vehicle_plant(px4rl_cuda):
    n = 16
    s = to_gpu(snapshots(n, 1, seed=2)[0])
    fleet, reference = px4rl_cuda.CudaFleet(n), px4rl_cuda.CudaFleet(n)
    for f in (fleet, reference):
        f.reset(px4rl.iris_params())
    for _ in range(3):
        run_gpu_chain(fleet, s)

    mask = torch.zeros(n, dtype=torch.uint8, device=fleet.device)
    mask[::2] = 1
    fleet.reset(px4rl.iris_params(), mask)
    fleet.reset_plant(px4rl.iris_plant(), mask)

    after = [t.clone() for result in run_gpu_chain(fleet, s) for t in result]
    fresh = [t.clone() for result in run_gpu_chain(reference, s) for t in result]
    # reference ran 1 step, fleet 4 steps but reset on even vehicles: the even ones must equal a fresh fleet.
    for a, f in zip(after, fresh):
        assert torch.equal(a[::2], f[::2])


def test_works_on_a_non_default_stream(px4rl_cuda):
    snapshot = snapshots(N, 1, seed=9)[0]
    expected_fleet, fleet = px4rl_cuda.CudaFleet(N), px4rl_cuda.CudaFleet(N)
    for f in (expected_fleet, fleet):
        f.reset(px4rl.iris_params())
    expected = [t.clone() for result in run_gpu_chain(expected_fleet, to_gpu(snapshot)) for t in result]
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    inputs = to_gpu(snapshot)
    late = {k: torch.zeros_like(v) for k, v in vars(inputs).items() if isinstance(v, torch.Tensor)}
    with torch.cuda.stream(stream):
        # The inputs are written only after a long sleep on `stream`. A launch on any other stream would read
        # the zeros and disagree with the expected result.
        torch.cuda._sleep(200_000_000)
        staged = type(inputs)(**{**vars(inputs), **late})
        for name, tensor in late.items():
            tensor.copy_(getattr(inputs, name))
        result = [t.clone() for r in run_gpu_chain(fleet, staged) for t in r]
    stream.synchronize()
    for got, want in zip(result, expected):
        assert torch.equal(got, want)


def test_cuda_graph_replay_equals_eager(px4rl_cuda):
    params = px4rl.iris_params()
    history = snapshots(N, 1, seed=13)
    s = to_gpu(history[0])
    eager, captured, warmup = (px4rl_cuda.CudaFleet(N) for _ in range(3))
    for f in (eager, captured, warmup):
        f.reset(params)

    # Loads the kernels before capture (lazy module loading) on a throwaway fleet.
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        run_gpu_chain(warmup, s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorded = run_gpu_chain(captured, s)  # dt and time_us are baked into the graph

    for step in range(6):
        expected = [t.clone() for result in run_gpu_chain(eager, s) for t in result]
        graph.replay()
        replayed = [t for result in recorded for t in result]
        torch.cuda.synchronize()
        for got, want in zip(replayed, expected):
            assert torch.equal(got, want), f"step {step}"


def test_a_step_allocates_no_gpu_memory(px4rl_cuda):
    s = to_gpu(snapshots(N, 1, seed=17)[0])
    fleet = px4rl_cuda.CudaFleet(N)
    fleet.reset(px4rl.iris_params())
    for _ in range(5):
        run_gpu_chain(fleet, s)
    torch.cuda.synchronize()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    for _ in range(100):
        run_gpu_chain(fleet, s)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.memory_reserved() == reserved


def test_inputs_with_the_wrong_device_dtype_shape_or_layout_raise_value_error(px4rl_cuda):
    n = 8
    fleet = px4rl_cuda.CudaFleet(n)
    fleet.reset(px4rl.iris_params())
    s = to_gpu(snapshots(n, 1)[0])
    args = [s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp]

    def position_step(*replace):
        values = list(args)
        for index, value in replace:
            values[index] = value
        return fleet.position_step(*values, DT_POSITION, s.time_us)

    position_step()
    with pytest.raises(ValueError, match="cuda:"):
        position_step((0, s.position.cpu()))
    with pytest.raises(ValueError, match="float32"):
        position_step((0, s.position.double()))
    with pytest.raises(ValueError, match="contiguous"):
        position_step((0, s.position.t().contiguous().t()))
    with pytest.raises(ValueError, match="shape"):
        position_step((2, s.heading[:4]))
    with pytest.raises(ValueError, match="uint8"):
        fleet.reset(px4rl.iris_params(), torch.ones(n, dtype=torch.bool, device=fleet.device))
    with pytest.raises(ValueError, match="out.q_sp"):
        out = px4rl.PositionResult(torch.zeros(n, 4, dtype=torch.float64, device=fleet.device), *fleet._position_out[1:])
        fleet.position_step(*args, DT_POSITION, s.time_us, out=out)
    with pytest.raises(TypeError):
        position_step((0, s.position.cpu().numpy()))
    with pytest.raises(ValueError, match="position_step"):
        px4rl_cuda.CudaFleet(n).attitude_step(s.q)
    with pytest.raises(ValueError):
        px4rl_cuda.CudaFleet(0)
    with pytest.raises(ValueError):
        px4rl_cuda.CudaFleet(n, device="cpu")


def test_native_errors_become_runtime_errors_and_closed_fleets_are_rejected(px4rl_cuda):
    fleet = px4rl_cuda.CudaFleet(4)
    bad = px4rl.iris_params()
    bad.num_rotors = 0
    with pytest.raises(RuntimeError, match="num_rotors"):
        fleet.reset(bad)
    with pytest.raises(RuntimeError, match="n_params"):
        fleet.reset(px4rl.params_array(3))
    with pytest.raises(TypeError):
        fleet.reset(object())
    s = to_gpu(snapshots(4, 1)[0])
    with pytest.raises(RuntimeError, match="dt"):
        fleet.rotor_step(torch.zeros(4, 4, device=fleet.device), s.body_velocity, 0.0)
    assert len(fleet) == 4 and fleet.device.type == "cuda"
    fleet.close()
    fleet.close()
    with pytest.raises(RuntimeError, match="closed"):
        fleet.reset(px4rl.iris_params())
