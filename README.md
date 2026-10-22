# px4-rl

PX4-faithful multirotor control and rotor dynamics for reinforcement learning,
compiled from a single C++ source for both CPU and CUDA.

px4-rl takes the controllers that PX4-Autopilot (v1.16) flies with:
- the attitude controller;
- the body-rate PID with its anti-windup;
- the control allocator with sequential desaturation and airmode;
- the thrust curve and actuator-output mapping;
- the position/velocity controller and the hover-thrust estimator.

It strips the PX4 framework (uORB, parameters, work queues) and makes every function `__host__ __device__`. The result runs:
- **batched on the GPU** behind a small C API, so thousands of simulated vehicles can be stepped inside a GPU simulator (for example Isaac Lab) with no host round trip;
- **on the CPU** as a static library and CMake package, for other simulators, tooling and deployment-side code.

The rotor plant is a port of the Gazebo Classic motor model that PX4's own
SITL uses. A policy trained against it therefore sees the same actuator path
as PX4 SITL: quantized PWM, then rotor-speed lag, then thrust and drag.

## Status

| Component | CPU | CUDA |
|---|---|---|
| Attitude, rate, allocator, actuator output | ✓ | ✓ (batched fleet, C API) |
| Position/velocity control, hover-thrust estimator | ✓ | ✓ (batched fleet, C API) |
| Gazebo-faithful rotor model, ENU/FLU ↔ NED/FRD frames | ✓ | ✓ |
| Rotor plant in the fleet (`rotor_step`: PWM → wrench) | ✓ | ✓ (batched fleet, C API) |
| Iris airframe (PX4 SITL parameters, controller and plant) | ✓ | ✓ |
| Python package: `CpuFleet` and single-vehicle classes (pybind11, numpy), `CudaFleet` (ctypes, torch) | ✓ | ✓ |

## Build

Requirements:
- CMake ≥ 3.18 and a C++17 compiler.
- For the GPU library: the CUDA toolkit (tested with 12.4). Native SASS for Blackwell (sm_120) requires CUDA ≥ 12.8; older toolkits rely on PTX JIT there.
- The tests download GoogleTest at configure time.

```bash
cmake -S . -B build -DPX4RL_BUILD_CUDA=ON      # omit for a CPU-only build
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix <prefix>
```

| Option | Default | Effect |
|---|---|---|
| `PX4RL_BUILD_CUDA` | OFF | Builds `libpx4rl_cuda.so` and the CUDA tests |
| `PX4RL_BUILD_TESTS` | ON | Builds the unit, parity and benchmark tests |
| `PX4RL_OPENMP` | ON if found | Uses OpenMP in the CPU fleet |
| `PX4RL_STRICT_FP` | OFF | Disables FMA contraction on host and device (for parity studies) |

`scripts/remote_ci.sh` streams the working tree into a docker container on a remote GPU host and runs the same configure, build and test steps there. Configure it with `PX4RL_SSH` and `PX4RL_CONTAINER`.

## Use

**C++, CPU.** Link the CMake package:

```cmake
find_package(px4rl REQUIRED)
target_link_libraries(app PRIVATE px4rl::px4)   # controllers; px4rl::fleet for the batched CPU fleet
```

**CUDA, from any language.** `include/px4rl/c_api.h` declares a flat C API on device pointers and a caller-provided `cudaStream_t`:

```c
px4rl_cuda_fleet* fleet;
px4rl_params params;
px4rl_params_iris(&params);   /* or px4rl_params_default(&params) */
px4rl_cuda_fleet_create(n, /*device*/ 0, &fleet);   /* every vehicle also gets the Iris rotor plant */
px4rl_cuda_fleet_reset(fleet, /*mask*/ NULL, &params, 1, stream);
/* velocity + yaw: traj_sp rows are (pos[3], vel[3], acc[3], yaw, yawspeed) with NaN for unused fields */
px4rl_cuda_position_step(fleet, pos, vel, heading, az, traj_sp, NULL, dt_pos, time_us,
                         q_sp, thrust_sp_body, yawspeed, NULL, NULL, stream);
px4rl_cuda_attitude_step(fleet, q, /*q_sp: use position_step's*/ NULL, NULL, thrust_sp_body,
                         rates_sp, thrust_body, stream);
px4rl_cuda_rate_step(fleet, rates_sp, thrust_body, rates, angular_accel, dt, NULL, motors, pwm, NULL, stream);
/* rotor plant: PWM and body velocity in (body FLU, relative to the air), wrench on the body out */
px4rl_cuda_rotor_step(fleet, pwm, NULL, body_vel_flu, body_rates_flu, dt, force_flu, torque_flu, NULL, stream);
```

How the C API behaves:
- Step calls only launch kernels on the given stream. They never allocate or synchronize, so they can be captured in CUDA graphs.
- The shared library exports only `px4rl_*` C symbols. It does not depend on the Python or PyTorch version; a PyTorch user passes `tensor.data_ptr()` and `torch.cuda.current_stream().cuda_stream`.
- All PX4-side quantities use PX4's frames: NED world, FRD body, Hamilton quaternions (w, x, y, z). `include/px4rl/frames.hpp` converts to and from ENU/FLU.
- The rotor plant works in the Gazebo / Isaac Lab frames: the wrench returned by `px4rl_cuda_rotor_step` is in body FLU, and gravity and rigid-body integration stay with the simulator. `px4rl_cuda_fleet_reset_plant` replaces a plant and puts its rotors at rest; the controller reset leaves the rotors alone.

**Python.** The package builds with scikit-build-core, which needs CMake ≥ 3.18 and a C++17 compiler (pybind11 is fetched as a build dependency):

```bash
pip install .                                          # CPU: px4rl.CpuFleet and the single-vehicle classes
pip install . -C cmake.define.PX4RL_BUILD_CUDA=ON      # also ships libpx4rl_cuda.so for px4rl.cuda (CUDA toolkit)
```

`import px4rl` needs only numpy; `px4rl.cuda` needs torch. The CPU and GPU fleets have the same methods, and every step returns a named tuple of buffers the fleet owns (the next call overwrites them, so clone to keep; `out=` writes into your own arrays instead).

```python
import numpy as np
import px4rl

n = 1024
fleet = px4rl.CpuFleet(n)                       # num_threads=0 uses the OpenMP default, 1 runs serially
fleet.reset(px4rl.iris_params())                # one Params for all vehicles; px4rl.params_array(n) gives one each
zeros = lambda *shape: np.zeros(shape, np.float32)
q = np.tile(np.float32([1, 0, 0, 0]), (n, 1))
# velocity + yaw: 1 m/s along x, heading 0 (builders for position, velocity and acceleration + yaw)
setpoint = px4rl.setpoints.velocity_yaw(np.tile(np.float32([1, 0, 0]), (n, 1)), yaw=0.0)

p = fleet.position_step(zeros(n, 3), zeros(n, 3), zeros(n), zeros(n), setpoint, dt=0.02, time_us=20_000)
a = fleet.attitude_step(q)                      # uses p's attitude and thrust setpoint
r = fleet.rate_step(a.rates_sp, a.thrust_body, zeros(n, 3), zeros(n, 3), dt=0.004)
w = fleet.rotor_step(r.actuator, zeros(n, 3), dt=0.004)   # w.force, w.torque: wrench on the body, FLU
```

The CPU fleet parallelises over vehicles with OpenMP. On shared or heavily loaded machines, the default team (all cores, busy-waiting barriers) can be far slower than a few threads: pass `num_threads`, or set `OMP_NUM_THREADS` / `OMP_WAIT_POLICY=passive`.

On the GPU the same calls take float32 CUDA tensors and run on the current torch stream. A step allocates nothing, so a whole chain can be captured in a `torch.cuda.CUDAGraph` (`dt` and `time_us` are fixed at capture):

```python
import torch
from px4rl.cuda import CudaFleet

fleet = CudaFleet(n, device="cuda:0")
fleet.reset(px4rl.iris_params())
z1, z3 = torch.zeros(n, device="cuda"), torch.zeros(n, 3, device="cuda")
q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(n, 1)
setpoint = px4rl.setpoints.velocity_yaw(torch.tensor([[1.0, 0.0, 0.0]], device="cuda").repeat(n, 1), yaw=0.0)

def chain():
    fleet.position_step(z3, z3, z1, z1, setpoint, dt=0.02, time_us=20_000)
    a = fleet.attitude_step(q)
    r = fleet.rate_step(a.rates_sp, a.thrust_body, z3, z3, dt=0.004)
    return fleet.rotor_step(r.actuator, z3, dt=0.004)

chain()                                   # warm-up (advances the state by one step)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    wrench = chain()
graph.replay()                            # the four steps in one launch; wrench.force, wrench.torque
```

`px4rl.hover_thrust(params, plant, body, gravity)` (C: `px4rl_hover_thrust`, `include/px4rl/airframes.h`) returns the thrust at which a vehicle hovers, in PX4's `MPC_THR_HOVER` units (Iris: 0.6935); seed `params.mpc_thr_hover` with it to start an episode in steady flight.

Single vehicles, for code that does not batch: `px4rl.Vehicle` (the whole chain), `AttitudeController`, `RateController`, `ControlAllocator`, `ActuatorOutput`, `PositionController` and `HoverThrustEstimator`, each configured from a `Params` and taking numpy arrays or tuples.

## CPU vs GPU agreement

The GPU build is not bit-identical to the CPU build. CUDA's math library differs from glibc in the last ULPs, so agreement is defined at three levels and enforced by `tests/parity/`:
- **Single step from identical state**: the outputs agree within 1e-3 and almost all within 1e-6.
- **Open loop over many steps**: errors stay bounded.
- **Closed loop**: trajectories diverge, as with any two floating-point implementations, so comparisons should be statistical.

See `docs/architecture.md` and `experiments/nvcc_probe/`.

## Repository layout

| Path | Contents |
|---|---|
| `px4/` | Vendored PX4 v1.16 sources (annotated), wrappers and shims; changes listed in `px4/PROVENANCE.md` |
| `include/px4rl/`, `src/` | Fleet, C API, glue modules, Gazebo motor model, frames, airframes |
| `tests/` | Ported PX4 and px4_stack tests, fleet, parity, plant, C API, CMake-consumer tests |
| `bench/` | GPU throughput benchmark |
| `tools/annotate_hd.py` | Inserts the execution-space macro into vendored sources |
| `tools/extract_ulog_trace.py` | Extracts the inner-loop trace used by the L3 replay test from a PX4 ULog |
| `experiments/nvcc_probe/` | The original device-compilation feasibility probe |
| `docs/` | Architecture, Iris parameter sources |

## Authors

px4-rl was conceived and directed by Zhaohong Liu and implemented together with Claude (Anthropic), an AI coding agent. See `AUTHORS.md`.

## License

px4-rl is released under the BSD 3-Clause License (`LICENSE`). The vendored PX4 code is BSD 3-Clause. The Gazebo motor-model port is Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
