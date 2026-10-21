# Architecture

## Goals

- **Fidelity.** Behave like PX4-Autopilot v1.16.2 from the position controller down to the actuator outputs, and like PX4's Gazebo Classic SITL plant from the actuator outputs to forces. v1.16.0 and v1.16.2 are identical in every module used here.
- **One source, two targets.** The same C++ code runs as a CPU library and as CUDA kernels; no algorithm is written twice.
- **GPU-resident batching.** Step N vehicles per call on device buffers the caller owns, with no host round trip, no allocation and no synchronization.
- **No framework dependencies.** No PX4 runtime, simulator, Python or PyTorch is needed at build time.

Out of scope: state estimation (EKF2), commander, navigator and flight tasks, takeoff and landing logic, and mixers other than multicopter.

## Layers

```
trajectory setpoint (pos/vel/acc + yaw; NaN = unused)      vehicle state
          │                                                     │
  PositionModule ── PositionControl + ControlMath  ◄────────────┤  position, velocity, heading
   (velocity notch / low-pass, velocity derivative)             │
   HoverThrustModule ── ZeroOrderHoverThrustEkf    ◄────────────┤  vertical acceleration
          │ attitude setpoint (q_d, thrust_body)                │
  AttitudeControl                                  ◄────────────┤  attitude
          │ rate setpoint                                       │
  RateControl (+ yaw-torque low-pass) ◄─ saturation feedback ─┐ │  body rates
  ControlAllocation (sequential desaturation, airmode) ───────┘ │
  ActuatorOutput (thrust-curve inverse, PWM mapping)            │
          │ actuator values (PWM)                               │
  Gazebo motor model (rotor-speed lag, thrust, drag torque, rotor drag, rolling moment)
          │ body wrench / per-rotor forces
       simulator
```

Each controller block is PX4 source code, vendored with only an execution-space macro added (`px4/PROVENANCE.md`). The two `*Module` classes reproduce the in-flight paths of PX4's `MulticopterPositionControl` and `MulticopterHoverThrustEstimator`. Each block names the PX4 lines it mirrors.

The caller chooses the control interface by choosing where to enter:
- **Position + yaw, velocity + yaw, acceleration + yaw**: a trajectory setpoint with NaN in the unused fields, as PX4 does.
- **Collective thrust + attitude**: an attitude setpoint.
- **Collective thrust + body rates**: a rate setpoint.
- **Motor commands**: bypass the controllers.

Loop rates and decimation are also the caller's choice. Each loop is a separate call.

## One source, two builds

Every function in the vendored code and in px4-rl's own headers is prefixed with `PX4RL_HD`.

**CPU build.**
- `PX4RL_HD` is empty, and the sources compile into `px4rl::px4` (the controllers) and `px4rl::fleet` (a batched CPU fleet).
- Both are exported as a CMake package.

**CUDA build.**
- `PX4RL_HD` is `__host__ __device__`, and one translation unit includes all sources.
- It produces `libpx4rl_cuda.so`, which has hidden visibility and exports only the `px4rl_*` C functions. That lets it coexist with the CPU library in one process.
- The CUDA runtime is linked statically. The library does not depend on the Python or PyTorch version.

`px4rl::Vehicle` holds the per-vehicle state. It is the only place where the controller chain is composed. The CPU fleet loops over vehicles, and each CUDA kernel thread steps one vehicle. Both call the same `Vehicle` member functions.

## State and memory

Vehicles live as persistent objects in device memory:
- A reset kernel constructs them with placement new. It runs on a mask and takes per-vehicle parameters, which gives domain randomization at reset.
- Step kernels then update them in place.

This keeps PX4's private controller state intact: integrators, filters, the allocator's mix matrix, and hysteresis timers. It also computes the allocator's pseudo-inverse once per reset rather than once per step.

Consequences:
- **CPU and GPU instances are separate.** A device object cannot be copied to the host.
- **No checkpointing of controller state.** Controller state is reset with the episode.
- **Stack.** Each thread needs about 1.9 KB of stack for the full chain. The CUDA runtime raises `cudaLimitStackSize` as needed. px4-rl never sets that limit itself, because it is process-wide.

## Frames

PX4-side quantities use world NED and body FRD, with Hamilton quaternions (w, x, y, z). The Gazebo motor model works in body FLU, like Gazebo itself. `frames.hpp` converts vectors, rates, quaternions and heading between ENU/FLU (Isaac Lab, Gazebo) and NED/FRD.

## Iris airframe

The controller side and the plant side deliberately use different sources:

| Side | Source |
|---|---|
| Controller | PX4 v1.16.2 defaults plus the `10015_gazebo-classic_iris` airframe file (`CA_ROTOR*` geometry, KM ±0.05, CT 6.5) |
| Plant | the Gazebo Classic Iris SDF at the commit PX4 v1.16.2 pins: mass 1.5 kg, inertia, rotor poses ±0.13/±0.22/±0.20, `motorConstant` 5.84e-6, `momentConstant` 0.06, time constants 0.0125/0.025 s, `maxRotVelocity` 1100, input mapping `ω = 1000·u + 100` |

PX4 SITL itself flies with this mismatch, and also with `THR_MDL_FAC = 0` against a plant whose thrust is quadratic in rotor speed. px4-rl reproduces both by default. Every number is tabulated with its source in `docs/iris-params-sources.md`.

## CPU vs GPU parity

CUDA's transcendental functions (`sinf`, `acosf`, `atan2f`, …) differ from glibc's in the last ULPs, and FMA contraction adds further differences, so the two builds cannot be bitwise identical. `tests/parity/` defines and enforces agreement at three levels. It runs on 4096 vehicles, with per-vehicle parameters and a masked reset half-way through.

| Level | Requirement | Measured (RTX 4090, CUDA 12.4, default flags) |
|---|---|---|
| Single step, identical state (also the first step after a reset) | motors, torque \|Δ\| ≤ 1e-3; rate setpoint \|Δ\|/max(1, \|v\|) ≤ 1e-3; PWM ≤ 1 µs; ≥ 95 % of values within 1e-6 | motors 2.8e-5, torque 3.5e-5, rate setpoint 2.4e-4, PWM 0; 97.6–99.9 % within 1e-6 |
| Open loop, 600 steps | finite, motors \|Δ\| ≤ 0.1 | motors 0.042, PWM 36 µs |
| Closed loop | compare statistics, not trajectories | — |

The open-loop growth is driven by discontinuities: the allocator's saturation flags flip at their ε threshold, and they feed the rate controller's anti-windup. The suite also runs with `PX4RL_STRICT_FP=ON`, which disables FMA contraction on both host and device.

## Performance

Benchmark: `bench/`, RTX 4090, CUDA 12.4. Times are µs per call for the whole fleet.

| Vehicles | position step | attitude step | rate → allocator → output step |
|---|---|---|---|
| 4 096 | 19 | 4.5 | 49 |
| 16 384 | 49 | 9.4 | 75 |
| 65 536 | 202 | 29 | 370 |

`Vehicle` is stored array-of-structures. Since the position loop was added it holds about 2 KB of controller state, and a rate step touches only part of it. Specialising the allocator to the actual rotor count and laying out hot state separately are the main levers for further speed.

## Verification

1. **Source identity.** Vendored files differ from upstream PX4 only by the `PX4RL_HD` macro and the edits listed in `px4/PROVENANCE.md`.
2. **Host behaviour.**
   - The extraction tests: L2 equivalence vectors and an L3 replay of a PX4 SITL log.
   - The upstream PX4 unit tests: `PositionControlTest`, `ControlMathTest`, `zero_order_hover_thrust_ekf_test`.
   - px4-rl's own module, plant and frame tests.
3. **CPU vs GPU parity**, as above.
4. **Packaging.** A CMake-consumer test installs px4-rl and builds an external project against it.
