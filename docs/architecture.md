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

The airframe is available as three functions, exported by `libpx4rl_cuda.so` as well as by the CPU library:
- `px4rl_params_iris()`: the controller side. The defaults plus the `CA_ROTOR*` geometry of the airframe file. Its rotor positions, KM, axes and rotor count are those of `createQuadXIris()`, the default geometry. The only difference is CT, 6.5 (the control allocator's default, which the airframe leaves alone) against 1.0; the allocator normalises its mix matrix, so the motor commands are the same (checked in `tests/fleet`). The airframe's other lines (`MAV_TYPE`, `IMU_GYRO_RATEMAX`, `NAV_ACC_RAD`, `RTL_*`, `GPS_UBX_DYNMODEL`, `EKF2_RNG_FOG`, `MC_AT_EN`) do not affect the controller chain and are ignored; it sets no `MC_*`, `MPC_*` or `HTE_*` parameter.
- `px4rl_plant_params_iris()` and `px4rl_rigid_body_params_iris()`: the plant side, from the SDF. PX4 output `i` drives SDF `rotor_i`.

**Plant in the fleet.** Every vehicle owns a `VehiclePlant` (the plant parameters and the speed state of its rotors) in an array next to the `Vehicle` array, so a controller reset does not touch the rotors and `rotor_step` does not touch the controller state. A new fleet holds the Iris plant with the rotors at rest. `px4rl_cuda_rotor_step` scales the PWM from `rate_step` to the actuator value u = (pwm − 1000) / 1000, as PX4's simulation PWM driver does, runs the Gazebo motor model for each vehicle, and returns the force and torque on the body in FLU (and optionally the force of each rotor). The caller supplies the body velocity relative to the air and, optionally, the body rates (which add the ω × r term to each rotor's air velocity), and integrates the rigid body.

**Closed loop.** `tests/closed_loop/iris_closed_loop_test.cpp` flies the CPU fleet against a 250 Hz ENU/FLU rigid body with the Iris mass and inertia, through the frame conversions. With the default `MPC_THR_HOVER = 0.5` and a plant that hovers at u = 0.6935:

| Test | Result |
|---|---|
| Position hold at 10 m, 20 s | horizontal drift < 1e-4 m; \|altitude error\| ≤ 0.13 m after 5 s; no setpoint fallback |
| Hover thrust estimate | 0.6938, against a measured collective thrust of 0.6935 (the estimator's ceiling is 0.7) |
| 1 m/s velocity step from hover | no overshoot; error 0.012 m/s at 8 s; altitude deviation 0.011 m; yaw error < 1e-4 rad |
| Acceleration hold with a +1 rad yaw step | yaw error 0.001 rad after 3 s |

## CPU vs GPU parity

CUDA's transcendental functions (`sinf`, `acosf`, `atan2f`, …) differ from glibc's in the last ULPs, and FMA contraction adds further differences, so the two builds cannot be bitwise identical. `tests/parity/` defines and enforces agreement at three levels. It runs on 4096 vehicles, with per-vehicle parameters and a masked reset half-way through.

| Level | Requirement | Measured (RTX 4090, CUDA 12.4, default flags) |
|---|---|---|
| Single step, identical state (also the first step after a reset) | motors, torque \|Δ\| ≤ 1e-3; rate setpoint \|Δ\|/max(1, \|v\|) ≤ 1e-3; PWM ≤ 1 µs; ≥ 95 % of values within 1e-6 | motors 2.8e-5, torque 3.5e-5, rate setpoint 2.4e-4, PWM 0; 97.6–99.9 % within 1e-6 |
| Open loop, 600 steps | finite, motors \|Δ\| ≤ 0.1 | motors 0.042, PWM 36 µs |
| Closed loop | compare statistics, not trajectories | — |

The open-loop growth is driven by discontinuities: the allocator's saturation flags flip at their ε threshold, and they feed the rate controller's anti-windup. The suite also runs with `PX4RL_STRICT_FP=ON`, which disables FMA contraction on both host and device.

## Performance

Benchmark: `bench/`, RTX 4090, CUDA 12.4. Times are µs per call for the whole fleet. They come from a shared GPU and vary by about ±20 % between runs; the rate step measured 49–64 µs at 4 096 vehicles.

| Vehicles | position step | attitude step | rate → allocator → output step |
|---|---|---|---|
| 4 096 | 19 | 4.5 | 49 |
| 16 384 | 49 | 9.4 | 75 |
| 65 536 | 202 | 29 | 370 |

The rotor step (`px4rl_cuda_rotor_step`, PWM to wrench) takes 5.2 µs at 4 096 vehicles, 9.7 µs at 16 384 and 48 µs at 65 536.

`Vehicle` is stored array-of-structures. Since the position loop was added it holds about 2 KB of controller state, and a rate step touches only part of it. Specialising the allocator to the actual rotor count and laying out hot state separately are the main levers for further speed.

## Verification

1. **Source identity.** Vendored files differ from upstream PX4 only by the `PX4RL_HD` macro and the edits listed in `px4/PROVENANCE.md`.
2. **Host behaviour.**
   - The extraction tests: L2 equivalence vectors and an L3 replay of a PX4 SITL log.
   - The upstream PX4 unit tests: `PositionControlTest`, `ControlMathTest`, `zero_order_hover_thrust_ekf_test`.
   - px4-rl's own module, plant and frame tests.
3. **CPU vs GPU parity**, as above.
4. **Packaging.** A CMake-consumer test installs px4-rl and builds an external project against it.
