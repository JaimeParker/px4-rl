# Provenance of `px4/`

This directory has two origins.

1. **Inner loop**: attitude, rate, allocator, actuator output, and the matrix/mathlib subset. It comes from *px4_stack*, an earlier standalone extraction of the PX4-Autopilot **v1.16.0** inner-loop modules by the px4-rl author (not published separately).
   - px4_stack removed the PX4 framework dependencies (uORB, parameter system, work queues) and added data-driven wrappers (`*Wrapper.*`, `*Types.hpp`) and the thrust-curve/PWM output stage.
   - It ships with its own test suite: unit tests, L2 equivalence vectors, and an L3 replay of a PX4 SITL log, in `tests/cpp/`.
2. **Position loop and hover-thrust estimator**: vendored directly from PX4-Autopilot **v1.16.2**.

The control-related modules are identical in v1.16.0 and v1.16.2.

| Directory | PX4 source |
|---|---|
| `lib/matrix/matrix/` | `src/lib/matrix/matrix/` |
| `lib/mathlib/math/` | `src/lib/mathlib/math/` (subset) |
| `lib/px4_platform_common/defines.h` | minimal shim (px4_stack) |
| `attitude_control/` | `src/modules/mc_att_control/AttitudeControl/` + px4_stack wrapper |
| `rate_control/` | `src/lib/rate_control/` + px4_stack wrapper |
| `control_allocator/` | `src/modules/control_allocator/` (subset) + px4_stack wrapper |
| `actuator_output/` | px4_stack (thrust-curve inverse, PWM/DShot mapping) |
| `position_control/` | `src/modules/mc_pos_control/PositionControl/` (v1.16.2), see below |
| `hover_thrust_estimator/` | `src/modules/mc_hover_thrust_estimator/` (v1.16.2), see below |

Changes made in px4-rl to the px4_stack sources:
1. `PX4RL_HD` was inserted before every function declaration and definition head by `tools/annotate_hd.py` (543 heads). The build defines `PX4RL_HD` as `__host__ __device__` for CUDA and as empty otherwise.
2. `ActuatorEffectiveness.cpp`: two `fprintf(stderr, ...)` calls became `PX4_ERR(...)`. `defines.h` gained a `PX4_ERR` macro that uses `printf` in device code and `fprintf(stderr)` on the host.
3. Copyright headers of px4_stack-authored files (the wrappers, `*Types.hpp`, the output stage, the `defines.h` shim and px4_stack's tests) now name Zhaohong Liu. They previously named "PX4 Extraction Project" or "PX4 Development Team", which was incorrect for these files. Files containing upstream PX4 code keep the PX4 Development Team notice.

Any further change must be listed here.

## Position loop and hover-thrust estimator (PX4 v1.16.2)

Everything below is from PX4-Autopilot at tag `v1.16.2`. Each vendored file is
byte-identical to upstream except for the `PX4RL_HD ` insertions made by
`python3 tools/annotate_hd.py PX4RL_HD <files>` (140 heads in total). Checked with
`diff <(sed 's/PX4RL_HD //g' px4/<file>) <PX4-Autopilot v1.16.2>/src/<source>`: no differences for any file.
No edit was needed for nvcc: all vendored `.cpp` files compile in the CUDA unity build (`src/cuda/px4rl_cuda.cu`),
and a throw-away kernel that called `PositionModule::update` and `HoverThrustModule::update` also compiled.

### Vendored files

| Destination | Upstream source (`src/...`) | `PX4RL_HD` heads |
|---|---|---|
| `position_control/PositionControl.hpp` | `modules/mc_pos_control/PositionControl/PositionControl.hpp` | 20 |
| `position_control/PositionControl.cpp` | `modules/mc_pos_control/PositionControl/PositionControl.cpp` | 14 |
| `position_control/ControlMath.hpp` | `modules/mc_pos_control/PositionControl/ControlMath.hpp` | 8 |
| `position_control/ControlMath.cpp` | `modules/mc_pos_control/PositionControl/ControlMath.cpp` | 8 |
| `hover_thrust_estimator/zero_order_hover_thrust_ekf.hpp` | `modules/mc_hover_thrust_estimator/zero_order_hover_thrust_ekf.hpp` | 29 |
| `hover_thrust_estimator/zero_order_hover_thrust_ekf.cpp` | `modules/mc_hover_thrust_estimator/zero_order_hover_thrust_ekf.cpp` | 15 |
| `lib/mathlib/mathlib.h` | `lib/mathlib/mathlib.h` | 0 |
| `lib/mathlib/math/SearchMin.hpp` | `lib/mathlib/math/SearchMin.hpp` | 2 |
| `lib/mathlib/math/TrajMath.hpp` | `lib/mathlib/math/TrajMath.hpp` | 4 |
| `lib/mathlib/math/Utilities.hpp` | `lib/mathlib/math/Utilities.hpp` | 13 |
| `lib/mathlib/math/filter/NotchFilter.hpp` | `lib/mathlib/math/filter/NotchFilter.hpp` | 14 |
| `lib/slew_rate/SlewRate.hpp` | `lib/slew_rate/SlewRate.hpp` | 5 |
| `lib/hysteresis/hysteresis.h` | `lib/hysteresis/hysteresis.h` | 5 |
| `lib/hysteresis/hysteresis.cpp` | `lib/hysteresis/hysteresis.cpp` | 3 |

`mathlib.h` brings in `SearchMin.hpp`, `TrajMath.hpp` and `Utilities.hpp` (the other two mathlib headers were
already vendored). `SlewRate.hpp` is used by the glue for the tilt limit, `hysteresis` by the hover thrust validity.
`PositionControl::empty_trajectory_setpoint` stays a host-only static data member; device code uses
`px4rl::emptyTrajectorySetpoint()` (`include/px4rl/position_module.hpp`), which has the same value.

### Upstream tests

Copied byte-identical to `tests/cpp/`: `PositionControlTest.cpp`, `ControlMathTest.cpp` (both from
`modules/mc_pos_control/PositionControl/`) and `zero_order_hover_thrust_ekf_test.cpp`
(from `modules/mc_hover_thrust_estimator/`). Not compiled into the library.
`ControlMathTest.cpp` uses the `M_*_F` float constants through `<px4_platform_common/defines.h>`; the px4-rl shim of
that header does not have them, so `tests/cpp/CMakeLists.txt` force-includes `px4/lib/px4_defines.h` for that one target.

### Shims (new px4-rl files, not upstream code)

| File | Replaces | Content |
|---|---|---|
| `lib/uORB/topics/trajectory_setpoint.h` | generated `trajectory_setpoint_s` | plain struct, fields of `msg/versioned/TrajectorySetpoint.msg` in order, `timestamp` first; the generator's trailing padding member is omitted |
| `lib/uORB/topics/vehicle_attitude_setpoint.h` | generated `vehicle_attitude_setpoint_s` | same, from `msg/versioned/VehicleAttitudeSetpoint.msg` |
| `lib/uORB/topics/vehicle_local_position_setpoint.h` | generated `vehicle_local_position_setpoint_s` | same, from `msg/VehicleLocalPositionSetpoint.msg` |
| `lib/geo/geo.h` | `lib/geo/geo.h` | only `CONSTANTS_ONE_G`, copied verbatim from `geo.h:54` |
| `lib/drivers/drv_hrt.h` | `drivers/drv_hrt.h` | only `typedef uint64_t hrt_abstime;` (`drv_hrt.h:62`), for `lib/hysteresis`; no clock functions |
| `lib/px4_defines.h` | `platforms/common/include/px4_defines.h` | includes `px4_platform_common/defines.h`, adds the `M_*_F` constants and `M_DEG_TO_RAD`/`M_RAD_TO_DEG` from upstream `defines.h:119-150`, each under `#ifndef` |

The existing `lib/px4_platform_common/defines.h` shim was not modified, so `NotchFilter.hpp` (which uses `M_PI_F`)
only compiles when `px4_defines.h` was included first. `include/px4rl/position_module.hpp` does that.
Moving the `M_*_F` block into `defines.h` would make `px4_defines.h` and the forced include unnecessary.

Include paths added to `px4rl_px4_headers` (`px4/CMakeLists.txt`): `px4/` (for `<lib/mathlib/mathlib.h>`),
`px4/lib/matrix` (for `<matrix/math.hpp>`), `px4/position_control`, `px4/hover_thrust_estimator`. Together with the
existing `px4/lib` they resolve `<mathlib/mathlib.h>`, `<geo/geo.h>` and `<uORB/topics/...>` as written upstream.
The four new `.cpp` files were added to `px4rl_px4` and to the CUDA unity build.

### px4_stack wrapper change (not PX4 algorithm code)

`control_allocator/ControlAllocatorTypes.hpp` gained `ControlAllocatorOutput::unallocated_thrust` (Vector3f), and
`control_allocator/ControlAllocatorWrapper.cpp` fills it from `unallocated(THRUST_X/Y/Z)`, the same expression that
fills `unallocated_torque`. Nothing else in the wrapper changed. The hover thrust estimator needs it
(`thrust_sp_body - unallocated_thrust`).

### Glue (new px4-rl code in `include/px4rl/`)

- `position_module.hpp`, `px4rl::PositionModule`: the in-flight path of
  `modules/mc_pos_control/MulticopterPositionControl.cpp`. Each block names the PX4 line range it mirrors.
- `hover_thrust_module.hpp`, `px4rl::HoverThrustModule`: `modules/mc_hover_thrust_estimator/MulticopterHoverThrustEstimator.cpp`,
  same convention. Time is a caller-provided microsecond counter; the validity hysteresis (2 s from false) uses it.
- `params.hpp` / `c_api.h`: `px4rl_params` gained the `mpc_*` and `hte_*` fields; defaults in `defaultParams()`.
  `MPC_ACC_DECOUPLE` (`multicopter_position_control_limits_params.c:152`) and `MPC_LAND_SPEED`
  (`multicopter_takeoff_land_params.c:111`) are not in `docs/iris-params-sources.md` section 4; their defaults come
  from those source lines.

### Behaviour excluded from MulticopterPositionControl

- Takeoff state machine (`TakeoffHandling`, ramp-up, spool-up, throw takeoff, `MPC_TKO_*`); the vehicle is always
  flying. The velocity limits are the flying-state values and `_control.setHoverThrust(MPC_THR_HOVER)` is applied
  only at configuration (`parameters_update`), not when not flying.
- Land detector input (`ground_contact`, `landed`) and the "not flying => zero-thrust setpoint" branch.
- Goto control (`GotoControl`).
- EKF-reset setpoint adjustment (`adjustSetpointForEKFResets`, the reset counters).
- `z_deriv` blending of the vertical velocity (`z_deriv` is vz in simulation).
- Offboard `want_takeoff` handling and `vehicle_constraints` input (constraints are the parameter defaults).
- The initial "no trajectory setpoint since position control started" failsafe, and the position-control-enabled
  mode flags. The disabled branch (`resetIntegral` when not enabled) does not exist.
- `vehicle_local_position.vxy_max` (treated as NaN), the xy/z validity flags (a non-finite input is the only
  invalid case), `MPC_TILTMAX_LND` and the "tilt constrained" clamps of `parameters_update` (:118-294:
  responsiveness, SYS_VEHICLE_RESP, MPC_XY_VEL_ALL, MPC_Z_VEL_ALL, tilt/hover/cruise constraining).
- Running-mean sample interval (`WelfordMean`): `configure()` takes a nominal interval (default 0.01 s, PX4's seed). In the fleets the interval is the per-vehicle parameter `position_dt` (default 0.02 s, a 50 Hz position loop).
- uORB publications, perf counters, `PX4_WARN`/`mavlink_log`/events output (the failsafe is silent).
- PX4 starts `_tilt_limit_slew_rate` at 0 and slews at 0.2 rad/s; this is kept. `setTiltLimitStateDeg()` lets a
  caller start from the configured limit.

### Behaviour excluded from MulticopterHoverThrustEstimator

- uORB topics: land detector, vehicle status, `parameter_update`; `armed` and `in_air` are caller flags. The
  `dist_bottom > 1` in-air heuristic and the `landed` handling are therefore not reproduced.
- The "thrust setpoint and attitude fresher than 20 ms" checks and the `control_allocator_status` subscription;
  the caller passes the unallocated thrust of the same step.
- Status publication (`publishStatus`, `publishInvalidStatus`, `hover_thrust_estimate_s`), perf counters.
- `hrt_absolute_time()`: the invalidating `set_state_and_update(false, ...)` uses the caller's time.
