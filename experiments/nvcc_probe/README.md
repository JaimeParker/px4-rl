# nvcc device-compilation probe

This is the feasibility experiment behind px4-rl's GPU path. It asks two
questions:
- Can PX4's inner-loop controllers (attitude, rate, control allocator, actuator output, from PX4 v1.16) be compiled as `__host__ __device__` code, with no change beyond an execution-space macro?
- Does the CUDA build then compute the same thing as the host build?

It is kept for reproducibility. The maintained equivalents are `tests/parity/` (CPU vs GPU parity) and `bench/` (throughput).

## How it works

`probe.cu` unity-includes the ten `.cpp` files from `px4/` and compiles them with `PX4RL_HD=__host__ __device__`. A `Drone` holds:
- the four controller wrappers;
- a second allocator, reached only through a `ControlAllocation*` base pointer, to exercise virtual dispatch on the device;
- a toy rigid-body attitude plant.

The same `Drone::step` runs as a host loop and as a CUDA kernel over N drones:
- Device drones are built by placement-new in an init kernel and persist across launches.
- A masked re-init kernel resets every 7th drone half-way through the run.
- Each step, host and device outputs are compared and accumulated into |host − device| histograms.
- Two modes run:
  - **Open loop**: state drawn from an RNG, so outputs cannot feed back.
  - **Closed loop**: state integrated through the plant.

```bash
experiments/nvcc_probe/run_probe.sh 89 4096 600   # sm arch, drones, steps; outputs in out/
```

The script builds two variants:
- **strict**: device `--fmad=false`, host `-ffp-contract=off`;
- **default**: no floating-point flags.

## Results

Measured on an RTX 4090 (sm_89) with nvcc 12.4.131 and driver 570.133.07, N = 4096, 600 steps.

- **Compilation.** Compiling for the device needed only:
  - the `PX4RL_HD` macro on function heads (543 heads, inserted by `tools/annotate_hd.py`);
  - a device-safe `PX4_ERR` in place of two `fprintf(stderr, …)` calls;
  - `--expt-relaxed-constexpr`.

  Neither build emitted warnings, including nvcc #20011/#20014 (host function called from host-device code). Every px4 file and the full change list are in `px4/PROVENANCE.md`.
- **Device objects.** Placement-new objects persist across launches. Virtual calls through a base pointer written by the init kernel work in later launches.
- **Parity is not bitwise.** CUDA's `sinf`, `cosf`, `acosf` and `atan2f` differ from glibc's in the last ULPs, and FMA contraction adds more. Max |host − device| per field:

  | | strict, open loop | strict, closed loop | default, open loop | default, closed loop |
  |---|---|---|---|---|
  | step 0: motors / torque_sp / rates_sp | 3.2e-6 / 8.8e-6 / 5.0e-5 | 9.7e-5 / 6.8e-5 / 3.9e-4 | 4.9e-6 / 6.4e-5 / 4.6e-4 | 1.4e-4 / 9.7e-5 / 5.5e-4 |
  | step 0: PWM (µs) | 0 | 0 | 0 | 1 |
  | 600 steps: motors / torque_sp / rates_sp | 4.4e-2 / 1.0e-2 / 1.6e-3 | diverged (≈1) | 3.9e-2 / 1.1e-2 / 1.8e-3 | diverged (≈1) |
  | 600 steps: PWM (µs) | 37 | 985 | 32 | 1000 |

  - **Single step.** Most values agree to 1e-6. The rare larger rates_sp differences occur at large attitude errors. They are consistent with ill-conditioning of the attitude law (acos/asin argument near ±1), but this was not traced further.
  - **Open loop.** Differences stay bounded and come from discontinuities. The allocator's saturation flags flip at their ε threshold and feed the rate controller's integrator anti-windup.
  - **Closed loop.** Trajectories diverge, as with any two floating-point implementations. No NaNs appeared.

  These numbers motivated the parity definition used by `tests/parity/`: single-step tolerances, bounded open-loop error, and statistical comparison in closed loop.
- **Memory and throughput.**
  - The kernel's stack frame is 1872 B per thread, above the 1024 B default `cudaLimitStackSize`. The CUDA runtime grows the limit automatically.
  - Full chain step, including the toy plant and the duplicate allocator: 132 µs at N = 4096, 161 µs at 16384, 745 µs at 65536.
  - Going from 4096 to 16384 vehicles costs only 1.2× the time, so most of the time is per-thread latency rather than throughput.
