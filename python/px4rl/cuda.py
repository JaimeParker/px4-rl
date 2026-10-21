"""Batched GPU fleet on torch CUDA tensors. Importing this module imports torch.

`CudaFleet` drives `libpx4rl_cuda.so` through ctypes. Every step call only launches kernels on the current torch
stream. It allocates nothing and never synchronizes, so a chain of steps can be captured in a CUDA graph
(`torch.cuda.CUDAGraph`). Scalars such as `dt` and `time_us` are baked into a captured graph.
"""
import ctypes

import torch

from . import _lib
from ._results import AttitudeResult, PositionResult, RateResult, RotorResult
from ._structs import Params, PlantParams
from .params import struct_buffer

__all__ = ["CudaFleet"]


def _device_of(device):
    """Resolves None, an int, a string or a torch.device to a torch.device with a CUDA index."""
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        device = torch.device("cuda", device)
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"CudaFleet needs a CUDA device, got {device}")
    return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)


class CudaFleet:
    """N vehicles whose controllers and rotors live in GPU memory.

    Inputs are torch CUDA tensors on the fleet's device: float32 (uint8 for masks), contiguous, with the shapes
    below; anything else raises ValueError. Outputs go into buffers the fleet owns and the call returns them, so
    the next call to the same method overwrites them (clone to keep). Pass `out=` (an object with the result's
    attributes, such as the result tuple) to write into your own tensors instead.
    """

    def __init__(self, n, device=None):
        if n < 1:
            raise ValueError("n must be >= 1")
        self._lib = _lib.load_cuda_library()
        self._device = _device_of(device)
        self._n = int(n)
        self._handle = ctypes.c_void_p()
        self._check(self._lib.px4rl_cuda_fleet_create(self._n, self._device.index, ctypes.byref(self._handle)))

        def buffer(*shape, dtype=torch.float32):
            return torch.zeros((self._n, *shape), dtype=dtype, device=self._device)

        self._position_out = PositionResult(buffer(4), buffer(3), buffer(), buffer(), buffer(dtype=torch.uint8))
        self._attitude_out = AttitudeResult(buffer(3), buffer(3))
        self._rate_out = RateResult(buffer(4), buffer(4), buffer(3))
        self._rotor_out = RotorResult(buffer(3), buffer(3), buffer(4, 3))
        self._last_thrust_sp_body = None

    def __len__(self):
        return self._n

    @property
    def n(self):
        return self._n

    @property
    def device(self):
        return self._device

    def close(self):
        """Frees the GPU memory of the fleet. The fleet cannot be used afterwards."""
        handle, self._handle = self._handle, None
        if handle is not None and handle.value:
            self._check(self._lib.px4rl_cuda_fleet_destroy(handle))

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown, or a CUDA context that is already gone
            pass

    # ---- checks -------------------------------------------------------------------------------------------

    def _check(self, status):
        if status != 0:
            raise RuntimeError(self._lib.px4rl_last_error().decode(errors="replace"))

    def _live_handle(self):
        if self._handle is None:
            raise RuntimeError("the fleet is closed")
        return self._handle

    def _stream(self):
        return torch.cuda.current_stream(self._device).cuda_stream

    def _address(self, tensor, name, shape, dtype=torch.float32, optional=False):
        """Validates `tensor` and returns its device address, or None for an allowed None."""
        if tensor is None:
            if optional:
                return None
            raise ValueError(f"{name} is required")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.device != self._device:
            raise ValueError(f"{name} is on {tensor.device}, the fleet is on {self._device}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return tensor.data_ptr()

    def _outputs(self, out, owned, shapes, optional):
        """Returns (result, addresses): `owned` when out is None, otherwise `out` after checking its tensors."""
        if out is None:
            out = owned
        result_type = type(owned)
        addresses = []
        for field, shape in zip(result_type._fields, shapes):
            tensor = getattr(out, field, None)
            dtype = torch.uint8 if field == "outcome" else torch.float32
            addresses.append(self._address(tensor, f"out.{field}", shape, dtype, optional=field in optional))
        return out, addresses

    # ---- resets -------------------------------------------------------------------------------------------

    def reset(self, params, mask=None):
        """Reconstructs the controllers of the vehicles selected by `mask` (uint8 (n,); None selects all).

        `params` is a `Params` (broadcast) or a ctypes array of n `Params` (one per vehicle), for example from
        `params_array`. The rotors are not touched.
        """
        handle = self._live_handle()
        address, count = struct_buffer(params, Params, "params")
        mask_address = self._address(mask, "mask", (self._n,), torch.uint8, optional=True)
        self._check(self._lib.px4rl_cuda_fleet_reset(handle, mask_address, address, count, self._stream()))

    def reset_plant(self, plants, mask=None):
        """Replaces the rotor plant of the selected vehicles and puts their rotors at rest.

        `plants` is a `PlantParams` (broadcast) or a ctypes array of n of them. A new fleet holds the Iris plant.
        """
        handle = self._live_handle()
        address, count = struct_buffer(plants, PlantParams, "plants")
        mask_address = self._address(mask, "mask", (self._n,), torch.uint8, optional=True)
        self._check(self._lib.px4rl_cuda_fleet_reset_plant(handle, mask_address, address, count, self._stream()))

    # ---- steps --------------------------------------------------------------------------------------------

    def position_step(self, position, velocity, heading, acceleration_z, trajectory_sp, dt, time_us, in_air=None,
                      *, out=None):
        """Position and velocity loop, with the hover thrust estimator (see `px4rl_cuda_position_step`).

        position, velocity (n, 3) NED; heading, acceleration_z (n,) with acceleration_z the NED vertical
        acceleration; trajectory_sp (n, 11) from `px4rl.setpoints`; in_air uint8 (n,) or None (all in the air).
        Returns a `PositionResult`. The thrust it contains is the default `thrust_sp_body` of `attitude_step`.
        """
        handle = self._live_handle()
        n = self._n
        addresses = [
            self._address(position, "position", (n, 3)),
            self._address(velocity, "velocity", (n, 3)),
            self._address(heading, "heading", (n,)),
            self._address(acceleration_z, "acceleration_z", (n,)),
            self._address(trajectory_sp, "trajectory_sp", (n, 11)),
            self._address(in_air, "in_air", (n,), torch.uint8, optional=True),
        ]
        result, outputs = self._outputs(out, self._position_out, [(n, 4), (n, 3), (n,), (n,), (n,)],
                                        optional=("hover_thrust", "outcome"))
        self._check(self._lib.px4rl_cuda_position_step(handle, *addresses, float(dt), int(time_us), *outputs,
                                                       self._stream()))
        self._last_thrust_sp_body = result.thrust_sp_body
        return result

    def attitude_step(self, q, q_sp=None, thrust_sp_body=None, yawspeed_ff=None, *, out=None):
        """Attitude loop (see `px4rl_cuda_attitude_step`). q (n, 4) is the attitude, (w, x, y, z).

        q_sp=None uses the attitude setpoint (and yaw rate) of the last `position_step`; with an explicit q_sp
        (n, 4), yawspeed_ff (n,) or None (zero) is the yaw rate feed-forward. thrust_sp_body=None uses the thrust
        of the last `position_step`. Returns an `AttitudeResult`.
        """
        handle = self._live_handle()
        n = self._n
        if thrust_sp_body is None:
            if self._last_thrust_sp_body is None:
                raise ValueError("thrust_sp_body is required before the first position_step")
            thrust_sp_body = self._last_thrust_sp_body
        addresses = [
            self._address(q, "q", (n, 4)),
            self._address(q_sp, "q_sp", (n, 4), optional=True),
            self._address(yawspeed_ff, "yawspeed_ff", (n,), optional=True),
            self._address(thrust_sp_body, "thrust_sp_body", (n, 3)),
        ]
        result, outputs = self._outputs(out, self._attitude_out, [(n, 3), (n, 3)], optional=())
        self._check(self._lib.px4rl_cuda_attitude_step(handle, *addresses, *outputs, self._stream()))
        return result

    def rate_step(self, rates_sp, thrust_body, rates, angular_accel, dt, landed=None, *, out=None):
        """Rate controller, allocator and actuator output (see `px4rl_cuda_rate_step`).

        All inputs (n, 3) in body FRD; landed uint8 (n,) or None. Returns a `RateResult`; `actuator` is what
        `rotor_step` takes.
        """
        handle = self._live_handle()
        n = self._n
        addresses = [
            self._address(rates_sp, "rates_sp", (n, 3)),
            self._address(thrust_body, "thrust_body", (n, 3)),
            self._address(rates, "rates", (n, 3)),
            self._address(angular_accel, "angular_accel", (n, 3)),
        ]
        landed_address = self._address(landed, "landed", (n,), torch.uint8, optional=True)
        result, outputs = self._outputs(out, self._rate_out, [(n, 4), (n, 4), (n, 3)],
                                        optional=("actuator", "torque_sp"))
        self._check(self._lib.px4rl_cuda_rate_step(handle, *addresses, float(dt), landed_address, *outputs,
                                                   self._stream()))
        return result

    def rotor_step(self, actuator, body_velocity_flu, dt, body_rates_flu=None, armed=None, *, out=None):
        """Gazebo rotor model (see `px4rl_cuda_rotor_step`): PWM to the wrench on the body, body FLU.

        actuator (n, 4) PWM from `rate_step`; body_velocity_flu (n, 3) relative to the air; body_rates_flu
        (n, 3) or None; armed uint8 (n,) or None (all armed). Returns a `RotorResult`.
        """
        handle = self._live_handle()
        n = self._n
        addresses = [
            self._address(actuator, "actuator", (n, 4)),
            self._address(armed, "armed", (n,), torch.uint8, optional=True),
            self._address(body_velocity_flu, "body_velocity_flu", (n, 3)),
            self._address(body_rates_flu, "body_rates_flu", (n, 3), optional=True),
        ]
        result, outputs = self._outputs(out, self._rotor_out, [(n, 3), (n, 3), (n, 4, 3)],
                                        optional=("rotor_force",))
        self._check(self._lib.px4rl_cuda_rotor_step(handle, *addresses, float(dt), *outputs, self._stream()))
        return result
