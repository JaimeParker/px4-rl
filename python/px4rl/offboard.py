"""PX4 offboard control modes on a `CudaFleet`. Importing this module imports torch.

`OffboardController` runs the controller chain for one command mode per call, the way PX4 does for its offboard
modes: the trajectory modes (position, velocity, acceleration + yaw) go through `position_step`, attitude modes
skip it and give `attitude_step` an explicit setpoint, and the body-rate mode skips both and gives `rate_step` its
setpoint (the command-mode conventions of `include/px4rl/c_api.h`). It also keeps what every caller of the chain
needs besides the fleet: the time counter, the setpoint held between commands, and the vertical and angular
accelerations, which a simulator does not measure, as finite differences of the velocity and the body rates.

Everything is world NED, body FRD, Hamilton quaternions (w, x, y, z) and PX4 units, as in the fleet. The rotor
step takes and returns body FLU, also as in the fleet.

Call `command` once per position period and `physics` once per rate period (for example 50 Hz and 250 Hz):

    controller = OffboardController(fleet, position_dt=0.02, rate_dt=0.004)
    controller.command("velocity", state, trajectory_sp=px4rl.setpoints.velocity_yaw(v_ned, heading))
    for _ in range(5):
        wrench = controller.physics(state, body_velocity_flu, body_rates_flu)   # RotorResult, body FLU

The controller only launches torch operations and fleet kernels on the current stream; it never synchronizes.
Unlike the fleet's step calls, it allocates small temporaries, so it is not meant to be captured in a CUDA graph.
"""
import math
from typing import Any, NamedTuple

import torch

from ._results import RotorResult
from .cuda import CudaFleet

__all__ = ["MODES", "OffboardController", "VehicleState"]

MODES = ("position", "velocity", "acceleration", "attitude", "stabilized", "body_rate")
TRAJECTORY_MODES = ("position", "velocity", "acceleration")


class VehicleState(NamedTuple):
    """State of n vehicles: position (n, 3) and velocity (n, 3) in world NED, attitude q (n, 4) body FRD to NED
    (w, x, y, z), body rates (n, 3) FRD in rad/s."""

    position: Any
    velocity: Any
    q: Any
    rates: Any


def heading_of(q):
    """PX4 heading (yaw about NED down, 0 = north) of body FRD -> NED quaternions (n, 4)."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_euler(roll, pitch, yaw):
    """(w, x, y, z) of Rz(yaw) Ry(pitch) Rx(roll), PX4's Eulerf convention; angles (n,)."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), dim=-1)


def _wrap_pi(angle):
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class OffboardController:
    """Command modes of PX4's offboard interface over a `CudaFleet` (one mode for the whole fleet per command).

    Modes and the arguments of `command`:

    - ``"position" | "velocity" | "acceleration"``: ``trajectory_sp`` (n, 11) rows from `px4rl.setpoints`
      (any of the three may be mixed per vehicle; the mode name only selects the trajectory path). Optional
      ``in_air`` as for `position_step`.
    - ``"attitude"``: ``q_sp`` (n, 4) and collective ``thrust`` (n,) in [0, 1]; optional ``yawspeed_ff`` (n,)
      in rad/s (PX4's ``yaw_sp_move_rate``).
    - ``"stabilized"``: ``roll``, ``pitch`` (n,) in rad, ``yaw_rate`` (n,) in rad/s and ``thrust`` (n,). The yaw
      setpoint is integrated from ``yaw_rate`` over the position period, as in PX4's Stabilized mode, starting from
      the heading given to `reset`; the setpoint is ``Euler(roll, pitch, yaw_sp)`` (roll and pitch as Euler angles,
      not PX4's stick tilt-vector mapping), with ``yaw_rate`` as the feed-forward.
    - ``"body_rate"``: ``rates_sp`` (n, 3) FRD in rad/s and ``thrust`` (n,).

    A collective thrust T becomes the body thrust setpoint (0, 0, -T), normalized as in PX4. The setpoint of a
    command is held until the next one.
    """

    def __init__(self, fleet: CudaFleet, position_dt: float, rate_dt: float):
        if not isinstance(fleet, CudaFleet):
            raise TypeError(f"fleet must be a CudaFleet, got {type(fleet).__name__}")
        if position_dt <= 0 or rate_dt <= 0:
            raise ValueError("position_dt and rate_dt must be positive")
        n, device = fleet.n, fleet.device
        self.fleet = fleet
        self.position_dt = float(position_dt)
        self.rate_dt = float(rate_dt)
        self.time_us = 0
        self._dt_us = int(round(self.position_dt * 1e6))
        self.mode = None
        self.position_result = None

        def zeros(*shape):
            return torch.zeros((n, *shape), dtype=torch.float32, device=device)

        self._previous_velocity_down = zeros()
        self._previous_rates = zeros(3)
        self._yaw_sp = zeros()
        self._q_sp = zeros(4)
        self._q_sp[:, 0] = 1.0
        self._yawspeed_ff = zeros()
        self._thrust_body = zeros(3)
        self._rates_sp = zeros(3)

    @property
    def n(self):
        return self.fleet.n

    @property
    def hover_thrust(self):
        """Hover thrust estimate (n,) of the last trajectory-mode command, or None before the first one."""
        return None if self.position_result is None else self.position_result.hover_thrust

    @property
    def yaw_setpoint(self):
        """The integrated yaw setpoint (n,) of the stabilized mode, in rad."""
        return self._yaw_sp

    def reset(self, mask=None, heading=None):
        """Forgets the finite-difference history of the vehicles selected by `mask` (uint8 or bool (n,); None
        selects all), so the next accelerations do not see a teleport, and sets their stabilized yaw setpoint to
        `heading` (n,) if given, else to 0. The fleet's own `reset`/`reset_plant` stay the caller's."""
        selected = self._selection(mask)
        self._previous_velocity_down[selected] = 0.0
        self._previous_rates[selected] = 0.0
        self._yaw_sp[selected] = 0.0 if heading is None else self._column(heading, "heading")[selected]

    def command(self, mode, state: VehicleState, *, trajectory_sp=None, in_air=None, q_sp=None, thrust=None,
                yawspeed_ff=None, roll=None, pitch=None, yaw_rate=None, rates_sp=None):
        """Applies one command of `mode` (see the class docstring) for the next position period. Returns the
        `PositionResult` for trajectory modes, else None."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._check_state(state)
        velocity_down = state.velocity[:, 2]
        acceleration_down = (velocity_down - self._previous_velocity_down) / self.position_dt
        self._previous_velocity_down.copy_(velocity_down)
        self.time_us += self._dt_us
        self.mode = mode
        if mode in TRAJECTORY_MODES:
            if trajectory_sp is None:
                raise ValueError(f"mode {mode!r} needs trajectory_sp")
            self.position_result = self.fleet.position_step(
                state.position, state.velocity, heading_of(state.q), acceleration_down, trajectory_sp,
                self.position_dt, self.time_us, in_air)
            return self.position_result
        self._set_thrust(self._column(thrust, "thrust"))
        if mode == "attitude":
            self._q_sp.copy_(self._array(q_sp, "q_sp", 4))
            if yawspeed_ff is None:
                self._yawspeed_ff.zero_()
            else:
                self._yawspeed_ff.copy_(self._column(yawspeed_ff, "yawspeed_ff"))
        elif mode == "stabilized":
            yaw_rate = self._column(yaw_rate, "yaw_rate")
            self._yaw_sp.copy_(_wrap_pi(self._yaw_sp + yaw_rate * self.position_dt))
            self._q_sp.copy_(quat_from_euler(self._column(roll, "roll"), self._column(pitch, "pitch"), self._yaw_sp))
            self._yawspeed_ff.copy_(yaw_rate)
        else:  # body_rate
            self._rates_sp.copy_(self._array(rates_sp, "rates_sp", 3))
        return None

    def physics(self, state: VehicleState, body_velocity_flu, body_rates_flu=None, armed=None) -> RotorResult:
        """One rate period: attitude step (not in body_rate mode), rate step and rotor step. `body_velocity_flu`
        (n, 3) is the body velocity relative to the air in FLU, `body_rates_flu` (n, 3) or None the body rates in
        FLU, as for `rotor_step`. Returns the fleet's `RotorResult` (force and torque on the body, FLU)."""
        if self.mode is None:
            raise RuntimeError("command must be called before physics")
        self._check_state(state)
        angular_accel = (state.rates - self._previous_rates) / self.rate_dt
        self._previous_rates.copy_(state.rates)
        fleet = self.fleet
        if self.mode in TRAJECTORY_MODES:
            attitude = fleet.attitude_step(state.q)
            rates_sp, thrust_body = attitude.rates_sp, attitude.thrust_body
        elif self.mode == "body_rate":
            rates_sp, thrust_body = self._rates_sp, self._thrust_body
        else:
            attitude = fleet.attitude_step(state.q, q_sp=self._q_sp, thrust_sp_body=self._thrust_body,
                                           yawspeed_ff=self._yawspeed_ff)
            rates_sp, thrust_body = attitude.rates_sp, attitude.thrust_body
        rate = fleet.rate_step(rates_sp, thrust_body, state.rates, angular_accel, self.rate_dt)
        return fleet.rotor_step(rate.actuator, body_velocity_flu, self.rate_dt, body_rates_flu, armed)

    # ---- helpers ----------------------------------------------------------------------------------------------

    def _set_thrust(self, thrust):
        self._thrust_body.zero_()
        self._thrust_body[:, 2] = -thrust

    def _selection(self, mask):
        if mask is None:
            return slice(None)
        mask = self._tensor(mask, "mask", (self.n,), dtype=None)
        return mask.bool()

    def _column(self, value, name):
        return self._tensor(value, name, (self.n,))

    def _array(self, value, name, width):
        return self._tensor(value, name, (self.n, width))

    def _tensor(self, value, name, shape, dtype=torch.float32):
        if value is None:
            raise ValueError(f"{name} is required in mode {self.mode!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
        if value.device != self.fleet.device:
            raise ValueError(f"{name} is on {value.device}, the fleet is on {self.fleet.device}")
        if dtype is not None and value.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value

    def _check_state(self, state):
        for name, width in (("position", 3), ("velocity", 3), ("q", 4), ("rates", 3)):
            self._tensor(getattr(state, name), f"state.{name}", (self.n, width))
