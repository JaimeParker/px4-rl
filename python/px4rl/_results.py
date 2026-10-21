"""Result containers shared by the CPU and CUDA fleets and the single-vehicle classes. Pure Python."""
from typing import Any, NamedTuple


class PositionResult(NamedTuple):
    """Output of `position_step`.

    q_sp (n, 4) attitude setpoint (w, x, y, z), body FRD to NED. thrust_sp_body (n, 3) normalized thrust, body
    FRD. yawspeed (n,) yaw rate feed-forward, rad/s. hover_thrust (n,) current hover thrust estimate. outcome
    (n,) uint8, one of OUTCOME_OK, OUTCOME_FALLBACK, OUTCOME_FAILSAFE.
    """

    q_sp: Any
    thrust_sp_body: Any
    yawspeed: Any
    hover_thrust: Any
    outcome: Any


class AttitudeResult(NamedTuple):
    """Output of `attitude_step`: rate setpoint (n, 3) in rad/s and normalized thrust (n, 3), body FRD."""

    rates_sp: Any
    thrust_body: Any


class RateResult(NamedTuple):
    """Output of `rate_step`: motors (n, 4) normalized to [0, 1], actuator (n, 4) PWM or DShot value,
    torque_sp (n, 3) normalized torque."""

    motors: Any
    actuator: Any
    torque_sp: Any


class RotorResult(NamedTuple):
    """Output of `rotor_step`, all in body FLU: force (n, 3) in N, torque (n, 3) in N m about the body origin,
    rotor_force (n, 4, 3) in N per rotor."""

    force: Any
    torque: Any
    rotor_force: Any


class PositionControlResult(NamedTuple):
    """Output of `PositionController.update`: the fields of `PositionResult` without the hover thrust."""

    q_sp: Any
    thrust_sp_body: Any
    yawspeed: Any
    outcome: Any


class AllocatorResult(NamedTuple):
    """Output of `ControlAllocator.update`."""

    motors: Any
    saturation_positive: Any
    saturation_negative: Any
    unallocated_torque: Any
    unallocated_thrust: Any
