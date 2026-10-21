"""PX4-faithful multirotor control and rotor dynamics for reinforcement learning.

`import px4rl` gives the CPU classes (numpy), the parameter structs and factories, and the setpoint builders. It
does not import torch. The batched GPU fleet is `px4rl.cuda.CudaFleet`.
"""
from . import _lib, _structs, setpoints
from ._results import (
    AllocatorResult,
    AttitudeResult,
    PositionControlResult,
    PositionResult,
    RateResult,
    RotorResult,
)
from ._structs import (
    CONSTANTS,
    MAX_ROTORS,
    OUTCOME_FAILSAFE,
    OUTCOME_FALLBACK,
    OUTCOME_OK,
    Params,
    PlantParams,
    RigidBodyParams,
    RotorParams,
)
from ._version import __version__
from .params import default_params, hover_thrust, iris_params, iris_plant, iris_rigid_body, params_array

try:
    from . import _cpu
except ModuleNotFoundError as _error:
    if _error.name != f"{__name__}._cpu":
        raise
    _cpu = None

_CPU_CLASSES = (
    "ActuatorOutput",
    "AttitudeController",
    "ControlAllocator",
    "CpuFleet",
    "HoverThrustEstimator",
    "PositionController",
    "RateController",
    "Vehicle",
)

if _cpu is not None:
    ActuatorOutput = _cpu.ActuatorOutput
    AttitudeController = _cpu.AttitudeController
    ControlAllocator = _cpu.ControlAllocator
    CpuFleet = _cpu.CpuFleet
    HoverThrustEstimator = _cpu.HoverThrustEstimator
    PositionController = _cpu.PositionController
    RateController = _cpu.RateController
    Vehicle = _cpu.Vehicle
    _lib.check_struct_sizes("px4rl._cpu", {
        "Params": _cpu.params_size(),
        "PlantParams": _cpu.plant_params_size(),
        "RigidBodyParams": _cpu.rigid_body_params_size(),
    })

# A size mismatch with the CUDA library is an error now; a library that cannot be loaded is reported when
# px4rl.cuda is imported, so that the CPU classes keep working.
if _lib.has_cuda_library():
    try:
        _lib.load_cuda_library()
    except _lib.AbiMismatchError:
        raise
    except OSError:
        pass

__all__ = [
    "setpoints",
    "Params",
    "PlantParams",
    "RigidBodyParams",
    "RotorParams",
    "MAX_ROTORS",
    "OUTCOME_OK",
    "OUTCOME_FALLBACK",
    "OUTCOME_FAILSAFE",
    "default_params",
    "hover_thrust",
    "iris_params",
    "iris_plant",
    "iris_rigid_body",
    "params_array",
    "PositionResult",
    "AttitudeResult",
    "RateResult",
    "RotorResult",
    "PositionControlResult",
    "AllocatorResult",
    "__version__",
    *(_CPU_CLASSES if _cpu is not None else ()),
]
