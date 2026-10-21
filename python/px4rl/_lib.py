"""Loading of the CUDA shared library with ctypes, and the struct-size check. Does not import torch."""
import ctypes
from pathlib import Path

from . import _structs

LIBRARY_NAME = "libpx4rl_cuda.so"

_STRUCT_SIZE_FUNCTIONS = {
    "Params": "px4rl_params_size",
    "PlantParams": "px4rl_plant_params_size",
    "RigidBodyParams": "px4rl_rigid_body_params_size",
}

_library = None


class AbiMismatchError(ImportError):
    """The ctypes structs generated from the headers differ in size from the native library's structs."""


def check_struct_sizes(source, native_sizes):
    """Raises AbiMismatchError unless ctypes.sizeof of each generated struct equals the size `native_sizes`
    (a dict keyed like _STRUCT_SIZE_FUNCTIONS) reports for the library named `source`."""
    for name, native_size in native_sizes.items():
        python_size = ctypes.sizeof(getattr(_structs, name))
        if python_size != native_size:
            raise AbiMismatchError(
                f"px4rl: {name} is {python_size} bytes in the Python package but {native_size} bytes in {source}; "
                "the package and the native library were built from different headers. Reinstall px4rl."
            )


def cuda_library_path():
    return Path(__file__).with_name(LIBRARY_NAME)


def has_cuda_library():
    return cuda_library_path().is_file()


def load_cuda_library():
    """Loads (once) libpx4rl_cuda.so from the package directory, declares its functions and checks its struct
    sizes. Raises ImportError if the package was built without CUDA, OSError if the library cannot be loaded."""
    global _library
    if _library is not None:
        return _library

    path = cuda_library_path()
    if not path.is_file():
        raise ImportError(
            "px4rl was built without CUDA support (libpx4rl_cuda.so is missing). Reinstall it with "
            "`pip install . -C cmake.define.PX4RL_BUILD_CUDA=ON` on a machine with the CUDA toolkit."
        )

    lib = ctypes.CDLL(str(path))
    _declare(lib)
    check_struct_sizes(LIBRARY_NAME, {name: getattr(lib, fn)() for name, fn in _STRUCT_SIZE_FUNCTIONS.items()})
    _library = lib
    return lib


def _declare(lib):
    void_p = ctypes.c_void_p

    for function in _STRUCT_SIZE_FUNCTIONS.values():
        getattr(lib, function).restype = ctypes.c_size_t
        getattr(lib, function).argtypes = []

    lib.px4rl_last_error.restype = ctypes.c_char_p
    lib.px4rl_last_error.argtypes = []

    for function in ("px4rl_params_default", "px4rl_params_iris"):
        getattr(lib, function).restype = None
        getattr(lib, function).argtypes = [void_p]

    lib.px4rl_plant_params_iris.restype = _structs.PlantParams
    lib.px4rl_plant_params_iris.argtypes = []
    lib.px4rl_rigid_body_params_iris.restype = _structs.RigidBodyParams
    lib.px4rl_rigid_body_params_iris.argtypes = []

    lib.px4rl_cuda_fleet_create.restype = ctypes.c_int
    lib.px4rl_cuda_fleet_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(void_p)]
    lib.px4rl_cuda_fleet_destroy.restype = ctypes.c_int
    lib.px4rl_cuda_fleet_destroy.argtypes = [void_p]
    lib.px4rl_cuda_fleet_size.restype = ctypes.c_int
    lib.px4rl_cuda_fleet_size.argtypes = [void_p]

    for function in ("px4rl_cuda_fleet_reset", "px4rl_cuda_fleet_reset_plant"):
        getattr(lib, function).restype = ctypes.c_int
        getattr(lib, function).argtypes = [void_p, void_p, void_p, ctypes.c_int, void_p]

    lib.px4rl_cuda_position_step.restype = ctypes.c_int
    lib.px4rl_cuda_position_step.argtypes = [
        void_p, void_p, void_p, void_p, void_p, void_p, void_p,  # fleet, pos, vel, heading, az, traj_sp, in_air
        ctypes.c_float, ctypes.c_uint64,  # dt, time_us
        void_p, void_p, void_p, void_p, void_p, void_p,  # q_sp, thrust_sp_body, yawspeed, hover_thrust, outcome, stream
    ]
    lib.px4rl_cuda_attitude_step.restype = ctypes.c_int
    lib.px4rl_cuda_attitude_step.argtypes = [void_p] * 8
    lib.px4rl_cuda_rate_step.restype = ctypes.c_int
    lib.px4rl_cuda_rate_step.argtypes = [
        void_p, void_p, void_p, void_p, void_p,  # fleet, rates_sp, thrust_body, rates, angular_accel
        ctypes.c_float, void_p,  # dt, landed
        void_p, void_p, void_p, void_p,  # motors, actuator, torque_sp, stream
    ]
    lib.px4rl_cuda_rotor_step.restype = ctypes.c_int
    lib.px4rl_cuda_rotor_step.argtypes = [
        void_p, void_p, void_p, void_p, void_p,  # fleet, actuator, armed, body_velocity, body_rates
        ctypes.c_float,  # dt
        void_p, void_p, void_p, void_p,  # force, torque, rotor_force, stream
    ]
