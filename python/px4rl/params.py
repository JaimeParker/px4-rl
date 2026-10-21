"""Parameter factories and the hover thrust of a vehicle. The values come from the native library, so Python holds
no copy of any default."""
import ctypes
import math

from . import _lib
from ._structs import Params, PlantParams, RigidBodyParams

try:
    from . import _cpu
except ModuleNotFoundError as error:
    if error.name != f"{__package__}._cpu":
        raise
    _cpu = None


def _filled(struct_type, cpu_function, cuda_function):
    instance = struct_type()
    if _cpu is not None:
        getattr(_cpu, cpu_function)(instance)
    else:
        getattr(_lib.load_cuda_library(), cuda_function)(ctypes.byref(instance))
    return instance


def default_params():
    """`Params` with the defaults: PX4 v1.16 controller parameters and the default quad-X geometry."""
    return _filled(Params, "params_default", "px4rl_params_default")


def iris_params():
    """`Params` of PX4 SITL's Gazebo Classic Iris: the defaults plus the airframe's rotor geometry."""
    return _filled(Params, "params_iris", "px4rl_params_iris")


def iris_plant():
    """`PlantParams` of the Gazebo Classic Iris rotors."""
    if _cpu is not None:
        plant = PlantParams()
        _cpu.plant_params_iris(plant)
        return plant
    return _lib.load_cuda_library().px4rl_plant_params_iris()


def iris_rigid_body():
    """`RigidBodyParams` (mass and inertia) of the Gazebo Classic Iris."""
    if _cpu is not None:
        body = RigidBodyParams()
        _cpu.rigid_body_params_iris(body)
        return body
    return _lib.load_cuda_library().px4rl_rigid_body_params_iris()


def hover_thrust(params=None, plant=None, body=None, gravity=9.80665):
    """Hover thrust of an upright vehicle in steady flight, in PX4's MPC_THR_HOVER units (normalized collective
    thrust). Use it to seed `Params.mpc_thr_hover` so an episode can start in steady flight.

    `params`, `plant` and `body` are a `Params`, a `PlantParams` and a `RigidBodyParams`, defaulting to the
    Gazebo Classic Iris. `gravity` is in m/s^2. Raises ValueError if the vehicle cannot hover (its rotors cannot
    lift the weight) or an input is invalid.
    """
    params = iris_params() if params is None else params
    plant = iris_plant() if plant is None else plant
    body = iris_rigid_body() if body is None else body
    if _cpu is not None:
        return _cpu.hover_thrust(params, plant, body, gravity)
    lib = _lib.load_cuda_library()
    thrust = lib.px4rl_hover_thrust(ctypes.byref(params), ctypes.byref(plant), ctypes.byref(body), gravity)
    if math.isnan(thrust):
        raise ValueError(lib.px4rl_last_error().decode(errors="replace"))
    return thrust


def params_array(n, base=None):
    """A ctypes array of `n` copies of `base` (default: `default_params()`), for per-vehicle values.

    `base` can be a `Params` or a `PlantParams`; the array type follows it. Edit the elements, then pass the
    array to `reset` or `reset_plant`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base is None:
        base = default_params()
    if not isinstance(base, (Params, PlantParams)):
        raise TypeError(f"base must be a Params or a PlantParams, got {type(base).__name__}")
    array = (type(base) * n)()
    for i in range(n):
        array[i] = base
    return array


def struct_buffer(value, struct_type, name):
    """Returns (address, count) of a `struct_type` instance or a ctypes array of them; raises otherwise."""
    if isinstance(value, struct_type):
        return ctypes.addressof(value), 1
    if isinstance(value, ctypes.Array) and value._type_ is struct_type:
        return ctypes.addressof(value), len(value)
    raise TypeError(f"{name} must be a {struct_type.__name__} or a ctypes array of them, got {type(value).__name__}")
