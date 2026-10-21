"""Import behaviour, the struct-size check, and the parameter factories."""
import ctypes
import subprocess
import sys
import textwrap

import pytest

import px4rl
from px4rl import _lib, _structs


def run_python(code):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True)


def test_import_and_cpu_path_work_without_torch_and_without_the_cuda_library():
    # Blocks torch with a meta-path finder, so an import anywhere in px4rl would fail loudly, and makes loading
    # any shared library with ctypes fail, as if libpx4rl_cuda.so were missing or broken.
    result = run_python(
        """
        import ctypes
        import sys

        def blocked(*args, **kwargs):
            raise OSError("loading shared libraries is blocked in this test")

        ctypes.CDLL = blocked

        class BlockTorch:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch is blocked in this test")

        sys.meta_path.insert(0, BlockTorch())
        import px4rl
        import numpy as np

        assert "torch" not in sys.modules and "px4rl.cuda" not in sys.modules
        params = px4rl.iris_params()
        fleet = px4rl.CpuFleet(4, num_threads=1)
        fleet.reset(params)
        n = 4
        z = lambda *shape: np.zeros(shape, np.float32)
        sp = px4rl.setpoints.position_yaw(z(n, 3), 0.0)
        fleet.position_step(z(n, 3), z(n, 3), z(n), z(n), sp, 0.02, 20000)
        fleet.attitude_step(np.tile(np.float32([1, 0, 0, 0]), (n, 1)))
        out = fleet.rate_step(z(n, 3), z(n, 3), z(n, 3), z(n, 3), 0.004)
        assert np.isfinite(out.motors).all()
        assert "torch" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_ctypes_struct_sizes_match_the_native_library():
    assert ctypes.sizeof(px4rl.Params) == px4rl._cpu.params_size()
    assert ctypes.sizeof(px4rl.PlantParams) == px4rl._cpu.plant_params_size()
    assert ctypes.sizeof(px4rl.RigidBodyParams) == px4rl._cpu.rigid_body_params_size()


def test_size_mismatch_is_reported_as_import_error():
    good = {"Params": ctypes.sizeof(px4rl.Params)}
    _lib.check_struct_sizes("test", good)
    with pytest.raises(_lib.AbiMismatchError, match="Params"):
        _lib.check_struct_sizes("test", {"Params": good["Params"] + 4})
    assert issubclass(_lib.AbiMismatchError, ImportError)


def test_generated_structs_cover_the_header_and_constants():
    assert set(_structs.STRUCTS) == {"Params", "RotorParams", "PlantParams", "RigidBodyParams"}
    assert px4rl.MAX_ROTORS == 4
    assert (px4rl.OUTCOME_OK, px4rl.OUTCOME_FALLBACK, px4rl.OUTCOME_FAILSAFE) == (0, 1, 2)
    assert len(px4rl.PlantParams().rotor) == px4rl.MAX_ROTORS
    assert len(px4rl.Params().rotor_position) == px4rl.MAX_ROTORS


def flatten(value, prefix=""):
    """Yields (path, python float or int) for every scalar of a ctypes struct, array or scalar."""
    if isinstance(value, ctypes.Structure):
        for name, _ in value._fields_:
            yield from flatten(getattr(value, name), f"{prefix}.{name}")
    elif isinstance(value, ctypes.Array):
        for i, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{i}]")
    else:
        yield prefix, value


def test_factories_have_the_documented_values():
    default, iris = px4rl.default_params(), px4rl.iris_params()
    assert default.airmode == 0 and default.mpc_thr_hover == pytest.approx(0.5)
    assert default.position_dt == pytest.approx(0.02)
    assert default.num_rotors == 4 and iris.num_rotors == 4
    assert default.rotor_ct[0] == pytest.approx(1.0) and iris.rotor_ct[0] == pytest.approx(6.5)
    assert iris.rotor_km[0] == pytest.approx(0.05) and iris.rotor_km[2] == pytest.approx(-0.05)
    differing = {path for (path, a), (_, b) in zip(flatten(default), flatten(iris)) if a != b}
    assert differing and all(path.split("[")[0].split(".")[-1] in ("rotor_ct", "rotor_position") for path in differing)

    plant = px4rl.iris_plant()
    assert plant.num_rotors == 4
    assert plant.rotor[0].motor_constant == pytest.approx(5.84e-6)
    assert plant.rotor[0].max_rot_velocity == pytest.approx(1100.0)
    body = px4rl.iris_rigid_body()
    assert body.mass == pytest.approx(1.5) and body.inertia[0] > 0


def test_pybind_and_ctypes_factories_agree_field_by_field():
    if not _lib.has_cuda_library():
        pytest.skip("px4rl was built without CUDA, so there is no ctypes path")
    lib = _lib.load_cuda_library()
    for cpu_factory, symbol in ((px4rl.default_params, "px4rl_params_default"), (px4rl.iris_params, "px4rl_params_iris")):
        from_ctypes = px4rl.Params()
        getattr(lib, symbol)(ctypes.byref(from_ctypes))
        from_pybind = cpu_factory()
        pairs = list(zip(flatten(from_pybind), flatten(from_ctypes)))
        assert len(pairs) > 100
        for (path, a), (_, b) in pairs:
            assert a == b or (a != a and b != b), f"{symbol}{path}: {a} != {b}"
        assert bytes(from_pybind) == bytes(from_ctypes)

    plant, body = lib.px4rl_plant_params_iris(), lib.px4rl_rigid_body_params_iris()
    assert dict(flatten(px4rl.iris_plant())) == dict(flatten(plant))
    assert dict(flatten(px4rl.iris_rigid_body())) == dict(flatten(body))


def test_cuda_library_check_runs_at_import_when_the_library_is_present():
    if not _lib.has_cuda_library():
        with pytest.raises(ImportError, match="without CUDA"):
            _lib.load_cuda_library()
    else:
        lib = _lib.load_cuda_library()
        assert lib.px4rl_params_size() == ctypes.sizeof(px4rl.Params)
        assert lib.px4rl_plant_params_size() == ctypes.sizeof(px4rl.PlantParams)
        assert lib.px4rl_rigid_body_params_size() == ctypes.sizeof(px4rl.RigidBodyParams)


def iris_hover_actuator_value(body):
    """Closed form for the Iris plant, whose four rotors share one motor constant: 4 k_f (1000 u + 100)^2 = m g."""
    rotor = px4rl.iris_plant().rotor[0]
    speed = (body.mass * 9.80665 / (4 * rotor.motor_constant)) ** 0.5
    return (speed - rotor.zero_position_armed) / rotor.input_scaling


def test_hover_thrust_defaults_to_the_iris_and_follows_the_thrust_model_factor():
    expected = iris_hover_actuator_value(px4rl.iris_rigid_body())
    assert px4rl.hover_thrust() == pytest.approx(0.6935, abs=5e-4)
    assert px4rl.hover_thrust() == pytest.approx(expected, abs=1e-5)  # THR_MDL_FAC is 0 for the Iris
    if _lib.has_cuda_library():
        c_value = _lib.load_cuda_library().px4rl_hover_thrust(
            ctypes.byref(px4rl.iris_params()), ctypes.byref(px4rl.iris_plant()),
            ctypes.byref(px4rl.iris_rigid_body()), 9.80665)
        assert px4rl.hover_thrust() == c_value

    params = px4rl.iris_params()
    params.thr_mdl_fac = 0.3
    assert px4rl.hover_thrust(params) == pytest.approx(0.3 * expected**2 + 0.7 * expected, abs=1e-5)

    body = px4rl.iris_rigid_body()
    body.mass = 1.8
    assert px4rl.hover_thrust(body=body) > px4rl.hover_thrust()


def test_hover_thrust_raises_value_error_when_the_vehicle_cannot_hover_or_an_input_is_invalid():
    body = px4rl.iris_rigid_body()
    body.mass = 3.0  # the rotors lift at most 28.3 N, the weight is 29.4 N
    with pytest.raises(ValueError, match="weight"):
        px4rl.hover_thrust(body=body)
    with pytest.raises(ValueError, match="gravity"):
        px4rl.hover_thrust(gravity=0.0)


def test_hover_thrust_through_ctypes_matches_pybind(monkeypatch):
    if not _lib.has_cuda_library():
        pytest.skip("px4rl was built without CUDA, so there is no ctypes path")
    from px4rl import params as params_module

    expected = px4rl.hover_thrust()
    monkeypatch.setattr(params_module, "_cpu", None)
    assert params_module.hover_thrust() == pytest.approx(expected, abs=1e-6)
    body = px4rl.iris_rigid_body()
    body.mass = 3.0
    with pytest.raises(ValueError, match="weight"):
        params_module.hover_thrust(body=body)


def test_params_array_makes_independent_copies():
    base = px4rl.iris_params()
    base.att_roll_p = 5.0
    array = px4rl.params_array(3, base)
    array[1].att_roll_p = 7.0
    assert [p.att_roll_p for p in array] == [5.0, 7.0, 5.0] and base.att_roll_p == 5.0
    assert isinstance(px4rl.params_array(2)[0], px4rl.Params)
    assert isinstance(px4rl.params_array(2, px4rl.iris_plant())[0], px4rl.PlantParams)
    with pytest.raises(ValueError):
        px4rl.params_array(0)
    with pytest.raises(TypeError):
        px4rl.params_array(2, object())
