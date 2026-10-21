import numpy as np
import pytest

import px4rl
from px4rl import setpoints

BUILDERS = [
    (setpoints.position_yaw, slice(0, 3)),
    (setpoints.velocity_yaw, slice(3, 6)),
    (setpoints.acceleration_yaw, slice(6, 9)),
]


def assert_pattern(rows, used, yaw, yawspeed=None):
    rows = np.asarray(rows)
    assert rows.shape == (2, 11) and rows.dtype == np.float32
    unused = np.ones(11, bool)
    unused[used] = False
    unused[9] = unused[10] = False
    assert np.isnan(rows[:, unused]).all()
    assert not np.isnan(rows[:, used]).any()
    np.testing.assert_array_equal(rows[:, 9], yaw)
    if yawspeed is None:
        assert np.isnan(rows[:, 10]).all()
    else:
        np.testing.assert_array_equal(rows[:, 10], yawspeed)


@pytest.mark.parametrize("builder, used", BUILDERS)
def test_numpy_nan_patterns(builder, used):
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    rows = builder(values, 0.5)
    assert isinstance(rows, np.ndarray)
    assert_pattern(rows, used, np.float32(0.5))
    np.testing.assert_array_equal(rows[:, used], values.astype(np.float32))
    assert_pattern(builder(values, np.float32([0.1, 0.2]), yawspeed=0.25), used, np.float32([0.1, 0.2]), 0.25)


@pytest.mark.parametrize("builder, used", BUILDERS)
def test_torch_input_gives_a_torch_float32_tensor_of_the_same_device(builder, used):
    torch = pytest.importorskip("torch")
    values = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    rows = builder(values, 0.5)
    assert isinstance(rows, torch.Tensor) and rows.dtype == torch.float32 and rows.device == values.device
    assert_pattern(rows.numpy(), used, np.float32(0.5))
    assert_pattern(builder(values, torch.tensor([0.1, 0.2]), yawspeed=0.25).numpy(), used,
                   np.float32([0.1, 0.2]), 0.25)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        setpoints.position_yaw(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        setpoints.velocity_yaw(np.zeros((2, 4)), 0.0)


def test_setpoints_are_exposed_by_the_package():
    assert px4rl.setpoints is setpoints
