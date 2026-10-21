"""Builders for the trajectory-setpoint rows of `position_step`.

A row has 11 floats: position[3], velocity[3], acceleration[3], yaw, yawspeed. NaN marks a field as unused, as in
PX4. The builders take numpy arrays or torch tensors (any device) and return the same kind, float32. They never
import torch.
"""
import sys

import numpy as np

_NAN = float("nan")


def _is_torch(value):
    return type(value).__module__.split(".")[0] == "torch"


def _rows(vector, yaw, yawspeed, column, name):
    """Returns an (n, 11) array of NaN with `vector` in columns column..column+3, `yaw` and `yawspeed` set."""
    if _is_torch(vector):
        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n, 3), got {tuple(vector.shape)}")
        torch = sys.modules["torch"]
        rows = vector.new_full((vector.shape[0], 11), _NAN, dtype=torch.float32)
        rows[:, column:column + 3] = vector

    else:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 2 or vector.shape[1] != 3:
            raise ValueError(f"{name} must have shape (n, 3), got {vector.shape}")
        rows = np.full((vector.shape[0], 11), _NAN, dtype=np.float32)
        rows[:, column:column + 3] = vector

    rows[:, 9] = _as_column(yaw, rows)
    if yawspeed is not None:
        rows[:, 10] = _as_column(yawspeed, rows)
    return rows


def _as_column(value, rows):
    if _is_torch(rows):
        return value if _is_torch(value) else rows.new_tensor(np.asarray(value, dtype=np.float32))
    return np.asarray(value, dtype=np.float32)


def position_yaw(position, yaw, yawspeed=None):
    """Hold a position (n, 3) and a yaw angle (scalar or (n,))."""
    return _rows(position, yaw, yawspeed, 0, "position")


def velocity_yaw(velocity, yaw, yawspeed=None):
    """Track a velocity (n, 3) and hold a yaw angle (scalar or (n,))."""
    return _rows(velocity, yaw, yawspeed, 3, "velocity")


def acceleration_yaw(acceleration, yaw, yawspeed=None):
    """Command an acceleration (n, 3) and hold a yaw angle (scalar or (n,))."""
    return _rows(acceleration, yaw, yawspeed, 6, "acceleration")
