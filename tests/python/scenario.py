"""Random but valid inputs for the px4rl chain, shared by the CPU and CUDA tests."""
from types import SimpleNamespace

import numpy as np

import px4rl

DT_POSITION = 0.02
DT_RATE = 0.004


class Scenario:
    """Per-step state of n vehicles. `step()` draws a new random state and setpoints; the setpoint modes
    (position, velocity, acceleration + yaw) alternate over the vehicles."""

    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.time_us = 0
        self.step()

    def step(self):
        rng, n = self.rng, self.n

        def uniform(low, high, *shape):
            return rng.uniform(low, high, size=(n, *shape)).astype(np.float32)

        self.time_us += 20000
        self.position = uniform(-2, 2, 3)
        self.velocity = uniform(-1, 1, 3)
        self.heading = uniform(-1, 1)
        self.acceleration_z = uniform(-1, 1)
        yaw = uniform(-1, 1)
        targets = uniform(-2, 2, 3)
        mode = np.arange(n) % 3
        self.trajectory_sp = np.where(
            (mode == 0)[:, None], px4rl.setpoints.position_yaw(targets, yaw),
            np.where((mode == 1)[:, None], px4rl.setpoints.velocity_yaw(targets, yaw),
                     px4rl.setpoints.acceleration_yaw(targets, yaw)),
        ).astype(np.float32)
        # np.where over NaN-bearing rows keeps the NaN pattern of the chosen mode.
        q = np.concatenate([np.ones((n, 1), np.float32), uniform(-0.15, 0.15, 3)], axis=1)
        self.q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
        self.rates = uniform(-0.5, 0.5, 3)
        self.angular_accel = uniform(-1, 1, 3)
        self.body_velocity = uniform(-3, 3, 3)
        self.body_rates = uniform(-2, 2, 3)
        return self


def varied_params(n, seed=1):
    """Per-vehicle Params: the Iris values with a few gains changed by up to 20 %."""
    rng = np.random.default_rng(seed)
    params = px4rl.params_array(n, px4rl.iris_params())
    for p in params:
        p.att_roll_p *= rng.uniform(0.8, 1.2)
        p.att_yaw_p *= rng.uniform(0.8, 1.2)
        p.mpc_xy_p *= rng.uniform(0.8, 1.2)
        p.mpc_z_vel_p_acc *= rng.uniform(0.8, 1.2)
        for axis in range(3):
            p.rate_p[axis] *= rng.uniform(0.8, 1.2)
            p.rate_i[axis] *= rng.uniform(0.8, 1.2)
        p.thr_mdl_fac = rng.uniform(0.0, 0.3)
    return params


def snapshots(n, steps, seed=0):
    """`steps` independent draws of the state of n vehicles, as namespaces of numpy arrays."""
    scenario = Scenario(n, seed)
    result = []
    for _ in range(steps):
        scenario.step()
        result.append(SimpleNamespace(**{k: np.array(v) if isinstance(v, np.ndarray) else v
                                         for k, v in vars(scenario).items() if k not in ("rng", "n")}))
    return result


def row(snapshot, i):
    """The state of vehicle i alone, as arrays with a leading axis of length 1."""
    return SimpleNamespace(**{k: np.ascontiguousarray(v[i:i + 1]) if isinstance(v, np.ndarray) else v
                              for k, v in vars(snapshot).items()})


def run_chain(fleet, history):
    """Runs the position, attitude, rate and rotor steps on `fleet` for each snapshot of `history`, and returns
    one tuple of copied arrays (every field of the four results) per snapshot."""
    records = []
    for s in history:
        position = fleet.position_step(s.position, s.velocity, s.heading, s.acceleration_z, s.trajectory_sp,
                                       DT_POSITION, s.time_us)
        attitude = fleet.attitude_step(s.q)
        rate = fleet.rate_step(attitude.rates_sp, attitude.thrust_body, s.rates, s.angular_accel, DT_RATE)
        rotor = fleet.rotor_step(rate.actuator, s.body_velocity, DT_RATE, s.body_rates)
        records.append(tuple(np.array(a) for result in (position, attitude, rate, rotor) for a in result))
    return records


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
