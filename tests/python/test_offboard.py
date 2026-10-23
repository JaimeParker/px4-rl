"""OffboardController: every mode against the call sequence it replaces, held setpoints, masked reset, the
stabilized yaw setpoint, and a closed-loop Iris flight per mode on a small torch rigid-body model."""
import math

import numpy as np
import pytest

import px4rl
from scenario import DT_POSITION, DT_RATE, snapshots

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cuda

N = 64
SUBSTEPS = round(DT_POSITION / DT_RATE)
GRAVITY = 9.80665


@pytest.fixture(scope="module")
def offboard():
    pytest.importorskip("px4rl.cuda")
    from px4rl import offboard

    return offboard


def gpu(array):
    return torch.from_numpy(np.ascontiguousarray(array)).cuda()


def make_fleet(n=N):
    from px4rl.cuda import CudaFleet

    fleet = CudaFleet(n, "cuda")
    fleet.reset(px4rl.iris_params())
    fleet.reset_plant(px4rl.iris_plant())
    return fleet


def states(offboard, steps, seed=0):
    """Random but valid vehicle states, one per position period, with per-substep body rates."""
    result = []
    for s in snapshots(N, steps, seed):
        rates = [gpu(s.rates + 0.05 * k) for k in range(SUBSTEPS)]
        result.append((s, [offboard.VehicleState(gpu(s.position), gpu(s.velocity), gpu(s.q), r) for r in rates]))
    return result


def clone_all(result):
    return tuple(t.clone() for t in result)


def hand_written_chain(fleet, history, mode, setpoint):
    """The call sequence the controller replaces, written out: finite differences, clock, and per mode the
    position, attitude, rate and rotor steps. `setpoint(k)` gives the mode's inputs for position period k."""
    previous_velocity_down = torch.zeros(N, device="cuda")
    previous_rates = torch.zeros(N, 3, device="cuda")
    yaw_sp = torch.zeros(N, device="cuda")
    time_us = 0
    records = []
    for k, (s, substates) in enumerate(history):
        state = substates[0]
        acceleration_down = (state.velocity[:, 2] - previous_velocity_down) / DT_POSITION
        previous_velocity_down = state.velocity[:, 2].clone()
        time_us += round(DT_POSITION * 1e6)
        sp = setpoint(k)
        if mode in ("position", "velocity", "acceleration"):
            w, x, y, z = state.q.unbind(-1)
            heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            fleet.position_step(state.position, state.velocity, heading, acceleration_down, sp["trajectory_sp"],
                                DT_POSITION, time_us)
        thrust_body = torch.zeros(N, 3, device="cuda")
        if "thrust" in sp:
            thrust_body[:, 2] = -sp["thrust"]
        if mode == "stabilized":
            yaw_sp = torch.remainder(yaw_sp + sp["yaw_rate"] * DT_POSITION + math.pi, 2 * math.pi) - math.pi
            from px4rl.offboard import quat_from_euler

            q_sp, yawspeed = quat_from_euler(sp["roll"], sp["pitch"], yaw_sp), sp["yaw_rate"]
        elif mode == "attitude":
            q_sp, yawspeed = sp["q_sp"], sp["yawspeed_ff"]
        for sub in substates:
            angular_accel = (sub.rates - previous_rates) / DT_RATE
            previous_rates = sub.rates.clone()
            if mode in ("position", "velocity", "acceleration"):
                attitude = fleet.attitude_step(sub.q)
                rates_sp, thrust = attitude.rates_sp, attitude.thrust_body
            elif mode == "body_rate":
                rates_sp, thrust = sp["rates_sp"], thrust_body
            else:
                attitude = fleet.attitude_step(sub.q, q_sp=q_sp, thrust_sp_body=thrust_body, yawspeed_ff=yawspeed)
                rates_sp, thrust = attitude.rates_sp, attitude.thrust_body
            rate = fleet.rate_step(rates_sp, thrust, sub.rates, angular_accel, DT_RATE)
            body_velocity = gpu(s.body_velocity)
            records.append(clone_all(fleet.rotor_step(rate.actuator, body_velocity, DT_RATE, gpu(s.body_rates))))
    return records


def controller_chain(offboard, fleet, history, mode, setpoint):
    controller = offboard.OffboardController(fleet, DT_POSITION, DT_RATE)
    records = []
    for k, (s, substates) in enumerate(history):
        controller.command(mode, substates[0], **setpoint(k))
        for sub in substates:
            records.append(clone_all(controller.physics(sub, gpu(s.body_velocity), gpu(s.body_rates))))
    return records


def setpoints_for(mode, history):
    gen = torch.Generator(device="cuda").manual_seed(3)

    def rand(*shape, scale=1.0):
        return (torch.rand(*shape, device="cuda", generator=gen) * 2 - 1) * scale

    table = []
    for s, _ in history:
        if mode in ("position", "velocity", "acceleration"):
            table.append({"trajectory_sp": gpu(s.trajectory_sp)})
        elif mode == "attitude":
            q = torch.nn.functional.normalize(torch.cat((torch.ones(N, 1, device="cuda"), rand(N, 3, scale=0.2)), 1))
            table.append({"q_sp": q.contiguous(), "thrust": 0.5 + rand(N, scale=0.2), "yawspeed_ff": rand(N)})
        elif mode == "stabilized":
            table.append({"roll": rand(N, scale=0.3), "pitch": rand(N, scale=0.3), "yaw_rate": rand(N),
                          "thrust": 0.5 + rand(N, scale=0.2)})
        else:
            table.append({"rates_sp": rand(N, 3, scale=2.0), "thrust": 0.5 + rand(N, scale=0.2)})
    return table.__getitem__


@pytest.mark.parametrize("mode", ["velocity", "attitude", "stabilized", "body_rate"])
def test_each_mode_matches_the_call_sequence_it_replaces(offboard, mode):
    history = states(offboard, 8)
    setpoint = setpoints_for(mode, history)
    expected = hand_written_chain(make_fleet(), history, mode, setpoint)
    actual = controller_chain(offboard, make_fleet(), history, mode, setpoint)
    assert len(actual) == len(expected) == 8 * SUBSTEPS
    for a, e in zip(actual, expected):
        for ta, te in zip(a, e):
            assert torch.equal(ta, te)


def test_the_setpoint_is_held_between_commands(offboard):
    fleet = make_fleet()
    controller = offboard.OffboardController(fleet, DT_POSITION, DT_RATE)
    (s, substates), = states(offboard, 1)
    rates_sp = torch.full((N, 3), 0.3, device="cuda")
    controller.command("body_rate", substates[0], rates_sp=rates_sp, thrust=torch.full((N,), 0.6, device="cuda"))
    rates_sp.fill_(-5.0)  # the caller's tensor changes; the held copy must not
    assert torch.equal(controller._rates_sp, torch.full((N, 3), 0.3, device="cuda"))
    assert torch.equal(controller._thrust_body[:, 2], torch.full((N,), -0.6, device="cuda"))


def test_masked_reset_clears_only_the_selected_vehicles(offboard):
    controller = offboard.OffboardController(make_fleet(), DT_POSITION, DT_RATE)
    (s, substates), = states(offboard, 1)
    controller.command("body_rate", substates[0], rates_sp=torch.zeros(N, 3, device="cuda"),
                       thrust=torch.zeros(N, device="cuda"))
    controller.physics(substates[0], gpu(s.body_velocity))
    mask = (torch.arange(N, device="cuda") % 2 == 0).to(torch.uint8)
    heading = torch.full((N,), 1.25, device="cuda")
    before_velocity = controller._previous_velocity_down.clone()
    before_rates = controller._previous_rates.clone()
    controller.reset(mask, heading)
    even, odd = mask.bool(), ~mask.bool()
    assert (controller._previous_velocity_down[even] == 0).all() and (controller._previous_rates[even] == 0).all()
    assert torch.equal(controller._previous_velocity_down[odd], before_velocity[odd])
    assert torch.equal(controller._previous_rates[odd], before_rates[odd])
    assert (controller.yaw_setpoint[even] == 1.25).all() and (controller.yaw_setpoint[odd] == 0).all()


def test_stabilized_yaw_setpoint_integrates_the_yaw_rate_and_wraps(offboard):
    controller = offboard.OffboardController(make_fleet(), DT_POSITION, DT_RATE)
    (s, substates), = states(offboard, 1)
    controller.reset(heading=torch.full((N,), 3.0, device="cuda"))
    zero = torch.zeros(N, device="cuda")
    for _ in range(10):
        controller.command("stabilized", substates[0], roll=zero, pitch=zero,
                           yaw_rate=torch.full((N,), 1.0, device="cuda"), thrust=zero)
    expected = math.remainder(3.0 + 10 * DT_POSITION, 2 * math.pi)
    assert torch.allclose(controller.yaw_setpoint, torch.full((N,), expected, device="cuda"), atol=1e-5)


def test_missing_or_malformed_inputs_raise(offboard):
    controller = offboard.OffboardController(make_fleet(), DT_POSITION, DT_RATE)
    (s, substates), = states(offboard, 1)
    with pytest.raises(RuntimeError, match="command must be called"):
        controller.physics(substates[0], gpu(s.body_velocity))
    with pytest.raises(ValueError, match="mode must be one of"):
        controller.command("hover", substates[0])
    with pytest.raises(ValueError, match="needs trajectory_sp"):
        controller.command("velocity", substates[0])
    with pytest.raises(ValueError, match="thrust is required"):
        controller.command("body_rate", substates[0], rates_sp=torch.zeros(N, 3, device="cuda"))
    with pytest.raises(ValueError, match="rates_sp must have shape"):
        controller.command("body_rate", substates[0], rates_sp=torch.zeros(N, 4, device="cuda"),
                           thrust=torch.zeros(N, device="cuda"))
    with pytest.raises(TypeError, match="CudaFleet"):
        offboard.OffboardController(object(), DT_POSITION, DT_RATE)


# ---- closed loop ----------------------------------------------------------------------------------------------

def quat_rotate(q, v):
    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(u, v, dim=1)
    return v + w * t + torch.cross(u, t, dim=1)


def quat_conjugate(q):
    return q * q.new_tensor((1.0, -1.0, -1.0, -1.0))


def flip_yz(v):
    """Body FLU <-> FRD (its own inverse)."""
    return v * v.new_tensor((1.0, -1.0, -1.0))


class RigidBody:
    """Iris rigid body in world NED / body FRD, semi-implicit Euler at the rate period."""

    def __init__(self, n, altitude=3.0, heading=0.0):
        body = px4rl.iris_rigid_body()
        self.mass = float(body.mass)
        self.inertia = torch.tensor(list(body.inertia), device="cuda").reshape(3, 3)
        self.inertia_inv = torch.linalg.inv(self.inertia)
        self.position = torch.zeros(n, 3, device="cuda")
        self.position[:, 2] = -altitude
        self.velocity = torch.zeros(n, 3, device="cuda")
        self.q = torch.zeros(n, 4, device="cuda")
        self.q[:, 0], self.q[:, 3] = math.cos(heading / 2), math.sin(heading / 2)
        self.rates = torch.zeros(n, 3, device="cuda")

    def state(self, offboard):
        return offboard.VehicleState(self.position.clone(), self.velocity.clone(), self.q.clone(), self.rates.clone())

    def body_velocity_flu(self):
        return flip_yz(quat_rotate(quat_conjugate(self.q), self.velocity)).contiguous()

    def body_rates_flu(self):
        return flip_yz(self.rates).contiguous()

    def integrate(self, wrench, dt):
        force, torque = flip_yz(wrench.force), flip_yz(wrench.torque)
        acceleration = quat_rotate(self.q, force) / self.mass
        acceleration[:, 2] += GRAVITY
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        gyro = torch.cross(self.rates, self.rates @ self.inertia.T, dim=1)
        self.rates += (torque - gyro) @ self.inertia_inv.T * dt
        w = torch.cat((torch.zeros(len(self.q), 1, device="cuda"), self.rates), 1)
        aw, ax, ay, az = self.q.unbind(1)
        bw, bx, by, bz = w.unbind(1)
        q_dot = 0.5 * torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                                   aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), 1)
        self.q = torch.nn.functional.normalize(self.q + q_dot * dt, dim=1)


def fly(offboard, mode, inputs, seconds, n=16, heading=0.0):
    """Flies the Iris in `mode` with `inputs(controller, body)` per position period; returns the body."""
    from px4rl.cuda import CudaFleet

    params = px4rl.iris_params()
    params.mpc_thr_hover = px4rl.hover_thrust(params, px4rl.iris_plant(), px4rl.iris_rigid_body(), GRAVITY)
    fleet = CudaFleet(n, "cuda")
    fleet.reset(params)
    fleet.reset_plant(px4rl.iris_plant())
    controller = offboard.OffboardController(fleet, DT_POSITION, DT_RATE)
    body = RigidBody(n, heading=heading)
    controller.reset(heading=torch.full((n,), heading, device="cuda"))
    body.down_velocity = []
    for _ in range(round(seconds / DT_POSITION)):
        controller.command(mode, body.state(offboard), **inputs(n))
        for _ in range(SUBSTEPS):
            body.integrate(controller.physics(body.state(offboard), body.body_velocity_flu(), body.body_rates_flu()),
                           DT_RATE)
        body.down_velocity.append(float(body.velocity[:, 2].mean()))
    return body


def hover_inputs(mode):
    thrust = px4rl.hover_thrust(px4rl.iris_params(), px4rl.iris_plant(), px4rl.iris_rigid_body(), GRAVITY)

    def inputs(n):
        zeros = torch.zeros(n, device="cuda")
        full_thrust = torch.full((n,), thrust, device="cuda")
        if mode == "velocity":
            return {"trajectory_sp": px4rl.setpoints.velocity_yaw(torch.zeros(n, 3, device="cuda"), zeros)}
        if mode == "acceleration":
            return {"trajectory_sp": px4rl.setpoints.acceleration_yaw(torch.zeros(n, 3, device="cuda"), zeros)}
        if mode == "attitude":
            q = torch.zeros(n, 4, device="cuda")
            q[:, 0] = 1.0
            return {"q_sp": q, "thrust": full_thrust}
        if mode == "stabilized":
            return {"roll": zeros, "pitch": zeros, "yaw_rate": zeros, "thrust": full_thrust}
        return {"rates_sp": torch.zeros(n, 3, device="cuda"), "thrust": full_thrust}

    return inputs


def down_velocity_at(body, seconds):
    return body.down_velocity[round(seconds / DT_POSITION) - 1]


@pytest.mark.parametrize("mode", ["velocity", "acceleration", "attitude", "stabilized", "body_rate"])
def test_a_neutral_command_keeps_the_iris_level_and_near_hover(offboard, mode):
    """The rotors start at rest, so every mode sinks while they spin up (about 0.15 m/s in the first 20 ms). The
    velocity mode then holds the altitude. The other modes have no altitude feedback, and a constant hover thrust
    is not an altitude hold: the Gazebo motor model scales the thrust by 1 - |axial airspeed| / max relative
    airspeed, so a sink lowers the thrust further. For them the check is a level attitude, no horizontal drift,
    and a small vertical acceleration after the spin-up."""
    body = fly(offboard, mode, hover_inputs(mode), seconds=3.0)
    horizontal = body.position[:, :2].norm(dim=1)
    tilt = 2 * torch.acos(torch.sqrt(body.q[:, 0] ** 2 + body.q[:, 3] ** 2).clamp(max=1.0))
    down_acceleration = (down_velocity_at(body, 1.0) - down_velocity_at(body, 0.25)) / 0.75
    print(f"{mode}: altitude change {-(body.position[:, 2] + 3.0).mean():+.3f} m, max horizontal drift "
          f"{horizontal.max():.4f} m, max tilt {tilt.max():.4f} rad, mean down acceleration 0.25-1 s "
          f"{down_acceleration:+.3f} m/s^2; mean down velocity at "
          + ", ".join(f"{t:.2f} s {down_velocity_at(body, t):+.3f}" for t in (0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)),
          flush=True)
    assert torch.isfinite(body.position).all()
    assert horizontal.max() < 0.05
    assert tilt.max() < 0.05
    assert abs(down_acceleration) < 0.2  # thrust within 2 % of the weight
    if mode == "velocity":
        assert (body.position[:, 2] + 3.0).abs().max() < 0.1


def test_velocity_mode_flies_north(offboard):
    def inputs(n):
        v = torch.zeros(n, 3, device="cuda")
        v[:, 0] = 1.0
        return {"trajectory_sp": px4rl.setpoints.velocity_yaw(v, torch.zeros(n, device="cuda"))}

    body = fly(offboard, "velocity", inputs, seconds=4.0)
    assert (body.velocity[:, 0] - 1.0).abs().max() < 0.1
    assert body.position[:, 0].min() > 2.5


def test_stabilized_yaw_rate_turns_the_iris(offboard):
    thrust = px4rl.hover_thrust(px4rl.iris_params(), px4rl.iris_plant(), px4rl.iris_rigid_body(), GRAVITY)

    def inputs(n):
        zeros = torch.zeros(n, device="cuda")
        return {"roll": zeros, "pitch": zeros, "yaw_rate": torch.full((n,), 0.5, device="cuda"),
                "thrust": torch.full((n,), thrust, device="cuda")}

    body = fly(offboard, "stabilized", inputs, seconds=2.0)
    w, x, y, z = body.q.unbind(1)
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    print(f"stabilized yaw: heading after 2 s at 0.5 rad/s: {heading.mean():.3f} rad", flush=True)
    assert (heading - 1.0).abs().max() < 0.15


def test_body_rate_mode_tracks_a_roll_rate(offboard):
    thrust = px4rl.hover_thrust(px4rl.iris_params(), px4rl.iris_plant(), px4rl.iris_rigid_body(), GRAVITY)

    def inputs(n):
        rates = torch.zeros(n, 3, device="cuda")
        rates[:, 0] = 0.5
        return {"rates_sp": rates, "thrust": torch.full((n,), thrust, device="cuda")}

    body = fly(offboard, "body_rate", inputs, seconds=0.4)
    print(f"body_rate: roll rate after 0.4 s {body.rates[:, 0].mean():.3f} rad/s", flush=True)
    assert (body.rates[:, 0] - 0.5).abs().max() < 0.1
