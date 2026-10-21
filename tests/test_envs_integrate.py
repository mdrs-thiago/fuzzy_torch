"""Integrators: convergence order and differentiability."""
import math

import pytest
import torch

from fuzzy_torch.envs import ORDERS, euler_step, rk4_step, rollout
from fuzzy_torch.envs.core.integrate import get_integrator
from fuzzy_torch.envs.core.modes import ControlMode
from fuzzy_torch.envs.plants.robot import DiffDriveRobot

STEPPERS = {'euler': euler_step, 'rk4': rk4_step}


def _harmonic(state, u):
    """x' = v, v' = -x. Solution: x(t) = cos(t) for x(0)=1, v(0)=0."""
    return torch.stack([state[..., 1], -state[..., 0]], dim=-1)


def _forced(state, u):
    """x' = v, v' = -x + u. Same oscillator, but the control actually enters."""
    return torch.stack([state[..., 1], -state[..., 0] + u[..., 0]], dim=-1)


# Step counts per integrator, chosen so the discretisation error stays well above
# the float64 round-off floor. RK4 at dt=1/1600 is already near 1e-13, where
# accumulated round-off would corrupt the measured order.
ORDER_GRIDS = {'euler': (200, 400, 800, 1600), 'rk4': (10, 20, 40, 80)}


@pytest.mark.parametrize("name", ['euler', 'rk4'])
def test_convergence_order(name):
    """Halving dt must shrink the error at the integrator's advertised rate.

    This validates the *integrator*, not just the equations -- a plant can have
    correct dynamics and still be integrated wrongly.
    """
    step = STEPPERS[name]
    horizon = 1.0
    errors = []
    for n in ORDER_GRIDS[name]:
        dt = horizon / n
        s = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        u = torch.zeros((1, 1), dtype=torch.float64)
        for _ in range(n):
            s = step(_harmonic, s, u, dt)
        errors.append(abs(float(s[0, 0]) - math.cos(horizon)))

    for coarse, fine in zip(errors[:-1], errors[1:]):
        observed = math.log2(coarse / fine)
        assert observed == pytest.approx(ORDERS[name], abs=0.35), \
            f"{name}: observed order {observed:.2f}, expected {ORDERS[name]}"


def test_rk4_is_far_more_accurate_than_euler():
    s0 = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    u = torch.zeros((1, 1), dtype=torch.float64)
    out = {}
    for name, step in STEPPERS.items():
        s = s0.clone()
        for _ in range(1000):
            s = step(_harmonic, s, u, 1e-3)
        out[name] = abs(float(s[0, 0]) - math.cos(1.0))
    assert out['rk4'] < out['euler'] / 1e6


def test_unknown_integrator_is_rejected():
    with pytest.raises(ValueError, match="unknown integrator"):
        get_integrator('midpoint')


def test_rollout_shape_and_initial_state():
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    traj = rollout(_harmonic, s, lambda i, st: torch.zeros((2, 1)), steps=10, dt=0.01)
    assert traj.shape == (2, 11, 2)
    assert torch.allclose(traj[:, 0], s)


def test_rollout_is_differentiable():
    """Gradients must reach a controller parameter through the whole trajectory."""
    gain = torch.tensor(0.5, requires_grad=True)
    s = torch.tensor([[1.0, 0.0]])
    traj = rollout(_forced, s,
                   lambda i, st: (-gain * st[..., :1]),
                   steps=50, dt=0.01)
    traj[:, -1, 0].sum().backward()
    assert gain.grad is not None and torch.isfinite(gain.grad).all()
    assert gain.grad.abs() > 0


def test_batching_matches_scalar_rollouts():
    """A batch of N states must evolve exactly as N separate rollouts."""
    torch.manual_seed(0)
    batch = torch.randn(8, 2, dtype=torch.float64)
    u = torch.zeros((8, 1), dtype=torch.float64)
    together = batch.clone()
    for _ in range(100):
        together = rk4_step(_harmonic, together, u, 1e-2)
    for i in range(8):
        alone = batch[i:i + 1].clone()
        for _ in range(100):
            alone = rk4_step(_harmonic, alone, u[i:i + 1], 1e-2)
        assert torch.allclose(together[i], alone[0], atol=1e-12)


def test_robot_wheel_force_is_limited_by_tyre_grip():
    robot = DiffDriveRobot(mode=ControlMode.WHEEL_TORQUE, mass=10.0,
                           tire_friction=0.5, max_wheel_torque=100.0)
    state = torch.zeros((1, robot.state_dim))
    body_force = robot.map_action(torch.tensor([[100.0, 100.0]]), state)

    # Both wheels together can transmit at most mu * m * g to the ground.
    # float32 tolerance: the clamp bound itself is rounded to float32.
    assert float(body_force[0, 0]) <= 0.5 * 10.0 * 9.81 * (1 + 1e-6)
    assert float(body_force[0, 0]) > 0.0
