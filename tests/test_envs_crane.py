"""Crane plant: physics validated against analytical results, before any controller.

These are the checks that make an env trustworthy on its own -- a plant with a sign
error integrates perfectly happily and produces a plausible-looking trajectory.
"""

import pytest
import torch

from fuzzy_torch.envs import ControlMode, CranePlant, rk4_step


def _free_run(plant, state, steps, dt=1e-3, u=None):
    u = torch.zeros((state.shape[0], plant.action_dim), dtype=state.dtype) if u is None else u
    for _ in range(steps):
        state = rk4_step(plant.derivatives, state, u, dt)
    return state


@pytest.fixture
def crane():
    return CranePlant().double()


def test_hanging_equilibrium_is_stable(crane):
    """theta = 0 with the payload down is an equilibrium, and it is *stable*.

    A cart-pole (inverted pendulum) would diverge here. verify_physics.py and
    ui/app.jsx implement the inverted model despite calling it a crane.
    """
    s = crane.initial_state(1, randomize=False)
    assert float(crane.derivatives(s, torch.zeros(1, 1, dtype=torch.float64)).abs().max()) == 0.0

    s[0, 2] = 0.05
    final = _free_run(crane, s, 5000)
    assert abs(float(final[0, 2])) <= 0.05 + 1e-6, "swing grew: dynamics are inverted"


def test_gravity_restores_rather_than_destabilises(crane):
    """Displaced to +theta with everything at rest, angular acceleration is negative."""
    s = crane.initial_state(1, randomize=False)
    s[0, 2] = 0.2
    theta_acc = float(crane.derivatives(s, torch.zeros(1, 1, dtype=torch.float64))[0, 3])
    assert theta_acc < 0


def test_energy_is_conserved_without_damping_or_input(crane):
    """Several swing amplitudes at once -- batching is free, so the amplitudes ride
    in one rollout rather than one test invocation each."""
    amplitudes = torch.tensor([0.05, 0.1, 0.2, 0.4], dtype=torch.float64)
    s = crane.initial_state(len(amplitudes), randomize=False)
    s[:, 2] = amplitudes
    e0 = crane.energy(s)
    e1 = crane.energy(_free_run(crane, s, 5000))   # 5 s, several swing periods
    drift = ((e1 - e0).abs() / e0.abs())
    assert float(drift.max()) < 1e-10, dict(zip(amplitudes.tolist(), drift.tolist()))


def test_small_angle_period_matches_theory(crane):
    """Free-trolley swing period must equal 2*pi*sqrt(M L / (g (M+m)))."""
    s = crane.initial_state(1, randomize=False)
    s[0, 2] = 1e-3
    dt, t, prev, crossings = 1e-3, 0.0, float(s[0, 2]), []
    u = torch.zeros(1, 1, dtype=torch.float64)
    for _ in range(20000):
        s = rk4_step(crane.derivatives, s, u, dt)
        t += dt
        cur = float(s[0, 2])
        if prev > 0 >= cur:
            # Interpolate the crossing instead of taking the sample time, so dt
            # can stay coarse without capping the achievable accuracy.
            crossings.append(t - dt * cur / (cur - prev))
        prev = cur
        if len(crossings) >= 2:
            break
    measured = crossings[1] - crossings[0]
    assert measured == pytest.approx(crane.swing_period(), rel=1e-4)


def test_damping_removes_energy(crane):
    damped = CranePlant(swing_damping=0.5, trolley_damping=0.5).double()
    s = damped.initial_state(1, randomize=False)
    s[0, 2] = 0.3
    e0, e1 = damped.energy(s), damped.energy(_free_run(damped, s, 5000))
    assert float(e1) < float(e0)


def test_force_accelerates_the_trolley(crane):
    s = crane.initial_state(1, randomize=False)
    u = torch.tensor([[10.0]], dtype=torch.float64)
    # At theta = 0: x_acc = F / M.
    assert float(crane.derivatives(s, u)[0, 1]) == pytest.approx(10.0 / crane.M.item())


def test_batched_states_evolve_independently(crane):
    torch.manual_seed(0)
    s = crane.initial_state(6, generator=torch.Generator().manual_seed(0))
    together = _free_run(crane, s.clone(), 500)
    for i in range(6):
        alone = _free_run(crane, s[i:i + 1].clone(), 500)
        assert torch.allclose(together[i], alone[0], atol=1e-12)


def test_dynamics_are_differentiable(crane):
    s = crane.initial_state(1, randomize=False).requires_grad_(True)
    u = torch.tensor([[3.0]], dtype=torch.float64, requires_grad=True)
    _free_run(crane, s, 50, u=u).sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(u.grad).all()
    assert float(u.grad.abs().sum()) > 0


# ------------------------------------------------------------------- hoist

def test_hoist_free_fall_matches_kinematics():
    """With no hoist force the payload free-falls: L(t) = L0 + g t^2 / 2."""
    plant = CranePlant(hoist=True).double()
    s = plant.initial_state(1, randomize=False)
    final = _free_run(plant, s, 2000)          # 2 s at dt = 1e-3
    expected = plant.L0.item() + 0.5 * plant.g.item() * 2.0 ** 2
    assert float(final[0, 4]) == pytest.approx(expected, rel=1e-6)


def test_hoist_energy_conserved_with_no_input():
    plant = CranePlant(hoist=True).double()
    s = plant.initial_state(2, randomize=False)
    s[:, 2] = torch.tensor([0.1, 0.3], dtype=torch.float64)
    e0 = plant.energy(s)
    e1 = plant.energy(_free_run(plant, s, 2000))
    assert float(((e1 - e0).abs() / e0.abs()).max()) < 1e-9


def test_static_hold_is_exactly_static():
    """theta = 0 with F_L = -m g holds every state fixed, to the last bit."""
    plant = CranePlant(hoist=True).double()
    s = plant.initial_state(1, randomize=False)
    u = torch.tensor([[0.0, -(plant.m * plant.g).item()]], dtype=torch.float64)
    assert torch.equal(_free_run(plant, s, 2000, u=u), s)


def test_hoist_adds_states_and_an_actuator():
    plain, hoisting = CranePlant(), CranePlant(hoist=True)
    assert plain.state_dim == 4 and plain.action_dim == 1
    assert hoisting.state_dim == 6 and hoisting.action_dim == 2
    assert hoisting.state_names[-2:] == ('L', 'L_dot')


# ------------------------------------------------------------- control modes

def test_velocity_mode_drives_towards_the_commanded_speed():
    plant = CranePlant(mode=ControlMode.VELOCITY).double()
    s = plant.initial_state(1, randomize=False)
    u = plant.map_action(torch.tensor([[2.0]], dtype=torch.float64), s)
    assert float(u[0, 0]) > 0          # below target speed: accelerate
    s[0, 1] = 2.0
    assert float(plant.map_action(torch.tensor([[2.0]], dtype=torch.float64), s)[0, 0]) == 0.0


def test_velocity_mode_respects_the_force_limit():
    plant = CranePlant(mode=ControlMode.VELOCITY).double()
    s = plant.initial_state(1, randomize=False)
    u = plant.map_action(torch.tensor([[5.0]], dtype=torch.float64), s)
    assert abs(float(u[0, 0])) <= plant.max_force + 1e-9


def test_unsupported_mode_is_rejected():
    with pytest.raises(ValueError, match="does not support control mode"):
        CranePlant(mode=ControlMode.MOTOR_THRUST)


def test_state_index_error_names_the_options(crane):
    with pytest.raises(KeyError, match="no state 'nope'"):
        crane.state_index('nope')
