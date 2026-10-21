"""Batched, differentiable ODE integrators.

Every integrator takes a derivative function ``f(state, u) -> dstate`` and advances
one step under a zero-order hold on ``u`` (the control is constant across the step,
which is what a sampled-data controller actually does).

All operations are plain torch ops on ``(batch, state_dim)`` tensors, so a step is
batched over initial conditions and differentiable with respect to both ``state``
and ``u``.
"""
from typing import Callable

import torch

DerivFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_step(f: DerivFn, state: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """Forward Euler. First-order accurate; kept for convergence-order testing."""
    return state + dt * f(state, u)


def rk4_step(f: DerivFn, state: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """Classical 4th-order Runge-Kutta with zero-order hold on ``u``."""
    k1 = f(state, u)
    k2 = f(state + 0.5 * dt * k1, u)
    k3 = f(state + 0.5 * dt * k2, u)
    k4 = f(state + dt * k3, u)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


#: Name -> integrator. ``order`` is the expected global order of accuracy, which
#: tests/envs/test_integrate.py checks by halving dt.
INTEGRATORS = {
    'euler': euler_step,
    'rk4': rk4_step,
}

ORDERS = {
    'euler': 1,
    'rk4': 4,
}


def get_integrator(name: str) -> DerivFn:
    if name not in INTEGRATORS:
        raise ValueError(f"unknown integrator '{name}'; choose from {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]


def rollout(f: DerivFn,
            state: torch.Tensor,
            controller: Callable[[int, torch.Tensor], torch.Tensor],
            steps: int,
            dt: float,
            integrator: str = 'rk4') -> torch.Tensor:
    """
    Roll the dynamics forward, returning the full state trajectory.

    Args:
        f: derivative function.
        state: initial state, ``(batch, state_dim)``.
        controller: called as ``controller(step_index, state) -> u``.
        steps: number of steps to take.
        dt: timestep.
        integrator: key into :data:`INTEGRATORS`.

    Returns:
        ``(batch, steps + 1, state_dim)`` including the initial state. The graph is
        retained, so a loss on the trajectory backpropagates into anything the
        controller depends on.
    """
    step_fn = get_integrator(integrator)
    traj = [state]
    for i in range(steps):
        state = step_fn(f, state, controller(i, state), dt)
        traj.append(state)
    return torch.stack(traj, dim=1)
