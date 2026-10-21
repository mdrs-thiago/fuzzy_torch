"""Overhead / gantry crane: trolley with a suspended payload.

States (2-DOF, the default):
    x      trolley position            [m]
    x_dot  trolley velocity            [m/s]
    theta  cable angle from vertical   [rad]   (0 = payload hanging straight down)
    omega  swing rate                  [rad/s]

With ``hoist=True`` two more states are appended and a second actuator appears:
    L      cable length                [m]
    L_dot  hoist rate                  [m/s]

Equations of motion (Euler-Lagrange, rigid massless cable, point-mass payload)::

    (M+m) x'' + m[L'' sin0 + 2 L' 0' cos0 + L 0'' cos0 - L 0'^2 sin0] = F
    L 0'' + 2 L' 0' + x'' cos0 + g sin0 = 0
    L'' + x'' sin0 - L 0'^2 - g cos0 = F_L / m

Substituting the second into the other two leaves a 2x2 linear system in
``(x'', L'')`` whose determinant is exactly ``M``, so it inverts in closed form
with no matrix solve.

NOTE ON THE EXISTING CODE: ``verify_physics.py`` and ``ui/app.jsx`` label their
model a crane but implement **cart-pole** dynamics -- an *inverted* pendulum, with
gravity destabilising (``theta_acc ~ +g sin0``) and the ``4/3`` uniform-rod inertia
factor. A crane payload *hangs*: gravity restores (``theta_acc ~ -g sin0``) and a
point mass has no rod factor. This module implements the crane. The rule base
ported from the ``crane`` preset was tuned against the inverted model, so its
behaviour here is expected to differ.
"""
import math
from typing import List, Optional, Tuple

import torch

from ..core.modes import ControlMode
from ..core.plant import Plant


class CranePlant(Plant):
    """Gantry crane, optionally with a hoist degree of freedom."""

    supported_modes = (ControlMode.FORCE, ControlMode.VELOCITY)
    default_mode = ControlMode.FORCE
    # Position is the objective; sway is regulated to zero alongside it.
    tracked_states = ('x', 'theta')
    tracked_weights = (1.0, 0.5)
    # The swing period for L=2m is ~2.8s, so 0.01s resolves it with ~280 steps.
    default_dt = 0.01
    default_integrator = 'rk4'

    def __init__(self,
                 mode: Optional[ControlMode] = None,
                 trolley_mass: float = 2.0,
                 payload_mass: float = 0.5,
                 cable_length: float = 2.0,
                 gravity: float = 9.81,
                 trolley_damping: float = 0.0,
                 swing_damping: float = 0.0,
                 hoist: bool = False,
                 max_force: float = 30.0,
                 max_hoist_force: float = 30.0,
                 max_velocity: float = 5.0,
                 velocity_gain: float = 40.0,
                 x_limit: float = 10.0,
                 target_x: float = 5.0,
                 length_limits: Tuple[float, float] = (0.5, 5.0)):
        self.hoist = bool(hoist)
        self.state_names = (('x', 'x_dot', 'theta', 'omega', 'L', 'L_dot')
                            if self.hoist else ('x', 'x_dot', 'theta', 'omega'))
        super().__init__(mode)

        self.register_buffer('M', torch.tensor(float(trolley_mass)))
        self.register_buffer('m', torch.tensor(float(payload_mass)))
        self.register_buffer('L0', torch.tensor(float(cable_length)))
        self.register_buffer('g', torch.tensor(float(gravity)))
        self.register_buffer('b_x', torch.tensor(float(trolley_damping)))
        self.register_buffer('b_theta', torch.tensor(float(swing_damping)))

        self.max_force = float(max_force)
        self.max_hoist_force = float(max_hoist_force)
        self.max_velocity = float(max_velocity)
        self.velocity_gain = float(velocity_gain)
        self.x_limit = float(x_limit)
        self.target_x = float(target_x)
        self.length_limits = (float(length_limits[0]), float(length_limits[1]))

    # ------------------------------------------------------------------ shape

    @property
    def action_dim(self) -> int:
        return 2 if self.hoist else 1

    @property
    def action_names(self) -> List[str]:
        first = 'trolley_velocity' if self.mode is ControlMode.VELOCITY else 'trolley_force'
        return [first, 'hoist_force'] if self.hoist else [first]

    def action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        first = self.max_velocity if self.mode is ControlMode.VELOCITY else self.max_force
        highs = [first] + ([self.max_hoist_force] if self.hoist else [])
        high = torch.tensor(highs, dtype=self._dtype(), device=self._device())
        return -high, high

    def state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        highs = [self.x_limit, 10.0, math.pi / 2, 10.0]
        if self.hoist:
            highs += [self.length_limits[1], 5.0]
        high = torch.tensor(highs, dtype=self._dtype(), device=self._device())
        low = -high.clone()
        if self.hoist:
            low[4] = 0.0  # cable length is strictly positive
        return low, high

    def disturbance_channel(self) -> int:
        return 0  # a horizontal force on the trolley

    # --------------------------------------------------------------- dynamics

    def map_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """VELOCITY mode closes a fast proportional loop onto the commanded speed."""
        if self.mode is not ControlMode.VELOCITY:
            return action
        force = self.velocity_gain * (action[..., 0] - state[..., 1])
        force = force.clamp(-self.max_force, self.max_force)
        rest = [action[..., i] for i in range(1, action.shape[-1])]
        return torch.stack([force] + rest, dim=-1)

    def derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_dot = state[..., 1]
        theta = state[..., 2]
        omega = state[..., 3]
        L = state[..., 4] if self.hoist else self.L0.expand_as(theta)
        L_dot = state[..., 5] if self.hoist else torch.zeros_like(theta)

        F = u[..., 0] - self.b_x * x_dot
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)

        # rhs of the reduced (x'', L'') system
        rhs_x = F + self.m * self.g * sin_t * cos_t + self.m * L * omega ** 2 * sin_t

        if self.hoist:
            F_L = u[..., 1]
            rhs_L = L * omega ** 2 + self.g * cos_t + F_L / self.m
            # [M + m sin^2,  m sin] [x'']   [rhs_x]      det = M
            # [   sin     ,    1  ] [L'']   [rhs_L]
            x_acc = (rhs_x - self.m * sin_t * rhs_L) / self.M
            L_acc = ((self.M + self.m * sin_t ** 2) * rhs_L - sin_t * rhs_x) / self.M
        else:
            x_acc = rhs_x / (self.M + self.m * sin_t ** 2)
            L_acc = None

        # Gravity restores here: a hanging payload, not an inverted pendulum.
        theta_acc = -(x_acc * cos_t + self.g * sin_t + 2.0 * L_dot * omega) / L
        theta_acc = theta_acc - self.b_theta * omega

        cols = [x_dot, x_acc, omega, theta_acc]
        if self.hoist:
            cols += [L_dot, L_acc]
        return torch.stack(cols, dim=-1)

    # ------------------------------------------------------------------ reset

    def initial_state(self, batch: int, generator: Optional[torch.Generator] = None,
                      randomize: bool = True) -> torch.Tensor:
        device, dtype = self._device(), self._dtype()
        state = torch.zeros((batch, self.state_dim), device=device, dtype=dtype)
        if self.hoist:
            state[:, 4] = self.L0
        if randomize:
            def unif(scale):
                r = torch.rand((batch,), generator=generator, device=device, dtype=dtype)
                return (2.0 * r - 1.0) * scale
            state[:, 0] = unif(1.0)    # trolley within +/-1 m of origin
            state[:, 2] = unif(0.15)   # initial sway up to ~8.6 deg
        return state

    # ------------------------------------------------------------------ energy

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Total mechanical energy. Conserved when damping and inputs are zero."""
        x_dot = state[..., 1]
        theta = state[..., 2]
        omega = state[..., 3]
        L = state[..., 4] if self.hoist else self.L0.expand_as(theta)
        L_dot = state[..., 5] if self.hoist else torch.zeros_like(theta)

        payload_sq = (x_dot ** 2 + L_dot ** 2 + (L * omega) ** 2
                      + 2.0 * x_dot * L_dot * torch.sin(theta)
                      + 2.0 * x_dot * L * omega * torch.cos(theta))
        kinetic = 0.5 * self.M * x_dot ** 2 + 0.5 * self.m * payload_sq
        # Zero at the payload's lowest point for the nominal cable length.
        potential = self.m * self.g * (self.L0 - L * torch.cos(theta))
        return kinetic + potential

    def default_setpoint(self) -> torch.Tensor:
        """Move the trolley to ``target_x`` with no residual sway."""
        return torch.tensor([self.target_x, 0.0], dtype=self._dtype(), device=self._device())

    def swing_period(self) -> float:
        """Small-angle swing period with a free trolley: 2*pi*sqrt(M L / (g (M+m)))."""
        M, m, L, g = self.M.item(), self.m.item(), self.L0.item(), self.g.item()
        return 2.0 * math.pi * math.sqrt(M * L / (g * (M + m)))
