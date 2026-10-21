"""Differentiable six-degree-of-freedom quadrotor dynamics.

The state combines world position/velocity, a unit body-to-world quaternion and
body angular velocity::

    x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r

The native action is the four non-negative rotor thrusts.  A ``RATE_THRUST``
mode is also available for controllers that close a fast inner loop around body
rates.  Rotor thrust is mixed into collective thrust and body torques before the
ODE is evaluated, keeping the translational and rotational physics identical in
both modes.
"""
import math
from typing import List, Optional, Tuple

import torch

from ..core.modes import ControlMode
from ..core.plant import Plant


class QuadrotorPlant(Plant):
    """Rigid-body quadrotor with thrust, drag and full attitude dynamics."""

    state_names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz',
                   'p', 'q', 'r')
    supported_modes = (ControlMode.MOTOR_THRUST, ControlMode.RATE_THRUST)
    default_mode = ControlMode.MOTOR_THRUST
    default_dt = 0.005
    default_integrator = 'rk4'
    tracked_states = ('x', 'y', 'z')
    tracked_weights = (1.0, 1.0, 2.0)

    def __init__(self, mode: Optional[ControlMode] = None, *, mass: float = 0.65,
                 arm_length: float = 0.17, inertia: Tuple[float, float, float] =
                 (0.0075, 0.0075, 0.0130), gravity: float = 9.81,
                 linear_drag: float = 0.08, angular_drag: float = 0.0015,
                 yaw_torque_coeff: float = 0.012, max_rotor_thrust: float = 5.0,
                 max_rate: float = 8.0, rate_gain: float = 0.045,
                 world_limit: float = 8.0, target: Tuple[float, float, float] =
                 (0.0, 0.0, 1.5)):
        super().__init__(mode)
        if mass <= 0 or arm_length <= 0 or min(inertia) <= 0 or max_rotor_thrust <= 0:
            raise ValueError('mass, arm_length, inertia and max_rotor_thrust must be positive')
        self.register_buffer('m', torch.tensor(float(mass)))
        self.register_buffer('I', torch.tensor(inertia, dtype=torch.get_default_dtype()))
        self.register_buffer('g', torch.tensor(float(gravity)))
        self.register_buffer('linear_drag', torch.tensor(float(linear_drag)))
        self.register_buffer('angular_drag', torch.tensor(float(angular_drag)))
        self.arm_length = float(arm_length)
        self.yaw_torque_coeff = float(yaw_torque_coeff)
        self.max_rotor_thrust = float(max_rotor_thrust)
        self.max_rate = float(max_rate)
        self.rate_gain = float(rate_gain)
        self.world_limit = float(world_limit)
        self.target = tuple(float(v) for v in target)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def action_names(self) -> List[str]:
        if self.mode is ControlMode.MOTOR_THRUST:
            return ['front_left_thrust', 'rear_left_thrust',
                    'rear_right_thrust', 'front_right_thrust']
        return ['collective_thrust', 'roll_rate_cmd', 'pitch_rate_cmd', 'yaw_rate_cmd']

    def action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode is ControlMode.MOTOR_THRUST:
            low, high = [0.0] * 4, [self.max_rotor_thrust] * 4
        else:
            low = [0.0, -self.max_rate, -self.max_rate, -self.max_rate]
            high = [4.0 * self.max_rotor_thrust, self.max_rate, self.max_rate, self.max_rate]
        return (torch.tensor(low, dtype=self._dtype(), device=self._device()),
                torch.tensor(high, dtype=self._dtype(), device=self._device()))

    def state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.world_limit
        high = torch.tensor([w, w, w, 15.0, 15.0, 15.0, 1.1, 1.1, 1.1, 1.1,
                             self.max_rate * 1.5, self.max_rate * 1.5,
                             self.max_rate * 1.5], dtype=self._dtype(), device=self._device())
        low = -high
        low[2] = 0.0
        return low, high

    def disturbance_channel(self) -> int:
        return 0  # collective thrust, after rotor mixing

    def _mix_rotors(self, thrust: torch.Tensor) -> torch.Tensor:
        """Map FL, RL, RR, FR rotor forces to collective thrust and body torque."""
        fl, rl, rr, fr = thrust.unbind(dim=-1)
        a, k = self.arm_length / math.sqrt(2.0), self.yaw_torque_coeff
        collective = fl + rl + rr + fr
        roll = a * (fl + rl - rr - fr)
        pitch = a * (-fl + rl + rr - fr)
        yaw = k * (fl - rl + rr - fr)
        return torch.stack([collective, roll, pitch, yaw], dim=-1)

    def map_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.mode is ControlMode.MOTOR_THRUST:
            return self._mix_rotors(action.clamp(0.0, self.max_rotor_thrust))
        rate_error = action[..., 1:] - state[..., 10:13]
        torque = (self.rate_gain * rate_error).clamp(-self._max_torque(), self._max_torque())
        collective = action[..., 0].clamp(0.0, 4.0 * self.max_rotor_thrust)
        return torch.cat([collective.unsqueeze(-1), torque], dim=-1)

    def _max_torque(self) -> torch.Tensor:
        a = self.arm_length / math.sqrt(2.0)
        return torch.tensor([2.0 * a * self.max_rotor_thrust,
                             2.0 * a * self.max_rotor_thrust,
                             2.0 * self.yaw_torque_coeff * self.max_rotor_thrust],
                            dtype=self._dtype(), device=self._device())

    @staticmethod
    def _rotation(q: torch.Tensor) -> torch.Tensor:
        """Body-to-world rotation matrix from a (normalised) scalar-first quaternion."""
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, x, y, z = q.unbind(dim=-1)
        return torch.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), dim=-1).reshape(q.shape[:-1] + (3, 3))

    def derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        velocity, quat, omega = state[..., 3:6], state[..., 6:10], state[..., 10:13]
        collective, torque = u[..., 0], u[..., 1:4]
        rotation = self._rotation(quat)
        thrust_world = rotation[..., :, 2] * collective.unsqueeze(-1)
        gravity = torch.zeros_like(velocity)
        gravity[..., 2] = -self.g
        acceleration = thrust_world / self.m + gravity - self.linear_drag * velocity / self.m

        qw, qx, qy, qz = quat.unbind(dim=-1)
        p, q, r = omega.unbind(dim=-1)
        quat_dot = 0.5 * torch.stack([-qx * p - qy * q - qz * r,
                                      qw * p + qy * r - qz * q,
                                      qw * q - qx * r + qz * p,
                                      qw * r + qx * q - qy * p], dim=-1)
        angular_momentum = self.I * omega
        coriolis = torch.cross(omega, angular_momentum, dim=-1)
        omega_dot = (torque - coriolis - self.angular_drag * omega) / self.I
        return torch.cat([velocity, acceleration, quat_dot, omega_dot], dim=-1)

    def initial_state(self, batch: int, generator: Optional[torch.Generator] = None,
                      randomize: bool = True) -> torch.Tensor:
        state = torch.zeros((batch, self.state_dim), dtype=self._dtype(), device=self._device())
        state[:, 6] = 1.0
        if randomize:
            noise = torch.rand((batch, 3), generator=generator, dtype=state.dtype,
                               device=state.device) - 0.5
            state[:, :3] = noise * torch.tensor([0.4, 0.4, 0.2], dtype=state.dtype,
                                                 device=state.device)
            state[:, 2].clamp_(min=0.2)
        return state

    def default_setpoint(self) -> torch.Tensor:
        return torch.tensor(self.target, dtype=self._dtype(), device=self._device())

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        velocity, omega = state[..., 3:6], state[..., 10:13]
        kinetic = 0.5 * self.m * (velocity ** 2).sum(dim=-1)
        rotational = 0.5 * (self.I * omega ** 2).sum(dim=-1)
        return kinetic + rotational + self.m * self.g * state[..., 2]
