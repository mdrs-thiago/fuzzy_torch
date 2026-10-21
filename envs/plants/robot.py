"""Differential-drive mobile robot: planar, dynamic, three levels of actuation.

Configuration is planar 3-DOF (``x``, ``y``, ``psi``); the state adds body-frame
velocities, giving six state variables::

    x, y      position in the world frame        [m]
    psi       heading                            [rad]
    vx, vy    body-frame velocities              [m/s]
    omega     yaw rate                           [rad/s]

This is a *dynamic* model, not the kinematic shortcut where a commanded velocity is
integrated directly: mass and inertia are real, and motion follows from wheel forces.
``vy`` exists so the robot can **slip**.  Wheel forces are limited by the available
tyre-road friction, and lateral tyre force is limited by the remaining friction circle.
With ``lateral_drag`` large the behaviour approaches the ideal nonholonomic constraint;
on a hard turn it instead develops the small, physically necessary lateral slip.

Equations of motion, in the rotating body frame::

    x'    = vx cos(psi) - vy sin(psi)
    y'    = vx sin(psi) + vy cos(psi)
    psi'  = omega
    vx'   =  vy*omega + (Fx - c_x*vx) / m
    vy'   = -vx*omega + Fy / m
    omega'= (tau - c_psi*omega) / I

The Coriolis terms ``+vy*omega`` and ``-vx*omega`` do no net work, so with damping
and input zeroed the kinetic energy is conserved -- which is what the physics tests
check.
"""
import math
from typing import List, Optional, Tuple

import torch

from ..core.modes import ControlMode
from ..core.plant import Plant


class DiffDriveRobot(Plant):
    """Differential-drive robot driven at velocity, wheel-torque or motor-voltage level."""

    state_names = ('x', 'y', 'psi', 'vx', 'vy', 'omega')
    supported_modes = (ControlMode.VELOCITY, ControlMode.WHEEL_TORQUE,
                       ControlMode.MOTOR_VOLTAGE)
    default_mode = ControlMode.VELOCITY
    default_dt = 0.01
    default_integrator = 'rk4'
    tracked_states = ('x', 'y')
    tracked_weights = (1.0, 1.0)

    def __init__(self,
                 mode: Optional[ControlMode] = None,
                 mass: float = 12.0,
                 inertia: float = 0.25,
                 wheel_radius: float = 0.10,
                 track_width: float = 0.40,
                 rolling_drag: float = 4.0,
                 lateral_drag: float = 120.0,
                 tire_friction: float = 0.9,
                 yaw_drag: float = 0.6,
                 max_wheel_torque: float = 6.0,
                 max_voltage: float = 24.0,
                 max_speed: float = 2.0,
                 max_yaw_rate: float = 3.0,
                 motor_kt: float = 0.12,
                 motor_ke: float = 0.12,
                 motor_resistance: float = 4.8,
                 gear_ratio: float = 10.0,
                 velocity_gain: float = 6.0,
                 yaw_gain: float = 8.0,
                 world_size: float = 12.0,
                 goal: Tuple[float, float] = (4.0, 3.0)):
        super().__init__(mode)
        self.register_buffer('m', torch.tensor(float(mass)))
        self.register_buffer('I', torch.tensor(float(inertia)))
        self.register_buffer('c_x', torch.tensor(float(rolling_drag)))
        self.register_buffer('c_y', torch.tensor(float(lateral_drag)))
        self.register_buffer('c_psi', torch.tensor(float(yaw_drag)))

        self.wheel_radius = float(wheel_radius)
        self.track_width = float(track_width)
        if tire_friction <= 0.0:
            raise ValueError("tire_friction must be positive")
        self.tire_friction = float(tire_friction)
        self.max_wheel_torque = float(max_wheel_torque)
        self.max_voltage = float(max_voltage)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.kt = float(motor_kt)
        self.ke = float(motor_ke)
        self.resistance = float(motor_resistance)
        self.gear_ratio = float(gear_ratio)
        self.velocity_gain = float(velocity_gain)
        self.yaw_gain = float(yaw_gain)
        self.world_size = float(world_size)
        self.goal = (float(goal[0]), float(goal[1]))

    # ------------------------------------------------------------------ shape

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def action_names(self) -> List[str]:
        return {
            ControlMode.VELOCITY: ['speed_cmd', 'yaw_rate_cmd'],
            ControlMode.WHEEL_TORQUE: ['torque_left', 'torque_right'],
            ControlMode.MOTOR_VOLTAGE: ['voltage_left', 'voltage_right'],
        }[self.mode]

    def action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        highs = {
            ControlMode.VELOCITY: [self.max_speed, self.max_yaw_rate],
            ControlMode.WHEEL_TORQUE: [self.max_wheel_torque] * 2,
            ControlMode.MOTOR_VOLTAGE: [self.max_voltage] * 2,
        }[self.mode]
        high = torch.tensor(highs, dtype=self._dtype(), device=self._device())
        return -high, high

    def state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.world_size
        high = torch.tensor([w, w, math.pi, self.max_speed * 2, self.max_speed,
                             self.max_yaw_rate * 2],
                            dtype=self._dtype(), device=self._device())
        return -high, high

    def disturbance_channel(self) -> int:
        return 0

    # --------------------------------------------------------------- dynamics

    def _wheel_forces_to_body(self, f_left: torch.Tensor,
                              f_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left/right wheel forces -> body thrust and yaw torque."""
        return f_left + f_right, (f_right - f_left) * (self.track_width / 2.0)

    def _limit_wheel_forces(self, f_left: torch.Tensor,
                             f_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the longitudinal part of the tyre-road friction circle.

        Each driven wheel carries half the robot's static load.  This keeps all
        actuation modes honest: a large torque or voltage saturates at tyre grip
        rather than producing an unbounded acceleration.
        """
        limit = self.tire_friction * self.m * 9.81 / 2.0
        return f_left.clamp(-limit, limit), f_right.clamp(-limit, limit)

    def _body_force_to_wheels(self, force: torch.Tensor,
                              torque: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse of :meth:`_wheel_forces_to_body`."""
        difference = 2.0 * torque / self.track_width
        return (force - difference) / 2.0, (force + difference) / 2.0

    def map_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Convert the mode's action into body ``(Fx, tau)``."""
        vx, omega = state[..., 3], state[..., 5]

        if self.mode is ControlMode.VELOCITY:
            # Fast inner loop: a commanded velocity becomes the force needed to reach it.
            force = self.velocity_gain * self.m * (action[..., 0] - vx)
            torque = self.yaw_gain * self.I * (action[..., 1] - omega)
            max_force = 2.0 * self.max_wheel_torque / self.wheel_radius
            max_torque = max_force * self.track_width / 2.0
            f_l, f_r = self._body_force_to_wheels(force.clamp(-max_force, max_force),
                                                   torque.clamp(-max_torque, max_torque))
            f_l, f_r = self._limit_wheel_forces(f_l, f_r)
            return torch.stack(self._wheel_forces_to_body(f_l, f_r), dim=-1)

        if self.mode is ControlMode.WHEEL_TORQUE:
            f_l, f_r = self._limit_wheel_forces(action[..., 0] / self.wheel_radius,
                                                 action[..., 1] / self.wheel_radius)
            return torch.stack(self._wheel_forces_to_body(f_l, f_r), dim=-1)

        # MOTOR_VOLTAGE: DC motor with negligible inductance, so the current -- and
        # hence torque -- is algebraic in voltage and back-EMF. No extra state, but
        # back-EMF adds real speed-dependent damping the other modes do not have.
        v_left = vx - omega * self.track_width / 2.0
        v_right = vx + omega * self.track_width / 2.0
        w_left = v_left / self.wheel_radius * self.gear_ratio
        w_right = v_right / self.wheel_radius * self.gear_ratio

        tau_l = self.kt * (action[..., 0] - self.ke * w_left) / self.resistance
        tau_r = self.kt * (action[..., 1] - self.ke * w_right) / self.resistance
        tau_l = (tau_l * self.gear_ratio).clamp(-self.max_wheel_torque, self.max_wheel_torque)
        tau_r = (tau_r * self.gear_ratio).clamp(-self.max_wheel_torque, self.max_wheel_torque)
        f_l, f_r = self._limit_wheel_forces(tau_l / self.wheel_radius,
                                             tau_r / self.wheel_radius)
        return torch.stack(self._wheel_forces_to_body(f_l, f_r), dim=-1)

    def derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        psi, vx, vy, omega = state[..., 2], state[..., 3], state[..., 4], state[..., 5]
        force, torque = u[..., 0], u[..., 1]

        cos_p, sin_p = torch.cos(psi), torch.sin(psi)
        x_dot = vx * cos_p - vy * sin_p
        y_dot = vx * sin_p + vy * cos_p

        vx_dot = vy * omega + (force - self.c_x * vx) / self.m
        # The tyre force is viscous close to zero slip but cannot exceed what is
        # left after longitudinal drive/braking force has used part of the grip.
        # This is the standard friction-circle approximation for a two-wheel robot.
        f_left, f_right = self._body_force_to_wheels(force, torque)
        wheel_grip = self.tire_friction * self.m * 9.81 / 2.0
        lateral_limit = (torch.sqrt(torch.clamp(wheel_grip ** 2 - f_left ** 2, min=0.0)) +
                         torch.sqrt(torch.clamp(wheel_grip ** 2 - f_right ** 2, min=0.0)))
        lateral_force = (-self.c_y * vy).clamp(-lateral_limit, lateral_limit)
        vy_dot = -vx * omega + lateral_force / self.m
        omega_dot = (torque - self.c_psi * omega) / self.I

        return torch.stack([x_dot, y_dot, omega, vx_dot, vy_dot, omega_dot], dim=-1)

    # ------------------------------------------------------------------ reset

    def initial_state(self, batch: int, generator: Optional[torch.Generator] = None,
                      randomize: bool = True) -> torch.Tensor:
        device, dtype = self._device(), self._dtype()
        state = torch.zeros((batch, self.state_dim), device=device, dtype=dtype)
        if randomize:
            def unif(scale):
                r = torch.rand((batch,), generator=generator, device=device, dtype=dtype)
                return (2.0 * r - 1.0) * scale
            state[:, 0] = unif(1.0)
            state[:, 1] = unif(1.0)
            state[:, 2] = unif(math.pi)
        return state

    # ---------------------------------------------------------------- helpers

    def default_setpoint(self) -> torch.Tensor:
        return torch.tensor(self.goal, dtype=self._dtype(), device=self._device())

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Kinetic energy. Conserved with damping and input zeroed -- the Coriolis
        terms are workless, so a sign error in them shows up here."""
        vx, vy, omega = state[..., 3], state[..., 4], state[..., 5]
        return 0.5 * self.m * (vx ** 2 + vy ** 2) + 0.5 * self.I * omega ** 2

    def turn_radius(self, vx: float, omega: float) -> float:
        """Radius of the circle traced at constant ``vx`` and ``omega``."""
        return float('inf') if abs(omega) < 1e-12 else vx / omega
