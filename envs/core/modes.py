"""Control modes: at what level of the actuator stack does the controller act?

Mirrors the ``ActionType`` idea from gym-pybullet-drones. The same plant can be
driven by velocity commands (easy, kinematic-ish), by wheel torques or by motor
voltages (progressively closer to the real actuator, and progressively harder).

Each plant declares which modes it supports and maps an action in that mode onto
the physical input its dynamics expect.
"""
from enum import Enum


class ControlMode(str, Enum):
    """Level in the actuator stack at which an action is interpreted."""

    #: Command a velocity (or velocity pair) directly. The plant closes a fast
    #: internal loop onto it. Easiest to control; least physically honest.
    VELOCITY = 'velocity'

    #: Command generalized force / torque on the body (e.g. crane trolley force).
    FORCE = 'force'

    #: Command per-wheel torque; body motion follows from mass and inertia.
    WHEEL_TORQUE = 'wheel_torque'

    #: Command motor voltage through a DC motor model, adding an electrical time
    #: constant that forces a smaller dt.
    MOTOR_VOLTAGE = 'motor_voltage'

    #: Command per-rotor thrust (quadrotor).
    MOTOR_THRUST = 'motor_thrust'

    #: Command body angular rates plus collective thrust (quadrotor).
    RATE_THRUST = 'rate_thrust'

    def __str__(self) -> str:
        return self.value
