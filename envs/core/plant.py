"""The Plant abstraction: batched, differentiable continuous-time dynamics.

A :class:`Plant` is deliberately *not* a Gym environment. It is just the ODE plus
the metadata needed to drive and interpret it. That separation is what lets one
implementation serve three consumers:

* :class:`~fuzzy_torch.envs.core.gym_env.FuzzyControlEnv` wraps it as a standard
  ``gymnasium.Env`` for RL and for baseline comparison;
* batched GPU rollouts sweep thousands of initial conditions at once;
* a differentiable rollout backpropagates a trajectory loss into controller
  parameters.

Plants subclass ``nn.Module`` purely so their constants are buffers and ``.to(device)``
moves everything in one call.
"""
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .modes import ControlMode


class Plant(nn.Module):
    """Continuous-time dynamics, batched over the leading dimension."""

    #: Names of the state variables, in order. Length defines ``state_dim``.
    state_names: Sequence[str] = ()

    #: Control modes this plant accepts.
    supported_modes: Sequence[ControlMode] = ()

    #: Mode used when the caller does not specify one.
    default_mode: Optional[ControlMode] = None

    #: Timestep this plant is designed for, chosen from its fastest time constant.
    default_dt: float = 0.01

    #: Default integrator key (see :mod:`.integrate`).
    default_integrator: str = 'rk4'

    #: States that carry a setpoint. Step-response metrics are computed on these,
    #: in this order, against the setpoint vector held by the environment.
    tracked_states: Sequence[str] = ()

    #: Relative weight of each tracked state in the RL reward. Same length as
    #: ``tracked_states``; the first entry is normally the primary objective.
    tracked_weights: Sequence[float] = ()

    def __init__(self, mode: Optional[ControlMode] = None):
        super().__init__()
        mode = ControlMode(mode) if mode is not None else self.default_mode
        if mode not in self.supported_modes:
            raise ValueError(
                f"{type(self).__name__} does not support control mode '{mode}'; "
                f"supported: {[str(m) for m in self.supported_modes]}"
            )
        self.mode = mode

    # ------------------------------------------------------------------ shape

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of action components in the current control mode."""

    @property
    @abstractmethod
    def action_names(self) -> List[str]:
        """Names of the action components in the current control mode."""

    @abstractmethod
    def action_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(low, high)`` action limits for the current mode, each ``(action_dim,)``."""

    @abstractmethod
    def state_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(low, high)`` observation limits, each ``(state_dim,)``. Used for the
        Gym observation space; the dynamics are not clipped to them."""

    # --------------------------------------------------------------- dynamics

    @abstractmethod
    def derivatives(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        State derivative, ``(batch, state_dim)``.

        ``u`` is the *physical* input produced by :meth:`map_action` (plus any
        disturbance the caller has already added), not the raw agent action.
        """

    def map_action(self, action: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Convert an action in the current control mode to the physical input ``u``.

        ``state`` is passed because modes higher up the actuator stack close a fast
        inner loop (a velocity command needs the current velocity to produce a force).

        The default is the identity, correct whenever the mode already speaks the
        dynamics' native input.
        """
        return action

    def disturbance_channel(self) -> int:
        """Index into ``u`` that a scalar disturbance is added to. Override per plant."""
        return 0

    # ------------------------------------------------------------------ reset

    @abstractmethod
    def initial_state(self, batch: int, generator: Optional[torch.Generator] = None,
                      randomize: bool = True) -> torch.Tensor:
        """Sample ``(batch, state_dim)`` initial states."""

    # -------------------------------------------------------------- utilities

    def default_setpoint(self) -> torch.Tensor:
        """Default target for :attr:`tracked_states`, shape ``(len(tracked_states),)``."""
        return torch.zeros(len(self.tracked_states), dtype=self._dtype(), device=self._device())

    @property
    def tracked_indices(self) -> List[int]:
        return [self.state_index(n) for n in self.tracked_states]

    def state_index(self, name: str) -> int:
        try:
            return list(self.state_names).index(name)
        except ValueError:
            raise KeyError(
                f"{type(self).__name__} has no state '{name}'; "
                f"available: {list(self.state_names)}"
            ) from None

    def named_state(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a state tensor into ``{name: column}`` for readable code and logging."""
        return {name: state[..., i] for i, name in enumerate(self.state_names)}

    def energy(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Total mechanical energy, ``(batch,)``, or ``None`` if not defined.

        Used by the physics tests: with damping and input zeroed, a conservative
        plant must hold this constant, which catches sign errors and missing terms
        in the equations of motion that a trajectory eyeball test would not.
        """
        return None

    def _device(self) -> torch.device:
        for t in self.buffers():
            return t.device
        return torch.device('cpu')

    def _dtype(self) -> torch.dtype:
        for t in self.buffers():
            if t.is_floating_point():
                return t.dtype
        return torch.get_default_dtype()

    def extra_repr(self) -> str:
        return f"mode={self.mode}, dt={self.default_dt}, states={list(self.state_names)}"
