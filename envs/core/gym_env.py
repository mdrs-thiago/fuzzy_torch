"""Wrap a :class:`~fuzzy_torch.envs.core.plant.Plant` as a ``gymnasium.Env``.

Gymnasium's API is a contract over "arbitrary behind-the-scenes dynamics" -- it
requires only ``step``/``reset``/``render``/``close`` plus the two spaces, and says
nothing about physics engines or robot description files. So the torch dynamics sit
directly behind a standard ``Env``: SB3 and every Gym tool work unchanged, while the
same ``Plant`` is still available for batched GPU and differentiable rollouts.

Observations are the **raw plant state**. Engineered features (error, error rate)
come from :class:`~fuzzy_torch.envs.core.wrappers.ErrorFeatures`, so that the fuzzy
controller and the RL agent provably consume the same signal.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "fuzzy_torch.envs needs gymnasium; install with: pip install 'torch-fis[envs]'"
    ) from exc

from .integrate import get_integrator
from .plant import Plant
from .scenarios import Scenario


class FuzzyControlEnv(gym.Env):
    """A single-instance Gymnasium environment over a batched torch plant."""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self,
                 plant: Plant,
                 scenario: Optional[Scenario] = None,
                 dt: Optional[float] = None,
                 integrator: Optional[str] = None,
                 episode_steps: Optional[int] = None,
                 render_mode: Optional[str] = None,
                 render_backend: str = 'auto'):
        super().__init__()
        self.plant = plant
        self.dt = float(dt if dt is not None else plant.default_dt)
        self.integrator_name = integrator or plant.default_integrator
        self._step_fn = get_integrator(self.integrator_name)

        if scenario is None:
            steps = episode_steps if episode_steps is not None else int(round(10.0 / self.dt))
            scenario = Scenario(name='default', episode_steps=steps,
                                setpoint=plant.default_setpoint().tolist())
        self.scenario = scenario
        self.episode_steps = int(episode_steps or scenario.episode_steps)

        low, high = plant.state_bounds()
        self.observation_space = spaces.Box(low=low.detach().cpu().numpy().astype(np.float32),
                                            high=high.detach().cpu().numpy().astype(np.float32),
                                            dtype=np.float32)
        a_low, a_high = plant.action_bounds()
        self.action_space = spaces.Box(low=a_low.detach().cpu().numpy().astype(np.float32),
                                       high=a_high.detach().cpu().numpy().astype(np.float32),
                                       dtype=np.float32)

        self.render_mode = render_mode
        self.render_backend = render_backend
        self._renderer = None
        self._state: Optional[torch.Tensor] = None
        self._step_index = 0
        self._generator: Optional[torch.Generator] = None

    # --------------------------------------------------------------- helpers

    @property
    def time(self) -> float:
        return self._step_index * self.dt

    def setpoint(self, t: Optional[float] = None) -> torch.Tensor:
        t = self.time if t is None else t
        return self.scenario.setpoint_at(t, self.plant.default_setpoint())

    def tracked(self, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self._state if state is None else state
        return s[0, self.plant.tracked_indices]

    def _obs(self) -> np.ndarray:
        return self._state[0].detach().cpu().numpy().astype(np.float32)

    def _reward(self) -> float:
        """Weighted negative squared tracking error.

        Present so RL baselines have something to optimize. The *evaluation* never
        uses it -- controllers are judged by the classical metrics in
        :mod:`.metrics`, which is what makes PID, fuzzy and RL comparable at all.
        """
        err = self.setpoint() - self.tracked()
        w = torch.tensor(self.plant.tracked_weights or [1.0] * err.numel(),
                         dtype=err.dtype, device=err.device)
        return float(-(w * err ** 2).sum())

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._generator = torch.Generator(device=self.plant._device()).manual_seed(seed)
        self._state = self.plant.initial_state(
            1, generator=self._generator,
            randomize=self.scenario.randomize_initial_state)
        self._step_index = 0
        if self.render_mode == 'human':
            self.render()
        return self._obs(), {'time': 0.0, 'setpoint': self.setpoint().cpu().numpy()}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        act = torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1),
                              dtype=self._state.dtype, device=self._state.device)
        a_low, a_high = self.plant.action_bounds()
        act = torch.max(torch.min(act, a_high), a_low)

        control = self.plant.map_action(act, self._state)
        dist = self.scenario.disturbance_at(self.time, self.dt, control.shape[-1],
                                            self.plant.disturbance_channel())
        applied = control if dist is None else control + dist.to(control).unsqueeze(0)

        self._state = self._step_fn(self.plant.derivatives, self._state, applied, self.dt)
        self._step_index += 1

        reward = self._reward()
        # Fixed-length episodes with no early termination, so metrics like IAE are
        # integrated over the same horizon for every controller.
        truncated = self._step_index >= self.episode_steps
        info = {
            'time': self.time,
            'setpoint': self.setpoint().cpu().numpy(),
            # What the actuator did, *excluding* the disturbance. Control-effort
            # metrics must not bill the controller for a force it did not command.
            'control': control[0].detach().cpu().numpy(),
            # What the plant actually received: control + disturbance.
            'applied': applied[0].detach().cpu().numpy(),
            'disturbance': None if dist is None else dist.cpu().numpy(),
        }
        if self.render_mode == 'human':
            self.render()
        return self._obs(), reward, False, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from .render import make_renderer
            self._renderer = make_renderer(self.plant, self.render_mode,
                                           backend=self.render_backend)
        return self._renderer.draw(self._state[0], self.setpoint(), self.time)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
