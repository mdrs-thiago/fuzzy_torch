"""Rendering backends.

Two of them, chosen automatically:

* **MuJoCo** (preferred) draws a real 3-D scene with lighting, shadows and materials.
  It is a *renderer only*: we set ``qpos`` from our torch state and call
  ``mj_forward``. ``mj_step`` is never called, so MuJoCo never integrates the plant.
* **pygame-ce** is a flat 2-D fallback that needs no GL context, so rendering keeps
  working headless and in CI where MuJoCo cannot get a display.

``make_renderer`` picks MuJoCo when it imports and can obtain a GL context, and
falls back silently otherwise.
"""
import math
import os
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch

BG = (245, 245, 248)
INK = (30, 30, 40)
ACCENT = (200, 60, 60)
TARGET = (60, 140, 200)
STEEL = (110, 115, 130)


class Renderer:
    """Draws one plant state. Subclasses implement a backend."""

    def draw(self, state: torch.Tensor, setpoint: torch.Tensor, t: float):
        raise NotImplementedError

    def close(self):
        pass


# --------------------------------------------------------------------- pygame

def _pygame():
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the pygame renderer needs pygame-ce; install with: pip install 'fuzzy_torch[envs]'"
        ) from exc
    return pygame


class PygameRenderer(Renderer):
    """Flat 2-D fallback. Owns the window/surface lifecycle."""

    width, height = 720, 320

    def __init__(self, plant, mode: str = 'human'):
        pygame = _pygame()
        self.plant = plant
        self.mode = mode
        if mode == 'rgb_array' and not os.environ.get('SDL_VIDEODRIVER'):
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        if mode == 'human':
            pygame.display.set_caption(type(plant).__name__)
            self.surface = pygame.display.set_mode((self.width, self.height))
        else:
            self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)

    def draw(self, state, setpoint, t):
        pygame = _pygame()
        if self.mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
        self.surface.fill(BG)
        self._draw_plant(state.detach().cpu(), setpoint.detach().cpu(), t)
        if self.mode == 'human':
            pygame.display.flip()
            self.clock.tick(int(round(1.0 / max(self.plant.default_dt, 1e-3))))
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def _label(self, lines, x=10, y=10):
        for i, line in enumerate(lines):
            self.surface.blit(self.font.render(line, True, INK), (x, y + 18 * i))

    def _draw_plant(self, state, setpoint, t):
        raise NotImplementedError

    def close(self):
        pygame = _pygame()
        if self.mode == 'human':
            pygame.display.quit()
        pygame.quit()


class CranePygameRenderer(PygameRenderer):
    """Side view: rail, trolley, cable and payload."""

    def _draw_plant(self, state, setpoint, t):
        pygame = _pygame()
        x = float(state[0])
        theta = float(state[2])
        L = float(state[4]) if self.plant.hoist else float(self.plant.L0)

        limit = max(self.plant.x_limit, 1e-6)
        # One scale for both axes: the cable must look the length it is, or the
        # swing angle reads wrong.
        px_per_m = (self.width * 0.44) / limit
        cx, rail_y = self.width // 2, 74

        pygame.draw.line(self.surface, STEEL, (30, rail_y), (self.width - 30, rail_y), 5)

        tx = cx + x * px_per_m
        target_x = cx + float(setpoint[0]) * px_per_m
        pygame.draw.line(self.surface, TARGET, (target_x, rail_y - 22),
                         (target_x, rail_y + 22), 3)
        pygame.draw.rect(self.surface, INK,
                         pygame.Rect(tx - 26, rail_y - 14, 52, 28), border_radius=4)

        px = tx + L * math.sin(theta) * px_per_m
        py = rail_y + L * math.cos(theta) * px_per_m
        pygame.draw.line(self.surface, STEEL, (tx, rail_y), (px, py), 2)
        pygame.draw.circle(self.surface, ACCENT, (int(px), int(py)), 14)

        ground = rail_y + (self.plant.length_limits[1] + 0.4) * px_per_m
        if ground < self.height - 78:
            pygame.draw.line(self.surface, (215, 217, 224),
                             (30, ground), (self.width - 30, ground), 2)

        self._label([
            f"t     {t:7.2f} s",
            f"x     {x:7.3f} m   target {float(setpoint[0]):+.3f}",
            f"theta {math.degrees(theta):+7.2f} deg",
            f"L     {L:7.3f} m",
        ], y=self.height - 76)


class DronePygameRenderer(PygameRenderer):
    """Top-down fallback with altitude and heading readouts."""

    def _draw_plant(self, state, setpoint, t):
        pygame = _pygame()
        scale, cx, cy = 32.0, self.width // 2, self.height // 2
        x, y, z = (float(v) for v in state[:3])
        qw, qx, qy, qz = (float(v) for v in state[6:10])
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        point = lambda px, py: (int(cx + px * scale), int(cy - py * scale))
        target = point(float(setpoint[0]), float(setpoint[1]))
        pygame.draw.circle(self.surface, TARGET, target, 14, 2)
        pygame.draw.line(self.surface, TARGET, target, point(float(setpoint[0]), float(setpoint[1])), 1)
        center = point(x, y)
        arm = .30
        for sign in (-1.0, 1.0):
            dx, dy = math.cos(yaw + sign * math.pi / 4) * arm, math.sin(yaw + sign * math.pi / 4) * arm
            pygame.draw.line(self.surface, STEEL, point(x - dx, y - dy), point(x + dx, y + dy), 4)
        pygame.draw.circle(self.surface, ACCENT, center, 8)
        nose = point(x + math.cos(yaw) * .22, y + math.sin(yaw) * .22)
        pygame.draw.line(self.surface, INK, center, nose, 3)
        self._label([f't       {t:6.2f} s', f'position {x:+5.2f}, {y:+5.2f}, {z:5.2f} m',
                     f'target   {float(setpoint[0]):+5.2f}, {float(setpoint[1]):+5.2f}, {float(setpoint[2]):5.2f} m'],
                    y=self.height - 62)


# --------------------------------------------------------------------- MuJoCo

class MujocoRenderer(Renderer):
    """
    3-D renderer driven by our state, not by MuJoCo's integrator.

    Each frame sets ``qpos`` from the plant state and calls ``mj_forward``, which
    updates the kinematics for drawing. ``mj_step`` is never called.

    The two modes use *different, mutually exclusive* GL paths:

    * ``human`` -> ``mujoco.viewer.launch_passive``, MuJoCo's own GLFW window.
    * ``rgb_array`` -> an offscreen ``mujoco.Renderer``.

    They must never both exist in one process. Mixing MuJoCo's GLFW context with an
    SDL/pygame window makes the second ``GLXMakeCurrent`` fail with
    ``BadAccess`` under XWayland -- so the live view is MuJoCo's own window, not a
    MuJoCo frame blitted into a pygame surface.
    """

    width, height = 1280, 720

    def __init__(self, plant, mode: str = 'human',
                 camera: Optional[Dict] = None):
        import mujoco
        self._mujoco = mujoco
        self.plant = plant
        self.mode = mode

        spec = _MJCF_BUILDERS[type(plant).__name__]
        self.model = mujoco.MjModel.from_xml_string(spec(plant, self.width, self.height))
        self.data = mujoco.MjData(self.model)
        self._qpos_of = _QPOS_BUILDERS[type(plant).__name__]
        self._cam_cfg = camera or _CAMERAS[type(plant).__name__](plant)

        self.renderer = None
        self._viewer = None
        self._last_frame = None

        if mode == 'human':
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False)
            self._apply_camera(self._viewer.cam)
        else:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self._apply_camera(self.camera)

    def _apply_camera(self, cam):
        for key, value in self._cam_cfg.items():
            if key == 'lookat':
                cam.lookat[:] = value
            else:
                setattr(cam, key, value)

    def draw(self, state, setpoint, t):
        qpos = self._qpos_of(self.plant, state.detach().cpu())
        self.data.qpos[:len(qpos)] = qpos
        self.data.qvel[:] = 0.0
        self.data.time = t
        self._mujoco.mj_forward(self.model, self.data)

        if self._viewer is not None:
            if not self._viewer.is_running():
                return None
            self._viewer.sync()
            self._pace(t)
            return None

        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def _pace(self, t: float):
        """Throttle the live view to wall-clock time; sync() does not block."""
        import time
        now = time.perf_counter()
        if self._last_frame is None:
            self._start = now - t
        delay = (self._start + t) - now
        if delay > 0:
            time.sleep(min(delay, 0.1))
        self._last_frame = now

    @property
    def is_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


# ------------------------------------------------------- per-plant MuJoCo specs

def _crane_mjcf(plant, width, height) -> str:
    from ..assets import crane_mjcf
    return crane_mjcf(
        trolley_mass=float(plant.M), payload_mass=float(plant.m),
        cable_length=float(plant.L0), gravity=float(plant.g),
        x_limit=plant.x_limit, target_x=plant.target_x,
        offwidth=width, offheight=height,
    )


def _crane_camera(plant) -> Dict:
    return {'lookat': [plant.target_x * 0.4, 0.0, float(plant.L0) + 1.2],
            'distance': plant.x_limit * 1.95, 'azimuth': 122.0, 'elevation': -16.0}


def _crane_qpos(plant, state) -> np.ndarray:
    return np.array([float(state[0]), float(state[2])])


def _drone_mjcf(plant, width, height) -> str:
    from ..assets import quadrotor_mjcf
    return quadrotor_mjcf(plant.target, offwidth=width, offheight=height)


def _drone_camera(plant) -> Dict:
    return {'lookat': [0.0, 0.0, plant.target[2] * .55], 'distance': 6.5,
            'azimuth': 135.0, 'elevation': -24.0}


def _drone_qpos(plant, state) -> np.ndarray:
    return np.array([float(state[0]), float(state[1]), float(state[2]),
                     float(state[6]), float(state[7]), float(state[8]), float(state[9])])


_MJCF_BUILDERS: Dict[str, Callable] = {}
_CAMERAS: Dict[str, Callable] = {}
_QPOS_BUILDERS: Dict[str, Callable] = {}
_PYGAME_RENDERERS: Dict[str, Type[PygameRenderer]] = {}


def register_plant(plant_cls, pygame_cls=None, mjcf=None, camera=None, qpos=None):
    """Register the rendering hooks for a plant class."""
    name = plant_cls.__name__
    if pygame_cls is not None:
        _PYGAME_RENDERERS[name] = pygame_cls
    if mjcf is not None:
        _MJCF_BUILDERS[name] = mjcf
        _CAMERAS[name] = camera
        _QPOS_BUILDERS[name] = qpos


def mujoco_available(plant) -> Tuple[bool, str]:
    """Whether the MuJoCo backend can actually be used for this plant."""
    if type(plant).__name__ not in _MJCF_BUILDERS:
        return False, "no MJCF registered for this plant"
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False, "mujoco is not installed (pip install 'fuzzy_torch[viz]')"
    try:
        # Obtaining a context is the part that fails headless without EGL/OSMesa,
        # and it fails at construction rather than at import.
        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
        mujoco.Renderer(model, 64, 64).close()
    except Exception as exc:                      # noqa: BLE001 - any GL failure
        return False, f"no usable OpenGL context ({type(exc).__name__})"
    return True, ""


def make_renderer(plant, mode: str, backend: str = 'auto') -> Renderer:
    """
    Build a renderer for ``plant``.

    Args:
        backend: ``'mujoco'``, ``'pygame'``, or ``'auto'`` (MuJoCo when usable).
    """
    name = type(plant).__name__
    if backend not in ('auto', 'mujoco', 'pygame'):
        raise ValueError(f"unknown backend '{backend}'")

    if backend in ('auto', 'mujoco'):
        ok, why = mujoco_available(plant)
        if ok:
            return MujocoRenderer(plant, mode)
        if backend == 'mujoco':
            raise RuntimeError(f"MuJoCo backend unavailable: {why}")

    cls = _PYGAME_RENDERERS.get(name)
    if cls is None:
        raise NotImplementedError(f"no renderer registered for {name}")
    return cls(plant, mode)


def _register_builtin():
    from ..plants.crane import CranePlant
    from ..plants.drone import QuadrotorPlant
    register_plant(CranePlant, pygame_cls=CranePygameRenderer,
                   mjcf=_crane_mjcf, camera=_crane_camera, qpos=_crane_qpos)
    register_plant(QuadrotorPlant, pygame_cls=DronePygameRenderer,
                   mjcf=_drone_mjcf, camera=_drone_camera, qpos=_drone_qpos)


_register_builtin()
