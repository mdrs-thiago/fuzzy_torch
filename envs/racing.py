"""Closed-circuit time-trial environment for fuzzy racing controllers.

The vehicle uses :class:`DiffDriveRobot`'s dynamic planar model, while this Gymnasium
task supplies the track geometry, lap accounting and a racing reward.  It is deliberately
a *time trial*: all controllers start at the same line and a run ends on a completed lap,
track exit, or the time limit.  This makes lap time meaningful rather than rewarding a
controller that cuts a corner.
"""
from dataclasses import dataclass
import math
import os
import time as wall_time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("RacingEnv needs gymnasium; install with: pip install 'fuzzy_torch[envs]'") from exc

from .core.integrate import get_integrator
from .core.modes import ControlMode
from .plants.robot import DiffDriveRobot


def _wrap(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class RaceTrack:
    """A sampled, closed centre line. Coordinates are metres, arc length is metres."""
    name: str
    points: np.ndarray
    width: float = 2.0

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError("points must have shape (n >= 4, 2)")
        object.__setattr__(self, 'points', points)
        delta = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("track has repeated adjacent points")
        object.__setattr__(self, '_delta', delta)
        object.__setattr__(self, '_lengths', lengths)
        object.__setattr__(self, '_starts', np.r_[0.0, np.cumsum(lengths[:-1])])
        object.__setattr__(self, 'length', float(lengths.sum()))
        tangent = np.arctan2(delta[:, 1], delta[:, 0])
        object.__setattr__(self, '_tangent', tangent)
        # Signed curvature of each polyline segment, useful as a look-ahead signal.
        turn = np.asarray([_wrap(tangent[(i + 1) % len(tangent)] - tangent[i])
                           for i in range(len(tangent))])
        object.__setattr__(self, '_curvature', turn / lengths)

    def project(self, position: np.ndarray) -> Tuple[float, float, float, int]:
        """Return ``(arc_length, signed_cross_track_error, tangent, segment)``."""
        rel = position[None, :] - self.points
        u = np.clip((rel * self._delta).sum(axis=1) / self._lengths ** 2, 0.0, 1.0)
        projected = self.points + u[:, None] * self._delta
        offset = position[None, :] - projected
        i = int(np.argmin((offset ** 2).sum(axis=1)))
        # 2-D cross product: positive means vehicle is left of the travel direction.
        cte = float((self._delta[i, 0] * offset[i, 1] -
                     self._delta[i, 1] * offset[i, 0]) / self._lengths[i])
        return float(self._starts[i] + u[i] * self._lengths[i]), cte, float(self._tangent[i]), i

    def ahead(self, arc: float) -> Tuple[np.ndarray, float]:
        """Point and curvature at a wrapped distance along the centre line."""
        arc = arc % self.length
        i = int(np.searchsorted(self._starts, arc, side='right') - 1) % len(self.points)
        local = (arc - self._starts[i]) / self._lengths[i]
        return self.points[i] + local * self._delta[i], float(self._curvature[i])

    def tangent_at(self, arc: float) -> float:
        """Travel heading at a wrapped distance along the centre line.

        This is deliberately separate from :meth:`ahead`: the latter's second
        return value is curvature, which is a very different quantity despite
        both being expressed as scalars.
        """
        arc = arc % self.length
        i = int(np.searchsorted(self._starts, arc, side='right') - 1) % len(self.points)
        return float(self._tangent[i])


def make_track(name: str, samples: int = 240, width: float = 2.0) -> RaceTrack:
    """Create one of the lecture circuits: ``oval``, ``chicane`` or ``technical``."""
    if samples < 32:
        raise ValueError("samples must be at least 32")
    t = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    if name == 'oval':
        points = np.c_[8.0 * np.cos(t), 4.5 * np.sin(t)]
    elif name == 'chicane':
        # An asymmetric, non-self-intersecting loop with two quick direction changes.
        points = np.c_[7.5 * np.cos(t), 3.8 * np.sin(t) + 1.05 * np.sin(3.0 * t)]
    elif name == 'technical':
        radius = 6.2 + 1.25 * np.sin(3.0 * t) - 0.55 * np.cos(2.0 * t)
        points = np.c_[radius * np.cos(t), radius * np.sin(t)]
    else:
        raise ValueError("unknown circuit; choose 'oval', 'chicane', or 'technical'")
    return RaceTrack(name, points, width)


class RacingEnv(gym.Env):
    """Gymnasium time-trial task over :class:`DiffDriveRobot`.

    Observation order is ``[cross_track_error, heading_error, speed, yaw_rate,
    lookahead_curvature, progress]``.  The action is ``[throttle, steering]`` in
    ``[-1, 1]``.  Negative throttle is braking; steering requests a yaw rate through
    the robot's fast inner loop.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    feature_names = ('cross_track_error', 'heading_error', 'speed', 'yaw_rate',
                     'lookahead_curvature', 'progress')

    def __init__(self, track: str = 'oval', *, dt: float = 0.02,
                 max_time: float = 45.0, lookahead: float = 1.8,
                 vehicle: Optional[DiffDriveRobot] = None,
                 render_mode: Optional[str] = None, render_backend: str = 'auto'):
        super().__init__()
        self.track = make_track(track)
        if render_mode not in (None, 'human', 'rgb_array'):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        self.render_mode = render_mode
        if render_backend not in ('auto', 'mujoco', 'pygame'):
            raise ValueError("render_backend must be 'auto', 'mujoco', or 'pygame'")
        self.render_backend = render_backend
        self.dt, self.max_time, self.lookahead = float(dt), float(max_time), float(lookahead)
        self.vehicle = vehicle or DiffDriveRobot(mode=ControlMode.VELOCITY, max_speed=4.0,
                                                   max_yaw_rate=3.2, world_size=20.0)
        if self.vehicle.mode is not ControlMode.VELOCITY:
            raise ValueError("RacingEnv requires DiffDriveRobot in VELOCITY mode")
        self._step_fn = get_integrator(self.vehicle.default_integrator)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-self.track.width, -math.pi, -0.1, -8.0, -3.0, 0.0], np.float32),
            high=np.array([self.track.width, math.pi, self.vehicle.max_speed * 1.2,
                           8.0, 3.0, 1.05], np.float32), dtype=np.float32)
        self._state = None
        self._renderer = None
        self._step_index = 0
        self._arc = self._last_arc = self._lap_progress = 0.0

    @property
    def time(self) -> float:
        return self._step_index * self.dt

    def _features(self) -> np.ndarray:
        state = self._state[0].detach().cpu().numpy()
        arc, cte, tangent, _ = self.track.project(state[:2])
        _, curvature = self.track.ahead(arc + self.lookahead)
        return np.array([cte, _wrap(tangent - state[2]), state[3], state[5], curvature,
                         np.clip(self._lap_progress / self.track.length, 0.0, 1.05)], np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        start, _ = self.track.ahead(0.0)
        tangent = self.track.tangent_at(0.0)
        self._state = torch.zeros((1, self.vehicle.state_dim), dtype=torch.float32)
        self._state[0, 0:2] = torch.as_tensor(start, dtype=torch.float32)
        self._state[0, 2] = tangent
        self._step_index = 0
        self._arc = self._last_arc = self._lap_progress = 0.0
        if self.render_mode == 'human':
            self.render()
        return self._features(), {'track': self.track.name, 'lap_time': None}

    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        # Brake is represented by a zero speed command; reverse is excluded from racing.
        command = torch.tensor([[max(0.0, float(action[0])) * self.vehicle.max_speed,
                                 float(action[1]) * self.vehicle.max_yaw_rate]], dtype=torch.float32)
        physical = self.vehicle.map_action(command, self._state)
        self._state = self._step_fn(self.vehicle.derivatives, self._state, physical, self.dt)
        self._step_index += 1

        state = self._state[0].detach().cpu().numpy()
        arc, cte, _, _ = self.track.project(state[:2])
        delta = arc - self._last_arc
        if delta < -self.track.length / 2:
            delta += self.track.length
        elif delta > self.track.length / 2:
            delta -= self.track.length
        self._arc, self._last_arc = arc, arc
        self._lap_progress += delta

        off_track = abs(cte) > self.track.width / 2.0
        finished = self._lap_progress >= self.track.length
        timeout = self.time >= self.max_time
        # Dense progress reward helps RL, but official ranking is lap_time in info.
        reward = 8.0 * delta - 0.02 * abs(cte) - 0.002 * float(action[1] ** 2)
        if off_track:
            reward -= 25.0
        info = {'track': self.track.name, 'lap_time': self.time if finished else None,
                'progress': max(0.0, self._lap_progress), 'cross_track_error': cte,
                'off_track': off_track, 'speed': float(state[3])}
        if self.render_mode == 'human':
            self.render()
        return self._features(), float(reward), bool(finished or off_track), bool(timeout), info

    def render(self):
        """Draw the road, centre line and car.  Imports pygame only when requested."""
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("render() called before reset()")
        if self._renderer is None:
            if self.render_backend in ('auto', 'mujoco'):
                try:
                    self._renderer = _MujocoRaceRenderer(self.track, self.render_mode, self.dt)
                except Exception:
                    if self.render_backend == 'mujoco':
                        raise
            if self._renderer is None:
                self._renderer = _RaceRenderer(self.track, self.render_mode, self.dt)
        state = self._state[0].detach().cpu().numpy()
        return self._renderer.draw(state, self.time, self._lap_progress / self.track.length)

    @property
    def is_running(self) -> bool:
        return self._renderer is None or self._renderer.running

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _race_mjcf(track: RaceTrack) -> str:
    """Make a visual MuJoCo scene; torch remains the authoritative dynamics engine."""
    road, kerbs, rails = [], [], []
    for i, (point, delta, length) in enumerate(zip(track.points, track._delta, track._lengths)):
        centre = point + delta / 2.0
        yaw = math.atan2(delta[1], delta[0])
        road.append(
            f'<geom type="box" pos="{centre[0]:.4f} {centre[1]:.4f} .025" '
            f'euler="0 0 {yaw:.6f}" size="{length / 2 + .025:.4f} {track.width / 2:.4f} .025" '
            'rgba=".11 .12 .14 1"/>')
        # Alternating red/white blocks make the circuit read like a racing circuit.
        if i % 4 == 0:
            normal = np.array([-math.sin(yaw), math.cos(yaw)]) * (track.width / 2 + .10)
            color = '.86 .12 .08 1' if (i // 4) % 2 == 0 else '.95 .95 .92 1'
            for side in (-1.0, 1.0):
                p = centre + side * normal
                kerbs.append(
                    f'<geom type="box" pos="{p[0]:.4f} {p[1]:.4f} .055" euler="0 0 {yaw:.6f}" '
                    f'size="{length / 2:.4f} .09 .055" rgba="{color}"/>')
        # Low guardrails make the racing boundary legible from the chase camera.
        if i % 6 == 0:
            normal = np.array([-math.sin(yaw), math.cos(yaw)]) * (track.width / 2 + .28)
            for side in (-1.0, 1.0):
                p = centre + side * normal
                rails.append(
                    f'<geom type="box" pos="{p[0]:.4f} {p[1]:.4f} .16" euler="0 0 {yaw:.6f}" '
                    f'size="{length / 2:.4f} .045 .16" rgba=".16 .20 .27 1"/>')
    start, _ = track.ahead(0.0)
    tangent = track.tangent_at(0.0)
    gantry_normal = np.array([-math.sin(tangent), math.cos(tangent)])
    # Build the line out of individual tiles in the local track frame.  A single
    # long rotated box looked like a skewed banner at the closed-seam on some tracks.
    finish_tiles = []
    tile_width = track.width / 8.0
    for i in range(8):
        p = start + gantry_normal * ((i + .5) * tile_width - track.width / 2.0)
        color = '.98 .98 .98 1' if i % 2 == 0 else '.06 .07 .09 1'
        finish_tiles.append(
            f'<geom type="box" pos="{p[0]:.4f} {p[1]:.4f} .062" euler="0 0 {tangent:.6f}" '
            f'size=".10 {tile_width / 2:.4f} .018" rgba="{color}"/>')
    # Original, toy-like scenery: colourful trees and a start gantry, rather than
    # copyrighted game assets.  It gives speed and scale cues to the camera.
    # Offset each tree along the local track normal and keep it only if it clears
    # the whole road: a radial offset from the origin lands on the asphalt wherever
    # the circuit is non-convex (the chicane's waist, the technical's lobes).
    trees = []
    clearance = track.width / 2 + 0.9
    for i in range(16):
        k = (i * len(track.points)) // 16
        yaw = math.atan2(track._delta[k, 1], track._delta[k, 0])
        normal = np.array([-math.sin(yaw), math.cos(yaw)])
        for side in ((1.0, -1.0) if i % 2 else (-1.0, 1.0)):
            p = track.points[k] + side * normal * (track.width / 2 + 1.4 + (i % 3) * .45)
            if np.min(np.linalg.norm(track.points - p, axis=1)) > clearance:
                break
        else:
            continue
        trunk = f'<geom type="cylinder" pos="{p[0]:.3f} {p[1]:.3f} .42" size=".10 .42" rgba=".34 .17 .06 1"/>'
        crown = f'<geom type="sphere" pos="{p[0]:.3f} {p[1]:.3f} 1.08" size=".48" rgba=".12 {.42 + .04 * (i % 3):.2f} .18 1"/>'
        trees.append(trunk + crown)
    # A genuine overhead FINISH board.  MuJoCo geoms have no text primitive, so the
    # lettering is made from small, high-contrast blocks on the front face.
    glyphs = {
        'F': ('11111', '10000', '11110', '10000', '10000', '10000', '10000'),
        'I': ('11111', '00100', '00100', '00100', '00100', '00100', '11111'),
        'N': ('10001', '11001', '10101', '10011', '10001', '10001', '10001'),
        'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'),
        'H': ('10001', '10001', '10001', '11111', '10001', '10001', '10001'),
    }
    cell, letter_width, gap = .075, .375, .055
    word_width = 6 * letter_width + 5 * gap
    board_half_width = max(track.width / 2 + .32, word_width / 2 + .16)
    letters = []
    for letter_i, char in enumerate('FINISH'):
        for row, pattern in enumerate(glyphs[char]):
            for col, bit in enumerate(pattern):
                if bit == '1':
                    y = -word_width / 2 + letter_i * (letter_width + gap) + (col + .5) * cell
                    z = 2.43 - (row + .5) * cell
                    p = start - np.array([math.cos(tangent), math.sin(tangent)]) * .091 + gantry_normal * y
                    letters.append(
                        f'<geom type="box" pos="{p[0]:.4f} {p[1]:.4f} {z:.4f}" '
                        f'euler="0 0 {tangent:.6f}" size=".012 {cell / 2:.4f} {cell / 2:.4f}" '
                        'rgba="1 .92 .14 1"/>')
    left, right = start - gantry_normal * board_half_width, start + gantry_normal * board_half_width
    gantry = f'''<geom type="cylinder" pos="{left[0]:.4f} {left[1]:.4f} 1.15" size=".075 1.15" rgba=".92 .72 .08 1"/>
      <geom type="cylinder" pos="{right[0]:.4f} {right[1]:.4f} 1.15" size=".075 1.15" rgba=".92 .72 .08 1"/>
      <geom type="box" pos="{start[0]:.4f} {start[1]:.4f} 2.18" euler="0 0 {tangent:.6f}" size=".08 {board_half_width:.4f} .42" rgba=".12 .36 .86 1"/>
      {''.join(letters)}'''
    return f'''<mujoco model="fuzzy_grand_prix">
    <compiler angle="radian"/>
    <visual><global offwidth="1280" offheight="720" fovy="52"/></visual>
    <asset><texture type="skybox" builtin="gradient" rgb1=".55 .75 .95" rgb2=".12 .22 .40" width="256" height="1536"/></asset>
    <worldbody>
      <light pos="-8 -6 18" dir=".3 .25 -1" directional="true" diffuse=".95 .95 .95"/>
      <light pos="8 6 9" dir="-.4 -.3 -1" directional="true" diffuse=".35 .45 .60"/>
      <geom type="plane" size="30 30 .1" rgba=".08 .40 .17 1"/>
      {''.join(road)}{''.join(kerbs)}{''.join(rails)}{''.join(trees)}{''.join(finish_tiles)}{gantry}
      <!-- A low, open-wheel rental kart: the visual is deliberately assembled from
           simple primitives, so it remains asset-free and readable at race speed. -->
      <body name="kart" pos="0 0 .20">
        <freejoint/>
        <!-- tubular chassis and floor pan -->
        <geom type="capsule" fromto="-.51 -.25 .066 .51 -.25 .066" size=".022" rgba=".08 .10 .12 1"/>
        <geom type="capsule" fromto="-.51 .25 .066 .51 .25 .066" size=".022" rgba=".08 .10 .12 1"/>
        <geom type="capsule" fromto="-.46 -.29 .074 -.46 .29 .074" size=".022" rgba=".08 .10 .12 1"/>
        <geom type="capsule" fromto=".39 -.31 .074 .39 .31 .074" size=".022" rgba=".08 .10 .12 1"/>
        <geom type="box" pos="-.04 0 .062" size=".39 .22 .021" rgba=".16 .18 .20 1"/>
        <!-- nose, side pods and rear bumper bodywork -->
        <geom type="ellipsoid" pos=".48 0 .13" size=".24 .19 .082" rgba=".90 .06 .035 1"/>
        <geom type="box" pos=".28 .30 .115" size=".25 .062 .062" rgba=".90 .06 .035 1"/>
        <geom type="box" pos=".28 -.30 .115" size=".25 .062 .062" rgba=".90 .06 .035 1"/>
        <geom type="box" pos="-.52 0 .14" size=".045 .35 .037" rgba=".05 .06 .07 1"/>
        <geom type="box" pos="-.52 0 .21" size=".029 .28 .021" rgba=".90 .06 .035 1"/>
        <geom type="box" pos=".65 0 .13" size=".029 .35 .029" rgba=".05 .06 .07 1"/>
        <!-- exposed wheels, hubs and front axle -->
        <geom type="capsule" fromto=".46 -.39 0 .46 .39 0" size=".020" rgba=".32 .34 .36 1"/>
        <geom type="cylinder" pos=".46 .39 0" euler="1.5708 0 0" size=".119 .062" rgba=".018 .020 .023 1"/>
        <geom type="cylinder" pos=".46 -.39 0" euler="1.5708 0 0" size=".119 .062" rgba=".018 .020 .023 1"/>
        <geom type="cylinder" pos="-.40 .40 .016" euler="1.5708 0 0" size=".144 .078" rgba=".018 .020 .023 1"/>
        <geom type="cylinder" pos="-.40 -.40 .016" euler="1.5708 0 0" size=".144 .078" rgba=".018 .020 .023 1"/>
        <geom type="cylinder" pos=".46 .46 0" euler="1.5708 0 0" size=".066 .008" rgba=".72 .74 .75 1"/>
        <geom type="cylinder" pos=".46 -.46 0" euler="1.5708 0 0" size=".066 .008" rgba=".72 .74 .75 1"/>
        <geom type="cylinder" pos="-.40 .48 .016" euler="1.5708 0 0" size=".078 .008" rgba=".72 .74 .75 1"/>
        <geom type="cylinder" pos="-.40 -.48 .016" euler="1.5708 0 0" size=".078 .008" rgba=".72 .74 .75 1"/>
        <!-- bucket seat, driver and a compact steering column -->
        <geom type="ellipsoid" pos="-.15 0 .205" size=".20 .19 .13" rgba=".035 .045 .055 1"/>
        <geom type="ellipsoid" pos="-.18 0 .345" size=".14 .16 .20" rgba=".12 .17 .24 1"/>
        <geom type="sphere" pos="-.04 0 .60" size=".12" rgba=".05 .26 .66 1"/>
        <geom type="ellipsoid" pos=".037 -.086 .60" size=".086 .015 .041" rgba=".08 .12 .16 1"/>
        <geom type="capsule" fromto=".14 0 .205 .05 0 .42" size=".017" rgba=".18 .20 .22 1"/>
        <!-- Its axis is pitched toward the driver; the old wheel lay vertical. -->
        <geom type="cylinder" pos=".05 0 .42" euler="0 .78 0" size=".094 .013" rgba=".07 .08 .09 1"/>
        <geom type="cylinder" pos=".05 0 .42" euler="0 .78 0" size=".021 .016" rgba=".92 .72 .08 1"/>
        <!-- rear engine, exhaust and chain guard make its layout unmistakable -->
        <geom type="box" pos="-.39 0 .30" size=".12 .16 .10" rgba=".16 .18 .20 1"/>
        <geom type="cylinder" pos="-.41 0 .42" size=".078 .037" rgba=".38 .40 .42 1"/>
        <geom type="capsule" fromto="-.48 -.15 .30 -.63 -.15 .33" size=".026" rgba=".55 .57 .58 1"/>
        <geom type="box" pos="-.35 .25 .16" size=".16 .015 .082" rgba=".70 .58 .10 1"/>
      </body>
    </worldbody></mujoco>'''


class _MujocoRaceRenderer:
    """3-D kart and circuit renderer driven from the Gym environment's state."""
    width, height = 1280, 720

    def __init__(self, track: RaceTrack, mode: str, dt: float):
        import mujoco
        self.mujoco, self.track, self.mode, self.dt = mujoco, track, mode, dt
        self.model = mujoco.MjModel.from_xml_string(_race_mjcf(track))
        self.data = mujoco.MjData(self.model)
        self.viewer = self.renderer = None
        self.running, self._wall_start = True, None
        if mode == 'human':
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                       show_left_ui=False, show_right_ui=False)
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
            self.viewer.cam.distance = 4.5
            self.viewer.cam.elevation = -16.0
        else:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.lookat[:] = [0.0, 0.0, 0.0]
            self.camera.distance, self.camera.elevation = 4.5, -16.0

    def draw(self, state: np.ndarray, sim_time: float, progress: float):
        if self.viewer is not None and not self.viewer.is_running():
            self.running = False
            return None
        psi = float(state[2])
        # MuJoCo free-joint qpos: x, y, z, then unit quaternion w, x, y, z.
        self.data.qpos[:7] = [float(state[0]), float(state[1]), .20,
                              math.cos(psi / 2), 0.0, 0.0, math.sin(psi / 2)]
        self.data.qvel[:] = 0.0
        self.data.time = sim_time
        self.mujoco.mj_forward(self.model, self.data)
        forward = np.array([math.cos(psi), math.sin(psi)])
        lookat = [float(state[0] + 1.5 * forward[0]),
                  float(state[1] + 1.5 * forward[1]), .20]
        # MuJoCo's azimuth is the direction the camera *looks*, so looking along the
        # kart's heading puts the camera behind it.  (psi + 90 deg gave a side view.)
        azimuth = math.degrees(psi)
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = lookat
            self.viewer.cam.azimuth = azimuth
            self.viewer.sync()
            if self._wall_start is None:
                self._wall_start = wall_time.perf_counter() - sim_time
            delay = self._wall_start + sim_time - wall_time.perf_counter()
            if delay > 0:
                wall_time.sleep(min(delay, .05))
            return None
        self.camera.lookat[:] = lookat
        self.camera.azimuth = azimuth
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class _RaceRenderer:
    """Small pygame renderer kept local to the task because it draws track geometry."""
    width, height = 820, 680

    def __init__(self, track: RaceTrack, mode: str, dt: float):
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover
            raise ImportError("live racing rendering needs pygame-ce; install fuzzy_torch[envs]") from exc
        self.pygame, self.track, self.mode, self.dt = pygame, track, mode, dt
        if mode == 'rgb_array' and not os.environ.get('SDL_VIDEODRIVER'):
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        pygame.init()
        self.surface = (pygame.display.set_mode((self.width, self.height)) if mode == 'human'
                        else pygame.Surface((self.width, self.height)))
        if mode == 'human':
            pygame.display.set_caption(f'Fuzzy Grand Prix — {track.name}')
        self.font = pygame.font.SysFont('monospace', 16)
        self.clock, self.running = pygame.time.Clock(), True
        lo, hi = track.points.min(axis=0), track.points.max(axis=0)
        span = np.maximum(hi - lo + track.width * 2.4, 1.0)
        self.scale = min((self.width - 70) / span[0], (self.height - 70) / span[1])
        self.centre = (lo + hi) / 2.0

    def _pixel(self, point):
        p = np.asarray(point)
        return (int(self.width / 2 + (p[0] - self.centre[0]) * self.scale),
                int(self.height / 2 - (p[1] - self.centre[1]) * self.scale))

    def draw(self, state: np.ndarray, time: float, progress: float):
        pygame = self.pygame
        if self.mode == 'human':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
        self.surface.fill((58, 119, 66))
        line = [self._pixel(p) for p in np.vstack([self.track.points, self.track.points[:1]])]
        road_width = max(4, round(self.track.width * self.scale))
        pygame.draw.lines(self.surface, (48, 50, 56), True, line, road_width)
        pygame.draw.lines(self.surface, (235, 235, 218), True, line, 2)
        start, tangent = self.track.ahead(0.0)
        normal = np.array([-math.sin(tangent), math.cos(tangent)]) * self.track.width / 2
        pygame.draw.line(self.surface, (255, 255, 255), self._pixel(start - normal),
                         self._pixel(start + normal), 4)

        # A triangular car makes heading immediately visible in a lecture.
        position, psi = state[:2], float(state[2])
        forward = np.array([math.cos(psi), math.sin(psi)])
        left = np.array([-forward[1], forward[0]])
        car = [position + forward * .38,
               position - forward * .23 + left * .20,
               position - forward * .23 - left * .20]
        pygame.draw.polygon(self.surface, (205, 57, 53), [self._pixel(p) for p in car])
        lines = [f'time     {time:6.2f} s', f'progress {min(progress, 1.0) * 100:6.1f} %',
                 f'speed    {state[3]:6.2f} m/s']
        for i, text in enumerate(lines):
            self.surface.blit(self.font.render(text, True, (250, 250, 250)), (12, 12 + 20 * i))
        if self.mode == 'human':
            pygame.display.flip()
            self.clock.tick(max(1, round(1.0 / self.dt)))
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self):
        if self.mode == 'human':
            self.pygame.display.quit()
        self.pygame.quit()
