"""SuperTuxKart races for fuzzy drivers.

`SuperTuxKart <https://supertuxkart.net>`_ is the open-source kart racer (GPLv3); its
Python bindings, ``pystk2``, run the real game: 3-D tracks with jumps and walls, AI
opponents, items, drifting and nitro.  :class:`SuperTuxKartEnv` wraps one race behind the
same interface as :class:`~fuzzy_torch.envs.racing.RacingEnv`: the observation has the
same six named features and the action is the same ``[throttle, steering]``.  A driver
written for the lecture circuits, including a Race Control JSON driver, therefore drives
a SuperTuxKart kart unchanged::

    env = SuperTuxKartEnv('lighthouse', opponents=3)
    driver = ConfigRaceDriver(json.load(open('my_driver.json')))
    obs, info = env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(driver(obs))
        if terminated or truncated:
            break
    print(info['race_time'], info['position'])

**Units.**  A SuperTuxKart road is about ten metres wide and karts reach 22 m/s, while
the lecture circuits are 2 m wide with a 4 m/s car.  With ``units='grand_prix'`` (the
default) the features are rescaled to that same dimensionless problem: one unit of
cross-track error is the road's half-width (so ``+-1`` is the road edge, exactly as in
``RacingEnv``), and speed is scaled so that the kart's top speed reads 4.  Curvature,
yaw rate and the look-ahead distance follow from those two scales.  ``units='si'`` gives
raw metres, radians and seconds instead.

Install with ``pip install 'torch-fis[kart]'``.  The first start downloads the game's
assets (about 700 MB) into the user cache directory.

``pystk2`` runs one game per process: it can be initialised only once, with or without
graphics, and only one race can exist at a time.  Creating a rendering environment after
a headless one in the same process therefore raises; start a new process instead.  A
second environment's ``reset()`` takes the game over from the first.
"""
import math
import os
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("SuperTuxKartEnv needs gymnasium; install with: pip install 'torch-fis[kart]'") from exc

#: Tracks that are closed circuits without branches, so the centre line is unambiguous.
#: Other race tracks from ``pystk2.list_tracks()`` work too; at a fork the features
#: follow the first branch.
TRACKS = ('lighthouse', 'zengarden', 'hacienda', 'cornfield_crossing', 'snowmountain',
          'scotland')

#: A kart off the road this many times ``rescue_after`` in a row is rescued too.
LOST_AFTER = 2.5

#: Speed and road half-width of the lecture circuits, which ``units='grand_prix'`` maps to.
_GP_TOP_SPEED, _GP_HALF_WIDTH = 4.0, 1.0

# pystk2 keeps one game per process.  None until initialised, then whether it has graphics.
_GAME = {'graphics': None, 'owner': None}


def _wrap(angle):
    """Wrap an angle to ``[-pi, pi)``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _pystk2(graphics: bool, width: int, height: int):
    """Import and initialise pystk2 once per process."""
    try:
        import pystk2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SuperTuxKartEnv needs pystk2; install with: "
                          "pip install 'torch-fis[kart]'") from exc
    if _GAME['graphics'] is None:
        if graphics:
            config = pystk2.GraphicsConfig.ld()
            config.screen_width, config.screen_height = width, height
            # Render off-screen; 'human' mode shows the frames in its own window.
            config.display = False
        else:
            config = pystk2.GraphicsConfig.none()
        # SDL's dummy video driver (set by headless pygame rendering) stops the game's
        # own off-screen renderer from starting; lift it while the game starts.
        dummy = os.environ.get('SDL_VIDEODRIVER') == 'dummy' and graphics
        if dummy:
            del os.environ['SDL_VIDEODRIVER']
        try:
            pystk2.init(config)
        finally:
            if dummy:
                os.environ['SDL_VIDEODRIVER'] = 'dummy'
        _GAME['graphics'] = graphics
    elif graphics and not _GAME['graphics']:
        raise RuntimeError("SuperTuxKart was already started without graphics in this process, "
                           "and pystk2 can only be started once. Create the rendering "
                           "environment first, or use a new process.")
    return pystk2


class StkTrack:
    """A SuperTuxKart centre line, projected onto the ground plane.

    pystk2 is y-up; the ground point ``(x, -z)`` makes a counter-clockwise turn a
    positive yaw, matching ``RacingEnv`` (and positive ``steer`` turning left).
    """

    def __init__(self, name: str, track):
        nodes = np.asarray(track.path_nodes, dtype=np.float64)
        self.name = name
        self.start = nodes[:, 0][:, [0, 2]] * [1.0, -1.0]
        self.delta = nodes[:, 1][:, [0, 2]] * [1.0, -1.0] - self.start
        self.lengths = np.maximum(np.linalg.norm(self.delta, axis=1), 1e-6)
        self.tangent = np.arctan2(self.delta[:, 1], self.delta[:, 0])
        self.half_width = np.asarray(track.path_width, dtype=np.float64).reshape(-1) / 2.0
        self.next = np.asarray([s[0] if len(s) else (i + 1) % len(nodes)
                                for i, s in enumerate(track.successors)], dtype=np.int64)
        self.prev = np.zeros_like(self.next)
        self.prev[self.next] = np.arange(len(nodes))
        self.length = float(track.length)
        #: The typical road half-width, which ``units='grand_prix'`` maps to 1.
        self.reference_half_width = float(np.median(self.half_width))

    def project(self, point: np.ndarray, node: int) -> Tuple[int, float, float]:
        """``(node, distance into it, signed cross-track error)`` near the kart's node.

        The kart's own node can lag a corner, so its neighbours are tried as well.
        """
        best = None
        for i in (int(self.prev[node]), int(node), int(self.next[node])):
            u = float(np.clip(np.dot(point - self.start[i], self.delta[i]) / self.lengths[i] ** 2,
                              0.0, 1.0))
            offset = point - (self.start[i] + u * self.delta[i])
            dist = float(np.dot(offset, offset))
            if best is None or dist < best[0]:
                # 2-D cross product: positive means left of the travel direction.
                cte = (self.delta[i, 0] * offset[1] - self.delta[i, 1] * offset[0]) / self.lengths[i]
                best = (dist, i, u * self.lengths[i], float(cte))
        return best[1:]

    def walk(self, node: int, along: float, distance: float) -> Tuple[int, float]:
        """Move ``distance`` metres (may be negative) along the first branch."""
        along += distance
        while along > self.lengths[node]:
            along -= self.lengths[node]
            node = int(self.next[node])
        while along < 0.0:
            node = int(self.prev[node])
            along += self.lengths[node]
        return node, along

    def heading(self, node: int, along: float) -> float:
        """Travel direction, blended across segment joints so it has no steps."""
        half = self.lengths[node] / 2.0
        if along >= half:
            other, weight = int(self.next[node]), (along - half) / (half + self.lengths[self.next[node]] / 2)
        else:
            other, weight = int(self.prev[node]), (half - along) / (half + self.lengths[self.prev[node]] / 2)
        return float(self.tangent[node] + weight * _wrap(self.tangent[other] - self.tangent[node]))

    def curvature(self, node: int, along: float, span: float) -> float:
        """Signed curvature (1/m, positive = left bend) by a central difference over ``2 span``."""
        before, after = self.walk(node, along, -span), self.walk(node, along, span)
        return _wrap(self.heading(*after) - self.heading(*before)) / (2.0 * span)



def _ground(kart) -> Tuple[np.ndarray, float]:
    """A kart's ground position ``(x, -z)`` and its yaw in that frame."""
    location, front = np.asarray(kart.location), np.asarray(kart.front)
    point = np.array([location[0], -location[2]])
    nose = np.array([front[0], -front[2]]) - point
    return point, math.atan2(nose[1], nose[0])


class _KartTracker:
    """Race progress, features and rescues of one player-controlled kart."""

    def __init__(self, track: StkTrack, laps: int, units: str, top_speed: float,
                 lookahead: float, rescue_after: Optional[float], dt: float):
        self.track, self.laps, self.units, self.lookahead = track, laps, units, lookahead
        self.rescue_after, self.dt = rescue_after, dt
        #: Metres per feature unit of length, and metres per second per unit of speed.
        self.L = track.reference_half_width / _GP_HALF_WIDTH if units == 'grand_prix' else 1.0
        self.V = top_speed / _GP_TOP_SPEED if units == 'grand_prix' else 1.0
        self.progress, self.rescues, self._stuck, self._off = 0.0, 0, 0, 0
        self._last = None

    def begin(self, kart):
        self._last = float(kart.distance_down_track)

    def command(self, stk, action: np.ndarray, kart, extra_actions: bool = False):
        """The game action for ``[throttle, steering, (drift, nitro, fire)]``."""
        throttle, steer = float(action[0]), float(action[1])
        command = stk.Action(acceleration=max(0.0, throttle), brake=throttle < -0.1, steer=steer)
        if extra_actions:
            command.drift, command.nitro, command.fire = (bool(a > 0) for a in action[2:5])
        self._stuck = self._stuck + 1 if abs(float(kart.speed)) < 0.5 else 0
        # A kart knocked off the road can wander the grass at speed and never stall.
        lost = self._off * self.dt >= LOST_AFTER * self.rescue_after if self.rescue_after else False
        if self.rescue_after is not None and (self._stuck * self.dt >= self.rescue_after or lost):
            command.rescue, self._stuck, self._off = True, 0, 0
            self.rescues += 1
        return command

    def advance(self, kart) -> float:
        """Metres gained since the last call."""
        distance = float(kart.distance_down_track)
        delta = distance - self._last
        # Crossing the line wraps distance_down_track back to 0; a rescue jumps a little.
        if delta < -self.track.length / 2:
            delta += self.track.length
        elif delta > self.track.length / 2:
            delta -= self.track.length
        self._last = distance
        self.progress += delta
        return delta

    def finished(self, kart) -> bool:
        return bool(kart.has_finished_race) or kart.finished_laps >= self.laps

    def features(self, kart) -> Tuple[np.ndarray, Dict]:
        point, yaw = _ground(kart)
        node, along, cte = self.track.project(point, int(kart.node))
        L, V = self.L, self.V
        ahead = self.track.walk(node, along, self.lookahead * L)
        curvature = self.track.curvature(*ahead, span=L)
        # Forward speed: the velocity's component along the kart's nose.
        velocity = np.asarray(kart.velocity)
        forward = float(velocity[0] * math.cos(yaw) - velocity[2] * math.sin(yaw))
        yaw_rate = float(np.asarray(kart.angular_velocity)[1])
        half_width = self.track.half_width[node]
        # In grand-prix units the road edge is +-1 wherever the road is wider or narrower.
        cte_feature = cte / half_width * _GP_HALF_WIDTH if self.units == 'grand_prix' else cte
        obs = np.array([cte_feature, _wrap(self.track.heading(node, along) - yaw),
                        forward / V, yaw_rate * L / V, curvature * L,
                        np.clip(self.progress / (self.laps * self.track.length), 0.0, 1.05)],
                       np.float32)
        off_track = abs(cte) > half_width or not kart.is_on_road
        self._off = self._off + 1 if off_track else 0
        info = {'cross_track_error_m': cte, 'off_track': off_track, 'speed_mps': forward}
        return obs, info


class _Game:
    """Owns the process's single pystk2 race; a new race takes the game over."""

    def __init__(self, stk):
        self.stk, self.race, self.world = stk, None, None

    def start(self, config):
        owner = _GAME['owner']
        if owner is not None and owner is not self:
            owner.stop()
        self.stop()
        self.race = self.stk.Race(config)
        self.race.start()
        self.world = self.stk.WorldState()
        _GAME['owner'] = self
        geometry = self.stk.Track()
        geometry.update()
        # Skip the track fly-by and the countdown: racing begins at the start signal.
        phase = self.stk.WorldState.Phase
        idle = [self.stk.Action() for _ in config.players]
        self.race.step(idle)
        self.world.update()
        while self.world.phase not in (phase.GO_PHASE, phase.RACE_PHASE):
            self.race.step(idle)
            self.world.update()
        return geometry

    def stop(self):
        if self.race is not None:
            self.race.stop()
            self.race = self.world = None
            if _GAME['owner'] is self:
                _GAME['owner'] = None

    def kart(self, player: int):
        """The kart of player ``player``. ``world.karts`` lists AI karts first."""
        return self.world.players[player].kart


class SuperTuxKartEnv(gym.Env):
    """A SuperTuxKart race against the clock or AI karts, for fuzzy drivers.

    Observation order is ``[cross_track_error, heading_error, speed, yaw_rate,
    lookahead_curvature, progress]``, as in ``RacingEnv``; ``progress`` is the fraction
    of the whole race (all laps).  The action is ``[throttle, steering]`` in ``[-1, 1]``:
    negative throttle brakes, positive steering turns left.  With ``extra_actions=True``
    three more entries press ``drift``, ``nitro`` and ``fire`` (use the held item) when
    positive.

    The episode ends when the kart finishes the race, or is truncated at ``max_time``.
    Driving off the road is allowed, as in the game (grass is just slow); ``info``
    reports it, and ``terminate_off_track=True`` ends the episode there instead, like
    the lecture time trial.  A kart stuck for ``rescue_after`` seconds, or off the
    road for 2.5 times that, is put back on the road by the game's rescue bird; set
    it to ``None`` to disable that.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}
    feature_names = ('cross_track_error', 'heading_error', 'speed', 'yaw_rate',
                     'lookahead_curvature', 'progress')

    def __init__(self, track: str = 'lighthouse', *, laps: int = 1, opponents: int = 0,
                 difficulty: int = 1, kart: str = 'tux', step_size: float = 0.05,
                 max_time: float = 180.0, lookahead: float = 1.8, units: str = 'grand_prix',
                 top_speed: float = 22.0, extra_actions: bool = False,
                 terminate_off_track: bool = False, rescue_after: Optional[float] = 2.0,
                 render_mode: Optional[str] = None, width: int = 640, height: int = 400):
        super().__init__()
        if render_mode not in (None, 'human', 'rgb_array'):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")
        if units not in ('grand_prix', 'si'):
            raise ValueError("units must be 'grand_prix' or 'si'")
        if laps < 1 or opponents < 0 or difficulty not in (0, 1, 2):
            raise ValueError("need laps >= 1, opponents >= 0 and difficulty in 0, 1, 2")
        self.render_mode, self.units = render_mode, units
        self.track_name, self.laps, self.opponents = track, int(laps), int(opponents)
        self.difficulty, self.kart = int(difficulty), kart
        self.dt, self.max_time, self.lookahead = float(step_size), float(max_time), float(lookahead)
        self.top_speed, self.extra_actions = float(top_speed), bool(extra_actions)
        self.terminate_off_track, self.rescue_after = bool(terminate_off_track), rescue_after
        self._stk = _pystk2(render_mode is not None, width, height)
        _check_names(self._stk, track, [kart])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5 if extra_actions else 2,), dtype=np.float32)
        # Bounds are generous: the kart may leave the road, jump, or spin.
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -math.pi, -50.0, -50.0, -50.0, 0.0], np.float32),
            high=np.array([50.0, math.pi, 50.0, 50.0, 50.0, 1.05], np.float32), dtype=np.float32)
        self.track: Optional[StkTrack] = None
        self._game = _Game(self._stk)
        self._tracker: Optional[_KartTracker] = None
        self._renderer = None
        self._step_index = 0

    @property
    def time(self) -> float:
        """Race time in seconds since the start signal."""
        return self._step_index * self.dt

    @property
    def rescues(self) -> int:
        return self._tracker.rescues if self._tracker else 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        stk = self._stk
        seed = int(self.np_random.integers(0, 2 ** 31 - 1)) if seed is None else int(seed)
        config = stk.RaceConfig(track=self.track_name, laps=self.laps, seed=seed,
                                num_kart=1 + self.opponents, difficulty=self.difficulty,
                                step_size=self.dt)
        config.players[0].kart = self.kart
        config.players[0].controller = stk.PlayerConfig.Controller.PLAYER_CONTROL
        if self.render_mode is not None:
            config.players[0].camera_mode = stk.PlayerConfig.CameraMode.ON
        self.track = StkTrack(self.track_name, self._game.start(config))
        self._tracker = _KartTracker(self.track, self.laps, self.units, self.top_speed,
                                     self.lookahead, self.rescue_after, self.dt)
        self._tracker.begin(self._game.kart(0))
        self._step_index = 0
        obs, _ = self._tracker.features(self._game.kart(0))
        if self.render_mode == 'human':
            self.render()
        return obs, {'track': self.track_name, 'lap_time': None, 'race_time': None,
                     'num_karts': 1 + self.opponents}

    def step(self, action):
        if self._game.race is None:
            raise RuntimeError("step() called before reset(), or another SuperTuxKart race "
                               "took over the game; call reset()")
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}")
        tracker = self._tracker
        self._game.race.step(tracker.command(self._stk, action, self._game.kart(0),
                                             self.extra_actions))
        self._game.world.update()
        self._step_index += 1

        kart = self._game.kart(0)
        delta = tracker.advance(kart)
        obs, extra = tracker.features(kart)
        finished = tracker.finished(kart)
        off_track = extra['off_track']
        timeout = self.time >= self.max_time
        # Same shaping as RacingEnv, in feature units, so rewards are comparable.
        reward = 8.0 * delta / tracker.L - 0.02 * abs(float(obs[0])) - 0.002 * float(action[1]) ** 2
        if off_track and self.terminate_off_track:
            reward -= 25.0
        race_time = self.time if finished else None
        info = {'track': self.track_name, 'lap_time': race_time, 'race_time': race_time,
                'progress': max(0.0, tracker.progress), 'cross_track_error': float(obs[0]),
                'off_track': off_track, 'speed': float(obs[2]), 'position': int(kart.position),
                'num_karts': 1 + self.opponents, 'lap': max(0, int(kart.finished_laps)) + (not finished),
                'rescues': tracker.rescues, 'xy': _ground(kart)[0].tolist(), **extra}
        if self.render_mode == 'human':
            self.render()
        terminated = finished or (off_track and self.terminate_off_track)
        return obs, float(reward), bool(terminated), bool(timeout and not terminated), info

    def render(self):
        """The chase camera as an ``(H, W, 3)`` image, or a live window in ``'human'`` mode."""
        if self.render_mode is None:
            return None
        if self._game.race is None:
            raise RuntimeError("render() called before reset()")
        frame = np.asarray(self._game.race.render_data[0].image)
        if self.render_mode == 'rgb_array':
            return frame
        if self._renderer is None:
            self._renderer = Window(frame.shape[1], frame.shape[0],
                                    f'Fuzzy SuperTuxKart — {self.track_name}')
            self._wall = time.perf_counter()
        self._renderer.show(frame)
        # Real-time pacing: one step of game time per step of wall time.
        self._wall += self.dt
        delay = self._wall - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self._wall = time.perf_counter()
        return None

    @property
    def is_running(self) -> bool:
        return self._renderer is None or self._renderer.running

    def close(self):
        self._game.stop()
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _check_names(stk, track: str, karts):
    if track not in stk.list_tracks():
        raise ValueError(f"unknown SuperTuxKart track {track!r}; e.g. {', '.join(TRACKS)}")
    for kart in karts:
        if kart not in stk.list_karts():
            raise ValueError(f"unknown kart {kart!r}; choose from {', '.join(stk.list_karts())}")


class KartRace:
    """One SuperTuxKart race with a fuzzy driver in every player kart.

    Built for a class: each student's driver gets its own kart (and, with ``render``,
    its own chase camera), optionally joined by AI karts, all in one race::

        race = KartRace({'Ana': ConfigRaceDriver(ana_json), 'Bruno': FuzzyRaceDriver()},
                        track='zengarden', ai_karts=2)
        race.reset(seed=0)
        while not race.step():
            pass
        for row in race.standings():
            print(row['position'], row['name'], row['race_time'])

    With ``render=True`` the first :attr:`max_cameras` drivers each get a chase camera;
    SuperTuxKart's split screen does not render more reliably, so race a large class
    in heats of up to that many.

    A driver is any ``action = driver(observation)`` callable with ``SuperTuxKartEnv``'s
    observation and ``[throttle, steering]`` action (including a student's laptop, see
    :mod:`~fuzzy_torch.envs.kart_net`).  A driver that raises or returns a non-finite
    action coasts for that step, and ``errors`` holds the error until it acts normally
    again, so one broken submission cannot stop the race.
    """
    #: SuperTuxKart renders up to this many player cameras per race.
    max_cameras = 8

    def __init__(self, drivers: Dict[str, Callable], track: str = 'lighthouse', *,
                 laps: int = 1, ai_karts: int = 0, difficulty: int = 1,
                 step_size: float = 0.05, max_time: float = 240.0, lookahead: float = 1.8,
                 units: str = 'grand_prix', top_speed: float = 22.0,
                 rescue_after: Optional[float] = 2.0, render: bool = False,
                 width: int = 800, height: int = 500):
        if not drivers:
            raise ValueError("need at least one driver")
        if units not in ('grand_prix', 'si'):
            raise ValueError("units must be 'grand_prix' or 'si'")
        self.drivers = dict(drivers)
        self.names = list(self.drivers)
        self.track_name, self.laps, self.ai_karts = track, int(laps), int(ai_karts)
        self.difficulty, self.dt, self.max_time = int(difficulty), float(step_size), float(max_time)
        self.lookahead, self.units, self.top_speed = float(lookahead), units, float(top_speed)
        self.rescue_after, self.render = rescue_after, bool(render)
        self._stk = _pystk2(self.render, width, height)
        _check_names(self._stk, track, [])
        karts = self._stk.list_karts()
        #: Each player's kart model, so karts on screen tell students apart.
        self.karts = {name: karts[i % len(karts)] for i, name in enumerate(self.names)}
        #: The drivers with a chase camera.
        self.cameras = self.names[:self.max_cameras] if self.render else []
        self._game = _Game(self._stk)
        self.track: Optional[StkTrack] = None
        self.observations: Dict[str, np.ndarray] = {}
        self.actions: Dict[str, np.ndarray] = {}
        self.errors: Dict[str, str] = {}
        self.finish_times: Dict[str, float] = {}
        self._step_index = 0

    @property
    def time(self) -> float:
        """Race time in seconds since the start signal."""
        return self._step_index * self.dt

    def reset(self, seed: int = 0):
        stk = self._stk
        config = stk.RaceConfig(track=self.track_name, laps=self.laps, seed=int(seed),
                                num_kart=len(self.names) + self.ai_karts,
                                difficulty=self.difficulty, step_size=self.dt,
                                # The game's HUD is laid out for split screen and
                                # covers the picture; RaceDirector draws its own.
                                overlay=False)
        mode = stk.PlayerConfig.CameraMode
        players = [stk.PlayerConfig(kart=self.karts[name], name=name[:20],
                                    camera_mode=mode.ON if name in self.cameras else mode.OFF,
                                    controller=stk.PlayerConfig.Controller.PLAYER_CONTROL)
                   for name in self.names]
        config.players[0] = players[0]
        for player in players[1:]:
            config.players.append(player)
        self.track = StkTrack(self.track_name, self._game.start(config))
        self._trackers = [_KartTracker(self.track, self.laps, self.units, self.top_speed,
                                       self.lookahead, self.rescue_after, self.dt)
                          for _ in self.names]
        self._step_index = 0
        self.errors.clear()
        self.finish_times.clear()
        for i, name in enumerate(self.names):
            if hasattr(self.drivers[name], 'reset'):
                self.drivers[name].reset()
            kart = self._game.kart(i)
            self._trackers[i].begin(kart)
            self.observations[name], _ = self._trackers[i].features(kart)
            self.actions[name] = np.zeros(2, np.float32)
        return dict(self.observations)

    def _act(self, name: str) -> np.ndarray:
        try:
            action = np.asarray(self.drivers[name](self.observations[name]), np.float32).reshape(-1)[:2]
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError(f"non-finite or malformed action {action.tolist()}")
            # `errors` holds current trouble only: a laptop that lagged a moment recovers.
            self.errors.pop(name, None)
            return np.clip(action, -1.0, 1.0)
        except Exception as exc:                      # noqa: BLE001 -- a student's driver
            self.errors[name] = f"t={self.time:.1f}s: {exc}"
            return np.zeros(2, np.float32)

    def step(self) -> bool:
        """Advance the race by one step; True once every driver finished or time is up."""
        if self._game.race is None:
            raise RuntimeError("step() called before reset(), or another race took over the game")
        commands = []
        for i, name in enumerate(self.names):
            kart = self._game.kart(i)
            if name in self.finish_times:
                commands.append(self._stk.Action())
                continue
            self.actions[name] = self._act(name)
            commands.append(self._trackers[i].command(self._stk, self.actions[name], kart))
        self._game.race.step(commands)
        self._game.world.update()
        self._step_index += 1
        for i, name in enumerate(self.names):
            if name in self.finish_times:
                continue
            kart, tracker = self._game.kart(i), self._trackers[i]
            tracker.advance(kart)
            self.observations[name], _ = tracker.features(kart)
            if tracker.finished(kart):
                self.finish_times[name] = self.time
        return len(self.finish_times) == len(self.names) or self.time >= self.max_time

    def standings(self) -> list:
        """Every kart, leader first: ``name``, ``position``, ``lap``, ``progress``
        (fraction of the race), ``race_time`` (None until finished), ``ai``, ``error``."""
        rows = []
        players = {self._game.kart(i).id: (i, name) for i, name in enumerate(self.names)}
        total = self.laps * self.track.length
        for kart in self._game.world.karts:
            if kart.id in players:
                i, name = players[kart.id]
                progress = self._trackers[i].progress / total
                race_time = self.finish_times.get(name)
                ai, error = False, self.errors.get(name)
            else:
                name, ai, error = f"AI {kart.name}", True, None
                progress = max(0.0, float(kart.overall_distance)) / total
                race_time = float(kart.finish_time) if kart.has_finished_race else None
            rows.append({'name': name, 'position': int(kart.position), 'ai': ai,
                         'lap': min(self.laps, max(0, int(kart.finished_laps)) + 1),
                         'progress': float(np.clip(progress, 0.0, 1.0)),
                         'race_time': race_time, 'error': error, 'ground': _ground(kart)})
        return sorted(rows, key=lambda row: row['position'])

    def frame(self, name: str) -> Optional[np.ndarray]:
        """The chase camera behind ``name``'s kart, or None if it has no camera."""
        if not self.render:
            raise RuntimeError("create the race with render=True to get camera frames")
        if name not in self.cameras:
            return None
        return np.asarray(self._game.race.render_data[self.cameras.index(name)].image)

    def close(self):
        self._game.stop()


class Window:
    """A pygame window in its own process, showing frames drawn in this one.

    SuperTuxKart and pygame each bundle SDL, and the two cannot open displays in one
    process (whichever starts second crashes).  So the window lives in a child
    process: frames go to it through shared memory, and key presses and clicks come
    back from :meth:`show` as ``('key', name)``, ``('click', (x, y))`` or ``('quit',)``.
    """

    #: Keys reported by name; others are ignored.
    keys = ('left', 'right', 'tab', 'l', 'space', 'f', 'return', 'escape')

    def __init__(self, width: int, height: int, title: str):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        self.size, self.running = (int(width), int(height)), True
        self._shm = shared_memory.SharedMemory(create=True, size=self.size[0] * self.size[1] * 3)
        self._buffer = np.ndarray((self.size[1], self.size[0], 3), np.uint8, buffer=self._shm.buf)
        # 'spawn' starts a clean interpreter: a forked child would inherit the game.
        ctx = mp.get_context('spawn')
        self._conn, child = ctx.Pipe()
        self._process = ctx.Process(target=_window_main, args=(child, self._shm.name, self.size, title),
                                    daemon=True)
        self._process.start()
        self._waiting = False

    def show(self, frame: np.ndarray) -> list:
        """Display ``frame`` (``(H, W, 3)`` uint8); returns the input events since the last call."""
        events = self._collect(block=True)
        if self.running:
            self._buffer[:] = frame
            try:
                self._conn.send('frame')
                self._waiting = True
            except (BrokenPipeError, OSError):
                self.running = False
        return events

    def _collect(self, block: bool) -> list:
        """Wait for the window to finish the previous frame and take its events."""
        if not self._waiting:
            return []
        try:
            if not self._conn.poll(5.0 if block else 0.0):
                return []
            events = self._conn.recv()
        except (EOFError, OSError):
            events = [('quit',)]
        self._waiting = False
        if ('quit',) in events:
            self.running = False
        return events

    def set_title(self, title: str):
        if self.running:
            self._collect(block=True)
            self._conn.send(('title', title))

    def close(self):
        try:
            self._conn.send('close')
        except (BrokenPipeError, OSError):
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
        self._shm.close()
        self._shm.unlink()


def _window_main(conn, shm_name: str, size: Tuple[int, int], title: str):
    """The window process: show frames as they arrive, report input back."""
    import pygame
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((size[1], size[0], 3), np.uint8, buffer=shm.buf)
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(title)
    names = {getattr(pygame, f'K_{k.upper() if len(k) > 1 else k}'): k for k in Window.keys}
    names[pygame.K_KP_ENTER] = 'return'
    try:
        while True:
            message = conn.recv()
            if message == 'close':
                break
            if isinstance(message, tuple) and message[0] == 'title':
                pygame.display.set_caption(message[1])
                continue
            pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))
            pygame.display.flip()
            events = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    events.append(('quit',))
                elif event.type == pygame.KEYDOWN and event.key in names:
                    events.append(('key', names[event.key]))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    events.append(('click', event.pos))
            conn.send(events)
    except (EOFError, OSError):
        pass
    finally:
        del frame
        shm.close()
        pygame.quit()
