"""Students drive SuperTuxKart karts from their own laptops.

The race runs on one machine (the instructor's); each student's laptop runs its own
controller.  :class:`KartHub` accepts laptops over WebSockets, and gives each one a
:class:`RemoteDriver`: an ordinary ``action = driver(observation)`` callable for
:class:`~fuzzy_torch.envs.supertuxkart.KartRace`, so remote and local drivers race
together.  On the laptop, :func:`join` connects a driver to the hub.

WebSockets travel over plain HTTP, so the hub works on a classroom network and
through an HTTP tunnel such as ``ngrok http 9000`` (whose free plan does not offer
raw TCP).  Laptops must give the hub's **join code**, which the instructor shows
on screen, so finding the address is not enough to join.

Each WebSocket message is one JSON object:

* laptop -> hub: ``{"type": "hello", "name": ..., "code": ..., "spec": {...} | null}``
  once, then ``{"type": "action", "step": k, "action": [throttle, steering]}`` per
  observation, and ``{"type": "spec", "spec": ...}`` when it reloads its driver file.
* hub -> laptop: ``welcome``, then for each race ``start`` (or ``wait`` when the
  student is not in that heat), an ``obs`` per step, and ``end`` with the standings.

The hub never waits for a laptop: each step it sends the observation and uses the
newest action received so far, so a remote kart steers at least one step (50 ms)
late, plus the network round trip.  A laptop that has not answered for ``timeout``
seconds coasts until it answers again, so a slow or disconnected laptop cannot hold
up the race.

Remote code runs on the student's machine, so the hub cannot check that it is a
fuzzy controller; races that must be fair should use JSON drivers instead.
"""
import json
import secrets
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

#: Bumped when the protocol changes, so old clients get a clear error.
PROTOCOL = 2
MAX_MESSAGE = 256 * 1024
MAX_NAME = 24
#: Join codes avoid look-alike characters (0/O, 1/I/L), since they are read off a screen.
CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'


def _websockets():
    try:
        import websockets  # noqa: F401
        from websockets.sync import client, server
    except ImportError as exc:  # pragma: no cover
        raise ImportError("racing from laptops needs websockets; install with: "
                          "pip install 'torch-fis[envs]'") from exc
    return client, server


def new_code(length: int = 5) -> str:
    return ''.join(secrets.choice(CODE_ALPHABET) for _ in range(length))


def ws_url(server: str) -> str:
    """The WebSocket URL for what a student types: ``host:port``, an ngrok
    ``https://...`` address, or a ``ws://``/``wss://`` URL."""
    server = server.strip().rstrip('/')
    for http, ws in (('https://', 'wss://'), ('http://', 'ws://')):
        if server.startswith(http):
            return ws + server[len(http):]
    if server.startswith(('ws://', 'wss://')):
        return server
    return 'ws://' + server


def _no_delay(connection):
    """Send small messages at once rather than batching them (Nagle's algorithm)."""
    try:
        connection.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except (AttributeError, OSError):
        pass


def _messages(connection):
    """Yield the JSON objects arriving on ``connection`` until it closes."""
    while True:
        try:
            raw = connection.recv()
        except Exception:                             # noqa: BLE001 -- closed or broken
            return
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        if isinstance(message, dict):
            yield message


class RemoteDriver:
    """A driver on a student's laptop, as seen by :class:`KartRace`.

    Calling it sends the observation and returns the newest action the laptop has
    sent.  ``steering`` and ``throttle`` are the laptop's fuzzy systems when it
    shared a JSON spec, so :class:`~fuzzy_torch.envs.kart_view.RaceDirector` can draw
    them; they are only drawn, never used to drive.
    """

    def __init__(self, hub: 'KartHub', name: str, connection, timeout: float):
        self.hub, self.name, self.timeout = hub, name, timeout
        self._connection, self._lock = connection, threading.Lock()
        self.connected = True
        self.steering = self.throttle = None
        self._action = np.zeros(2, np.float32)
        self._heard = time.monotonic()
        self._step = 0

    def set_spec(self, spec):
        from .controllers.racing import ConfigRaceDriver
        try:
            driver = ConfigRaceDriver(spec, max_rules=125, max_inputs=3, max_terms=7)
            self.steering, self.throttle = driver.steering, driver.throttle
        except Exception:                             # noqa: BLE001 -- display only
            self.steering = self.throttle = None

    def send(self, message: dict) -> bool:
        if not self.connected:
            return False
        try:
            with self._lock:
                self._connection.send(json.dumps(message, separators=(',', ':')))
        except Exception:                             # noqa: BLE001 -- closed or broken
            self.connected = False
        return self.connected

    def reset(self):
        self._action = np.zeros(2, np.float32)
        self._heard = time.monotonic()
        self._step = 0

    def _receive(self, step: int, action):
        try:
            action = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        except (TypeError, ValueError):
            return
        if np.isfinite(action).all() and step <= self._step:
            self._action, self._heard = action, time.monotonic()

    def __call__(self, obs) -> np.ndarray:
        self._step += 1
        self.send({'type': 'obs', 'step': self._step, 'obs': [round(float(v), 5) for v in obs],
                   **self.hub.status.get(self.name, {})})
        if not self.connected:
            raise ConnectionError('laptop disconnected')
        if time.monotonic() - self._heard > self.timeout:
            raise TimeoutError(f'no answer from the laptop for {self.timeout:.0f} s')
        return self._action.copy()


class KartHub:
    """Accepts students' laptops; one :class:`RemoteDriver` per connected name.

    ``host='0.0.0.0'`` accepts laptops on the local network; behind a tunnel
    (``ngrok http PORT``) the default ``127.0.0.1`` is enough.  ``code`` is the join
    code laptops must give; by default a random one, in :attr:`code`.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9000, *, code: Optional[str] = None,
                 timeout: float = 1.0):
        _, server = _websockets()
        self.timeout = timeout
        self.code = (code or new_code()).upper()
        self.drivers: Dict[str, RemoteDriver] = {}
        #: Live race facts per name (position, lap, ...), added to each ``obs`` message.
        self.status: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._server = server.serve(self._serve, host, port, max_size=MAX_MESSAGE,
                                    compression=None)
        self.address = self._server.socket.getsockname()[:2]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def names(self) -> List[str]:
        """Connected students, in the order they joined."""
        with self._lock:
            return [n for n, d in self.drivers.items() if d.connected]

    def _refuse(self, connection, error: str):
        try:
            connection.send(json.dumps({'type': 'error', 'error': error}))
            connection.close()
        except Exception:                             # noqa: BLE001
            pass

    def _serve(self, connection):
        _no_delay(connection)
        try:
            hello = json.loads(connection.recv(timeout=10))
        except Exception:                             # noqa: BLE001 -- silent or not JSON
            hello = None
        error = self._check_hello(hello)
        if error:
            self._refuse(connection, error)
            return
        name = ' '.join(hello['name'].split())
        driver = RemoteDriver(self, name, connection, self.timeout)
        if hello.get('spec') is not None:
            driver.set_spec(hello['spec'])
        with self._lock:
            # Names match ignoring case and spacing, as on the Race Control leaderboard.
            same = [n for n in self.drivers if n.casefold() == name.casefold()]
            if any(self.drivers[n].connected for n in same):
                taken = f'{name!r} is already connected'
            else:
                taken = None
                for n in same:
                    del self.drivers[n]               # a student reconnecting
                self.drivers[name] = driver
        if taken:
            self._refuse(connection, taken)
            return
        driver.send({'type': 'welcome', 'name': name, 'protocol': PROTOCOL})
        for message in _messages(connection):
            kind = message.get('type')
            if kind == 'action':
                try:
                    driver._receive(int(message.get('step', 0)), message.get('action'))
                except (TypeError, ValueError):
                    pass
            elif kind == 'spec':
                driver.set_spec(message.get('spec'))
        driver.connected = False

    def _check_hello(self, hello) -> Optional[str]:
        if not isinstance(hello, dict) or hello.get('type') != 'hello':
            return 'expected a hello message'
        if hello.get('protocol') != PROTOCOL:
            return f'this race uses protocol {PROTOCOL}; update fuzzy_torch on your laptop'
        code = str(hello.get('code') or '').strip().upper()
        if not secrets.compare_digest(code.encode(), self.code.encode()):
            return 'wrong join code: use the code shown on the race screen'
        name = hello.get('name')
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_NAME:
            return f'please give a name of 1 to {MAX_NAME} characters'
        return None

    def announce(self, title: str, track: str, entrants: List[str]):
        """Tell entrants their race is starting and everyone else to wait."""
        for name, driver in list(self.drivers.items()):
            if name in entrants:
                driver.send({'type': 'start', 'race': title, 'track': track})
            else:
                driver.send({'type': 'wait', 'race': title, 'entrants': entrants})

    def results(self, title: str, standings: List[dict]):
        rows = [{k: r[k] for k in ('position', 'name', 'race_time', 'progress', 'ai')} for r in standings]
        for driver in list(self.drivers.values()):
            driver.send({'type': 'end', 'race': title, 'standings': rows})

    def track_status(self, race):
        """Copy each remote kart's position and lap into :attr:`status`."""
        self.status = {r['name']: {'position': r['position'], 'lap': r['lap'],
                                   'karts': race.ai_karts + len(race.names), 'time': round(race.time, 2)}
                       for r in race.standings() if r['name'] in self.drivers}

    def close(self):
        self._server.shutdown()
        for driver in list(self.drivers.values()):
            driver.connected = False


class Join:
    """A laptop's connection to a :class:`KartHub` (see :func:`join`)."""

    def __init__(self, server: str, name: str, code: str, driver: Callable,
                 spec: Optional[dict] = None, reload: Optional[Callable[[], tuple]] = None):
        client, _ = _websockets()
        self.url = ws_url(server)
        # ngrok's free plan shows browsers a warning page first; this header skips it.
        # Entered by hand: the connection outlives this method, and closing it is
        # close()'s job.  (websockets 17 warns when connect() is used without `with`.)
        self.connection = client.connect(self.url, open_timeout=15, max_size=MAX_MESSAGE,
                                         compression=None,
                                         additional_headers={'ngrok-skip-browser-warning': '1'}
                                         ).__enter__()
        _no_delay(self.connection)
        self._lock = threading.Lock()
        self.driver, self.reload = driver, reload
        self._send({'type': 'hello', 'name': name, 'code': code, 'spec': spec, 'protocol': PROTOCOL})
        self.messages = _messages(self.connection)
        welcome = next(self.messages, None)
        if not welcome or welcome.get('type') != 'welcome':
            self.close()
            raise ConnectionError((welcome or {}).get('error', 'the race server closed the connection'))
        self.name = welcome['name']

    def _send(self, message: dict):
        try:
            with self._lock:
                self.connection.send(json.dumps(message, separators=(',', ':')))
        except Exception:                             # noqa: BLE001 -- the run loop notices
            pass

    def run(self, on_message: Optional[Callable[[dict, Optional[np.ndarray]], bool]] = None):
        """Answer observations until the server closes; ``on_message(message, action)``
        sees every message (with the action sent, for ``obs``) and may return False to stop."""
        for message in self.messages:
            action = None
            if message.get('type') == 'start' and self.reload is not None:
                self.driver, spec = self.reload()
                self._send({'type': 'spec', 'spec': spec})
                if hasattr(self.driver, 'reset'):
                    self.driver.reset()
            elif message.get('type') == 'obs':
                obs = np.asarray(message['obs'], np.float32)
                try:
                    action = np.asarray(self.driver(obs), np.float32).reshape(-1)[:2]
                except Exception as exc:              # noqa: BLE001 -- keep driving
                    print(f"your driver raised: {exc}")
                    action = np.zeros(2, np.float32)
                self._send({'type': 'action', 'step': message['step'],
                            'action': [float(a) for a in action]})
            if on_message is not None and on_message(message, action) is False:
                break
        self.close()

    def close(self):
        try:
            self.connection.close()
        except Exception:                             # noqa: BLE001
            pass


def join(server: str, name: str, code: str, driver: Callable, spec: Optional[dict] = None,
         **kwargs) -> Join:
    """Connect ``driver`` (``action = driver(observation)``) to a race.

    ``server`` is what the instructor shows: ``host:port`` on a local network, or the
    tunnel's ``https://...`` address.  ``code`` is the race's join code.  Pass the
    driver's JSON ``spec`` to have its fuzzy systems drawn on the race screen.  Returns
    the connection; call ``.run()`` on it to drive.
    """
    return Join(server, name, code, driver, spec, **kwargs)
