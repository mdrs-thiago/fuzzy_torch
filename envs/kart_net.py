"""Students drive SuperTuxKart karts from their own laptops.

The race runs on one machine (the projector's); each student's laptop runs its own
controller.  :class:`KartHub` accepts laptops over the network, and gives each one a
:class:`RemoteDriver`: an ordinary ``action = driver(observation)`` callable for
:class:`~fuzzy_torch.envs.supertuxkart.KartRace`, so remote and local drivers race
together.  On the laptop, :func:`join` connects a driver to the hub.

The protocol is newline-delimited JSON over TCP, one message per line:

* laptop -> hub: ``{"type": "hello", "name": ..., "spec": {...} | null}`` once, then
  ``{"type": "action", "step": k, "action": [throttle, steering]}`` per observation,
  and ``{"type": "spec", "spec": ...}`` when the laptop reloads its driver file.
* hub -> laptop: ``welcome``, then for each race ``start`` (or ``wait`` when the
  student is not in that heat), an ``obs`` per step, and ``end`` with the standings.

The hub never waits for a laptop: each step it sends the observation and uses the
newest action received so far, so a remote kart steers one step (50 ms) late.  A
laptop that has not answered for ``timeout`` seconds coasts until it answers again,
so a slow or disconnected laptop cannot hold up the race.

Remote code runs on the student's machine, so the hub cannot check that it is a
fuzzy controller; races that must be fair should use JSON drivers instead.  Only
share the hub on a network you trust (a classroom LAN): anyone who can reach the
port can join.
"""
import json
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

#: Bumped when the protocol changes, so old clients get a clear error.
PROTOCOL = 1
MAX_LINE = 256 * 1024
MAX_NAME = 24


def _send(sock: socket.socket, message: dict, lock: threading.Lock) -> bool:
    data = (json.dumps(message, separators=(',', ':')) + '\n').encode()
    try:
        with lock:
            sock.sendall(data)
        return True
    except OSError:
        return False


def _lines(sock: socket.socket):
    """Yield the JSON messages arriving on ``sock`` until it closes."""
    buffer = b''
    while True:
        try:
            chunk = sock.recv(65536)
        except OSError:
            return
        if not chunk:
            return
        buffer += chunk
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            if line.strip():
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    yield message
        if len(buffer) > MAX_LINE:
            return


class RemoteDriver:
    """A driver on a student's laptop, as seen by :class:`KartRace`.

    Calling it sends the observation and returns the newest action the laptop has
    sent.  ``steering`` and ``throttle`` are the laptop's fuzzy systems when it
    shared a JSON spec, so :class:`~fuzzy_torch.envs.kart_view.RaceDirector` can draw
    them; they are only drawn, never used to drive.
    """

    def __init__(self, hub: 'KartHub', name: str, sock: socket.socket, lock: threading.Lock,
                 timeout: float):
        self.hub, self.name, self.timeout = hub, name, timeout
        self._sock, self._lock = sock, lock
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
        if not _send(self._sock, message, self._lock):
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

    Use ``host='0.0.0.0'`` to accept laptops on the classroom network.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 9000, *, timeout: float = 1.0):
        self.timeout = timeout
        self.drivers: Dict[str, RemoteDriver] = {}
        #: Live race facts per name (position, lap, ...), added to each ``obs`` message.
        self.status: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._server = socket.create_server((host, port), reuse_port=False)
        self.address = self._server.getsockname()[:2]
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def names(self) -> List[str]:
        """Connected students, in the order they joined."""
        with self._lock:
            return [n for n, d in self.drivers.items() if d.connected]

    def _accept(self):
        while not self._closed:
            try:
                sock, _ = self._server.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(sock,), daemon=True).start()

    def _serve(self, sock: socket.socket):
        messages = _lines(sock)
        lock = threading.Lock()
        sock.settimeout(10.0)
        hello = next(messages, None)
        sock.settimeout(None)
        error = self._check_hello(hello)
        if error:
            _send(sock, {'type': 'error', 'error': error}, lock)
            sock.close()
            return
        name = ' '.join(hello['name'].split())
        driver = RemoteDriver(self, name, sock, lock, self.timeout)
        if hello.get('spec') is not None:
            driver.set_spec(hello['spec'])
        with self._lock:
            # Names match ignoring case and spacing, as on the Race Control leaderboard.
            same = [n for n in self.drivers if n.casefold() == name.casefold()]
            if any(self.drivers[n].connected for n in same):
                _send(sock, {'type': 'error', 'error': f'{name!r} is already connected'}, lock)
                sock.close()
                return
            for n in same:
                del self.drivers[n]               # a student reconnecting
            self.drivers[name] = driver
        driver.send({'type': 'welcome', 'name': name, 'protocol': PROTOCOL})
        for message in messages:
            kind = message.get('type')
            if kind == 'action':
                driver._receive(int(message.get('step', 0)), message.get('action'))
            elif kind == 'spec':
                driver.set_spec(message.get('spec'))
        driver.connected = False
        sock.close()

    def _check_hello(self, hello) -> Optional[str]:
        if not hello or hello.get('type') != 'hello':
            return 'expected a hello message'
        if hello.get('protocol') != PROTOCOL:
            return f'this race uses protocol {PROTOCOL}; update fuzzy_torch on your laptop'
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
        self._closed = True
        self._server.close()
        for driver in list(self.drivers.values()):
            driver.connected = False
            try:
                driver._sock.close()
            except OSError:
                pass


class Join:
    """A laptop's connection to a :class:`KartHub` (see :func:`join`)."""

    def __init__(self, server: str, name: str, driver: Callable, spec: Optional[dict] = None,
                 reload: Optional[Callable[[], tuple]] = None):
        host, _, port = server.rpartition(':')
        self.sock = socket.create_connection((host or 'localhost', int(port)), timeout=10)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self._lock = threading.Lock()
        self.driver, self.reload = driver, reload
        self.messages = _lines(self.sock)
        _send(self.sock, {'type': 'hello', 'name': name, 'spec': spec, 'protocol': PROTOCOL}, self._lock)
        welcome = next(self.messages, None)
        if not welcome or welcome.get('type') != 'welcome':
            self.sock.close()
            raise ConnectionError((welcome or {}).get('error', 'the race server closed the connection'))
        self.name = welcome['name']

    def run(self, on_message: Optional[Callable[[dict, Optional[np.ndarray]], bool]] = None):
        """Answer observations until the server closes; ``on_message(message, action)``
        sees every message (with the action sent, for ``obs``) and may return False to stop."""
        for message in self.messages:
            action = None
            if message.get('type') == 'start' and self.reload is not None:
                self.driver, spec = self.reload()
                _send(self.sock, {'type': 'spec', 'spec': spec}, self._lock)
                if hasattr(self.driver, 'reset'):
                    self.driver.reset()
            elif message.get('type') == 'obs':
                obs = np.asarray(message['obs'], np.float32)
                try:
                    action = np.asarray(self.driver(obs), np.float32).reshape(-1)[:2]
                except Exception as exc:              # noqa: BLE001 -- keep driving
                    print(f"your driver raised: {exc}")
                    action = np.zeros(2, np.float32)
                _send(self.sock, {'type': 'action', 'step': message['step'],
                                  'action': [float(a) for a in action]}, self._lock)
            if on_message is not None and on_message(message, action) is False:
                break
        self.sock.close()


def join(server: str, name: str, driver: Callable, spec: Optional[dict] = None, **kwargs) -> Join:
    """Connect ``driver`` (``action = driver(observation)``) to a race at ``host:port``.

    Pass the driver's JSON ``spec`` to have its fuzzy systems drawn on the projector.
    Returns the connection; call ``.run()`` on it to drive.
    """
    return Join(server, name, driver, spec, **kwargs)
