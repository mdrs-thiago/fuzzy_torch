"""Join the class SuperTuxKart race from your laptop, driving with your fuzzy driver.

    python examples/kart_join.py https://abcd-1234.ngrok-free.app --code K7QX2 --name "Ana Souza" --driver my_driver.json
    python examples/kart_join.py 192.168.0.10:9000 --code K7QX2 --name "Ana Souza" --driver my_driver.json

The address and the join code are shown on the race screen.

Your laptop runs your driver; the instructor's machine runs the race and shows it on
the projector.  Your window shows your own fuzzy systems firing as your kart races.
The driver file is read again before every race, so you can improve your rules
between heats: edit, save, and your kart uses them in the next race.

Only fuzzy_torch is needed here (``pip install 'torch-fis[envs]'``), not
SuperTuxKart.  Add ``--no-window`` to drive without a window.

To drive with your own Python controller instead of a JSON file, call
``fuzzy_torch.envs.kart_net.join(server, name, code, my_controller).run()``, where
``my_controller(observation)`` returns ``[throttle, steering]``.
"""
import argparse
import importlib
import json
import pathlib
import sys
import threading

if __package__ is None and 'fuzzy_torch' not in sys.modules:
    try:
        importlib.import_module('fuzzy_torch')
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fuzzy_torch.envs.controllers import ConfigRaceDriver
from fuzzy_torch.envs.kart_net import join

WIDTH, HEIGHT = 480, 720


def load(path: pathlib.Path):
    spec = json.loads(path.read_text())
    return ConfigRaceDriver(spec, max_rules=125, max_inputs=3, max_terms=7), spec


def describe(message: dict) -> str:
    kind = message.get('type')
    if kind == 'start':
        return f"Racing: {message['race']} on {message['track']}"
    if kind == 'wait':
        return f"Watching {message['race']} ({', '.join(message['entrants'])[:60]})"
    if kind == 'end':
        rows = message['standings']
        return f"{message['race']} over: " + ', '.join(
            f"{r['position']}. {r['name']}" for r in rows[:4])
    return ''


class Telemetry:
    """The latest message from the race, shared with the window."""

    def __init__(self, driver):
        self.driver, self.status, self.obs, self.action, self.info = driver, 'Waiting for the race', None, None, {}
        self.ended, self.lock = False, threading.Lock()

    def __call__(self, message, action):
        with self.lock:
            if message.get('type') == 'obs':
                self.obs, self.action = message['obs'], action
                self.info = {k: message.get(k) for k in ('position', 'karts', 'lap', 'time')}
            elif describe(message):
                self.status = describe(message)
                print(self.status)
                if message.get('type') == 'start':
                    self.obs = None


def window(connection, telemetry: Telemetry):
    import pygame
    from fuzzy_torch.envs.controllers.racing import race_inputs
    from fuzzy_torch.envs.kart_view import ACCENT, BG, DIM, PANEL, TEXT, FuzzyPanel, _text
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f'Fuzzy Kart — {connection.name}')
    mono = 'dejavusansmono,menlo,consolas,monospace'
    fonts = {'bold': pygame.font.SysFont(mono, 16, bold=True), 'small': pygame.font.SysFont(mono, 13),
             'tiny': pygame.font.SysFont(mono, 11), 'big': pygame.font.SysFont(mono, 22, bold=True)}
    clock, panels, shown = pygame.time.Clock(), [], None
    while not telemetry.ended:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                connection.close()
                return
        with telemetry.lock:
            driver, status, obs, action, info = (telemetry.driver, telemetry.status, telemetry.obs,
                                                 telemetry.action, dict(telemetry.info))
        driver = connection.driver
        if driver is not shown:
            panels = [FuzzyPanel(s, t) for s, t in ((driver.steering, 'STEERING'),
                                                     (driver.throttle, 'THROTTLE'))]
            shown = driver
        screen.fill(PANEL)
        _text(screen, fonts['big'], connection.name[:24], (10, 8), TEXT)
        _text(screen, fonts['small'], status[:62], (10, 36), ACCENT)
        y = 58
        if obs is not None and info.get('position'):
            _text(screen, fonts['bold'], f"P{info['position']}/{info['karts']}   lap {info['lap']}   "
                  f"{info['time']:.1f} s   speed {obs[2]:.1f}", (10, y), TEXT)
            if action is not None:
                _text(screen, fonts['tiny'], f"throttle {action[0]:+.2f}   steering {action[1]:+.2f}",
                      (10, y + 20), DIM)
            y += 40
            values = race_inputs(obs)
            for panel in panels:
                h = panel.height(3)
                panel.draw(screen, fonts, (0, y, WIDTH, h), values, 3)
                y += h + 4
        else:
            pygame.draw.rect(screen, BG, (10, y, WIDTH - 20, 40))
            _text(screen, fonts['small'], 'Your fuzzy systems appear here while you race.', (18, y + 12), DIM)
        pygame.display.flip()
        clock.tick(20)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('server', help="the race's address, as shown on the race screen")
    ap.add_argument('--code', required=True, help='the join code shown on the race screen')
    ap.add_argument('--name', required=True, help='your name on the standings')
    ap.add_argument('--driver', required=True, type=pathlib.Path, help='your JSON driver')
    ap.add_argument('--no-window', action='store_true', help='drive without a window')
    args = ap.parse_args()

    driver, spec = load(args.driver)
    current = [driver, spec]

    def reload():
        # A file saved mid-edit must not knock the kart out: keep the last good driver.
        try:
            current[:] = load(args.driver)
        except (OSError, ValueError) as exc:
            print(f"{args.driver} has a problem, racing the previous version: {exc}")
        return tuple(current)

    try:
        connection = join(args.server, args.name, args.code, driver, spec, reload=reload)
    except (ConnectionError, OSError) as exc:
        raise SystemExit(f"Could not join the race at {args.server}: {exc}")
    print(f"Joined as {connection.name}. Waiting for the race to start...")
    telemetry = Telemetry(driver)

    def drive():
        connection.run(telemetry)
        telemetry.ended = True
        print('The race server closed the connection.')

    if args.no_window:
        drive()
    else:
        threading.Thread(target=drive, daemon=True).start()
        window(connection, telemetry)


if __name__ == '__main__':
    main()
