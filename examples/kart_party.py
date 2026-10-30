"""The whole class in one SuperTuxKart race, every kart driven by a student's fuzzy driver.

    # one driver against AI karts, with its fuzzy systems shown live
    python examples/kart_party.py --driver examples/race_control/example_driver.json --ai 3

    # every student on the Race Control leaderboard (their latest valid driver each)
    python examples/kart_party.py --store examples/race_control/leaderboard.json --track zengarden

    # a folder of JSON drivers, one per student (file name = student name)
    python examples/kart_party.py --folder submissions/ --laps 2 --ai 2

    # students drive from their laptops (examples/kart_join.py), plus any JSON drivers
    python examples/kart_party.py --join 9000 --ai 2
    # ...from anywhere, through a tunnel: run `ngrok http 9000` and pass its address
    python examples/kart_party.py --join 9000 --public-url https://abcd-1234.ngrok-free.app

SuperTuxKart renders at most 8 chase cameras per race, so a class larger than
``--heat-size`` (8) races in heats; the fastest 8 overall then meet in a final.

Drivers are Race Control JSON files (see ``examples/race_control/README.md``), so they
are data, not code, and safe to run from anyone.  In the window: ``Left``/``Right``
or a click on the standings follows another driver, ``L`` follows the leader,
``Space`` pauses, ``F`` toggles real-time pacing, ``Enter`` starts the next race
once one is over, ``Esc`` quits.  Every race's standings are printed as it ends.
"""
import argparse
import importlib
import json
import pathlib
import socket
import sys
import threading
import time
from typing import Optional

if __package__ is None and 'fuzzy_torch' not in sys.modules:
    try:
        importlib.import_module('fuzzy_torch')
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fuzzy_torch.config import ConfigError
from fuzzy_torch.envs import KartRace
from fuzzy_torch.envs.controllers import ConfigRaceDriver, FuzzyRaceDriver
from fuzzy_torch.envs.kart_net import KartHub
from fuzzy_torch.envs.kart_view import RaceDirector

# The same limits as the Race Control server, since the specs come from students.
LIMITS = {'max_rules': 125, 'max_inputs': 3, 'max_terms': 7}


def from_store(path: pathlib.Path) -> dict:
    """Each student's most recent valid submission in a Race Control results file.

    Names match as Race Control matches them, ignoring case and spacing, so
    'Ana Souza' and 'ana  souza' are one student, shown under the latest spelling.
    """
    latest = {}
    entries = sorted(json.loads(path.read_text()), key=lambda e: e['submitted_at'])
    for entry in entries:
        if entry.get('status') != 'error':
            name = ' '.join(entry['name'].split())
            latest[name.casefold()] = (name, entry['spec'])
    return dict(latest.values())


def load_drivers(args) -> dict:
    specs = {}
    if args.store:
        specs.update(from_store(pathlib.Path(args.store)))
    if args.folder:
        for path in sorted(pathlib.Path(args.folder).glob('*.json')):
            specs[path.stem] = json.loads(path.read_text())
    for path in args.driver or ():
        specs[pathlib.Path(path).stem] = json.loads(pathlib.Path(path).read_text())
    drivers = {}
    for name, spec in specs.items():
        try:
            drivers[name] = ConfigRaceDriver(spec, **LIMITS)
        except (ConfigError, ValueError, TypeError) as exc:
            print(f"skipping {name}: {exc}")
    if args.baseline or not drivers:
        drivers['Baseline'] = FuzzyRaceDriver()
    return drivers


def lan_address() -> str:
    """This machine's address on the local network, for students to connect to."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(('192.0.2.1', 9))          # TEST-NET: nothing is sent
            return probe.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def lobby(hub: KartHub, seconds: Optional[float]):
    """Wait for students to join: until Enter, or for ``seconds``."""
    done = threading.Event()
    if seconds is None:
        print('Waiting for students. Press Enter to start the race.')
        threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    shown, start = [], time.monotonic()
    while not done.is_set() and (seconds is None or time.monotonic() - start < seconds):
        if hub.names != shown:
            shown = hub.names
            print(f"  {len(shown)} joined: {', '.join(shown)}")
        done.wait(0.5)


def ranking_key(row):
    """Finishers by time, then the rest by distance covered."""
    finished = row['race_time'] is not None
    return (not finished, row['race_time'] if finished else -row['progress'])


def print_standings(title, standings):
    print(f"\n{title}\n{'pos':>3}  {'driver':24s} result")
    for row in standings:
        result = (f"{row['race_time']:.2f} s" if row['race_time'] is not None
                  else f"lap {row['lap']}, {100 * row['progress']:.0f}% done")
        error = f"  (driver error {row['error']})" if row['error'] else ''
        print(f"{row['position']:3d}  {row['name'][:24]:24s} {result}{error}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', help='Race Control results file (leaderboard.json)')
    ap.add_argument('--folder', help='folder of JSON drivers, one per student')
    ap.add_argument('--driver', action='append', help='a JSON driver file (repeatable)')
    ap.add_argument('--baseline', action='store_true', help='add the hand-written baseline driver')
    ap.add_argument('--track', default='lighthouse')
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--ai', type=int, default=0, help='AI karts to race against')
    ap.add_argument('--difficulty', type=int, choices=[0, 1, 2], default=1, help='AI skill')
    ap.add_argument('--heat-size', type=int, default=KartRace.max_cameras,
                    help='students per heat (default: 8, one camera each)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fast', action='store_true', help='run as fast as possible, not in real time')
    ap.add_argument('--headless', action='store_true',
                    help='no window: run the whole session and print the results (a dry run)')
    ap.add_argument('--screenshot', help='headless: race --steps steps of the first race, '
                                         'save the window to this PNG and stop')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--join', type=int, metavar='PORT',
                    help='let students join from their laptops on this port')
    ap.add_argument('--code', help='with --join: the join code (default: a random one)')
    ap.add_argument('--public-url', help='with --join: the address students use, e.g. the '
                                         "https:// address printed by 'ngrok http PORT'")
    ap.add_argument('--lobby', type=float, metavar='SECONDS',
                    help='with --join: start after this many seconds instead of on Enter')
    args = ap.parse_args()

    hub = None
    if args.join:
        hub = KartHub('0.0.0.0', args.join, code=args.code)
        address = args.public_url or f"{lan_address()}:{args.join}"
        banner = f"Join: {address}   code {hub.code}"
        print(f"Students join with:\n  python examples/kart_join.py {address} --code {hub.code} "
              f"--name \"Your Name\" --driver my_driver.json")
        lobby(hub, args.lobby)
    drivers = {} if hub and not (args.store or args.folder or args.driver or args.baseline) \
        else load_drivers(args)
    if hub:
        # A student on a laptop drives live, in place of any file under the same name.
        drivers.update({name: hub.drivers[name] for name in hub.names})
    if not drivers:
        raise SystemExit('no drivers: nobody joined, and no JSON drivers were given')
    names = list(drivers)
    print(f"{len(names)} driver(s): {', '.join(names)}")
    size = max(1, min(args.heat_size, KartRace.max_cameras))
    # Deal the class round-robin into the fewest heats of at most `size`.
    count = -(-len(names) // size)
    heats = [names[i::count] for i in range(count)]
    races = [(f"Heat {i + 1}/{count}" if count > 1 else 'Race', heat) for i, heat in enumerate(heats)]

    director = RaceDirector(realtime=not args.fast, headless=bool(args.screenshot or args.headless))
    if hub:
        director.banner = banner
    results = []
    try:
        while races:
            title, entrants = races.pop(0)
            race = KartRace({n: drivers[n] for n in entrants}, args.track, laps=args.laps,
                            ai_karts=args.ai, difficulty=args.difficulty, render=True)
            if hub:
                hub.announce(title, args.track, entrants)
            standings = director.run(race, seed=args.seed, title=f"{title} · {args.track}",
                                     max_steps=args.steps if args.screenshot else None,
                                     on_step=hub.track_status if hub else None)
            race.close()
            if standings is None:
                break
            if hub:
                hub.results(title, standings)
            print_standings(title, standings)
            if args.screenshot:
                director.screenshot(args.screenshot)
                print(f"wrote {args.screenshot}")
                break
            results += [row for row in standings if not row['ai']]
            if not races and title.startswith('Heat'):
                finalists = [row['name'] for row in sorted(results, key=ranking_key)[:size]]
                races.append(('Final', finalists))
    finally:
        director.close()
        if hub:
            hub.close()


if __name__ == '__main__':
    main()
