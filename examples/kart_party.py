"""The whole class in one SuperTuxKart race, every kart driven by a student's fuzzy driver.

    # one driver against AI karts, with its fuzzy systems shown live
    python examples/kart_party.py --driver examples/race_control/example_driver.json --ai 3

    # every student on the Race Control leaderboard (their latest valid driver each)
    python examples/kart_party.py --store examples/race_control/leaderboard.json --track zengarden

    # a folder of JSON drivers, one per student (file name = student name)
    python examples/kart_party.py --folder submissions/ --laps 2 --ai 2

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
import sys

if __package__ is None and 'fuzzy_torch' not in sys.modules:
    try:
        importlib.import_module('fuzzy_torch')
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from fuzzy_torch.config import ConfigError
from fuzzy_torch.envs import KartRace
from fuzzy_torch.envs.controllers import ConfigRaceDriver, FuzzyRaceDriver
from fuzzy_torch.envs.kart_view import RaceDirector

# The same limits as the Race Control server, since the specs come from students.
LIMITS = {'max_rules': 125, 'max_inputs': 3, 'max_terms': 7}


def from_store(path: pathlib.Path) -> dict:
    """Each name's most recent valid submission in a Race Control results file."""
    specs = {}
    entries = sorted(json.loads(path.read_text()), key=lambda e: e['submitted_at'])
    for entry in entries:
        if entry.get('status') != 'error':
            specs[' '.join(entry['name'].split())] = entry['spec']
    return specs


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
    args = ap.parse_args()

    drivers = load_drivers(args)
    names = list(drivers)
    print(f"{len(names)} driver(s): {', '.join(names)}")
    size = max(1, min(args.heat_size, KartRace.max_cameras))
    # Deal the class round-robin into the fewest heats of at most `size`.
    count = -(-len(names) // size)
    heats = [names[i::count] for i in range(count)]
    races = [(f"Heat {i + 1}/{count}" if count > 1 else 'Race', heat) for i, heat in enumerate(heats)]

    director = RaceDirector(realtime=not args.fast, headless=bool(args.screenshot or args.headless))
    results = []
    try:
        while races:
            title, entrants = races.pop(0)
            race = KartRace({n: drivers[n] for n in entrants}, args.track, laps=args.laps,
                            ai_karts=args.ai, difficulty=args.difficulty, render=True)
            standings = director.run(race, seed=args.seed, title=f"{title} · {args.track}",
                                     max_steps=args.steps if args.screenshot else None)
            race.close()
            if standings is None:
                break
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


if __name__ == '__main__':
    main()
