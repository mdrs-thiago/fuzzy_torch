"""Race a fuzzy driver in SuperTuxKart against AI karts.

    python examples/supertuxkart_demo.py                                  # baseline, headless
    python examples/supertuxkart_demo.py my_driver.json --track zengarden --live
    python examples/supertuxkart_demo.py --opponents 5 --laps 3 --gif out/race.gif

A driver is any Race Control JSON file (see ``examples/race_control/README.md``);
without one the hand-written ``FuzzyRaceDriver`` baseline drives.  ``--live`` and
``--gif`` render the game, which pystk2 allows only if no headless race ran before in
the same process, so this script renders either all tracks or none.
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

from fuzzy_torch.envs import SuperTuxKartEnv, record
from fuzzy_torch.envs.controllers import ConfigRaceDriver, FuzzyRaceDriver
from fuzzy_torch.envs.supertuxkart import TRACKS


def race(env, driver):
    obs, _ = env.reset(seed=0)
    while True:
        obs, _, terminated, truncated, info = env.step(driver(obs))
        if terminated or truncated or not env.is_running:
            return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('driver', nargs='?', help='Race Control JSON driver (default: baseline)')
    ap.add_argument('--track', action='append', help=f'track(s) to race (default: {", ".join(TRACKS[:3])})')
    ap.add_argument('--opponents', type=int, default=3, help='AI karts (default: 3)')
    ap.add_argument('--difficulty', type=int, choices=[0, 1, 2], default=1, help='AI skill')
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--live', action='store_true', help='watch the race in a window')
    ap.add_argument('--gif', help='record the (first) race to this .gif or .webp file')
    ap.add_argument('--every', type=int, default=2, help='with --gif: keep one frame in this many')
    args = ap.parse_args()

    driver = ConfigRaceDriver(json.loads(pathlib.Path(args.driver).read_text())) if args.driver \
        else FuzzyRaceDriver()
    mode = 'human' if args.live else ('rgb_array' if args.gif else None)
    for i, track in enumerate(args.track or TRACKS[:3]):
        env = SuperTuxKartEnv(track, opponents=args.opponents, difficulty=args.difficulty,
                              laps=args.laps, render_mode=mode)
        if args.gif and i == 0:
            video = record(env, driver, every=args.every)
            info = video.info
            print(f'wrote {video.save(args.gif)}')
        else:
            info = race(env, driver)
        env.close()
        result = f"{info['race_time']:6.2f} s" if info.get('race_time') else '   DNF  '
        print(f"{track:20s} {result}  position {info['position']}/{info['num_karts']}  "
              f"rescues {info['rescues']}")


if __name__ == '__main__':
    main()
