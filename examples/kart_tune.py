"""Tune your fuzzy driver in SuperTuxKart on your own computer, before race day.

    python examples/kart_tune.py my_driver.json                      # writes my_driver_tuned.json
    python examples/kart_tune.py my_driver.json --budget 60 --tracks lighthouse zengarden
    python examples/kart_tune.py my_driver.json --lag 3              # racing from a laptop

Only the numbers change: membership-function breakpoints and TSK rule outputs.  Your
inputs, terms and rules stay as you wrote them, so the tuned driver is still yours
and still reads the same.  Each membership function keeps its shape: corners stay in
order, terms that touch the edge of a range stay shoulders, and widths stay positive.

The search is a (1+1) evolution strategy: perturb all numbers a little, keep the
change if the total race time drops, and widen or narrow the steps by the 1/5 success
rule.  Races are solo, which makes them repeatable, so every comparison is fair.  A
driver that leaves the race unfinished scores 150 s plus the share of the lap left.

``--lag N`` delays every action by N steps of 50 ms while tuning: use the round trip
of your connection when you will race from your laptop (2 or 3 for most home
networks).  ``--check`` races the result on other tracks, to see whether it learnt to
drive or only learnt the tuning tracks.

Each evaluation races every tuning track once (a few seconds each), so the default
budget of 40 takes about 5 to 10 minutes.  Needs SuperTuxKart: pip install 'torch-fis[kart]'.
"""
import argparse
import collections
import copy
import importlib
import json
import math
import pathlib
import sys

if __package__ is None and 'fuzzy_torch' not in sys.modules:
    try:
        importlib.import_module('fuzzy_torch')
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from fuzzy_torch.config import ConfigError
from fuzzy_torch.envs.controllers import ConfigRaceDriver

LIMITS = {'max_rules': 125, 'max_inputs': 3, 'max_terms': 7}
DNF = 150.0


class Parameters:
    """The tunable numbers of a driver spec, as one vector.

    Each entry knows where it lives in the spec and its natural scale: the width of
    its variable's range.  Numbers sitting on a range edge are left alone, since they
    make a term a shoulder.
    """

    def __init__(self, spec: dict):
        self.spec = copy.deepcopy(spec)
        self.slots = []                   # (container, index, scale)
        self.terms = []                   # (params list, shape, lo, hi) to repair after a move
        for system in ('steering', 'throttle'):
            sub = self.spec[system]
            for name, var in sub['inputs'].items():
                self._variable(var)
            if 'terms' in sub['output']:
                self._variable(sub['output'])
            out_lo, out_hi = sub['output']['range']
            names = list(sub['inputs'])
            for rule in sub['rules']:
                then = rule['then']
                if isinstance(then, (int, float)) and not isinstance(then, bool):
                    self.slots.append((rule, 'then', out_hi - out_lo))
                elif isinstance(then, list):
                    self.slots.append((then, 0, out_hi - out_lo))
                    for k, name in enumerate(names[:len(then) - 1]):
                        lo, hi = sub['inputs'][name]['range']
                        self.slots.append((then, k + 1, (out_hi - out_lo) / max(hi - lo, 1e-9)))

    def _variable(self, var):
        lo, hi = var['range']
        for params in var['terms'].values():
            shape = params[0]
            self.terms.append((params, shape, lo, hi))
            for i in range(1, len(params)):
                if shape in ('tri', 'trap') and params[i] in (lo, hi):
                    continue                  # a shoulder's anchor
                self.slots.append((params, i, hi - lo))

    @property
    def vector(self) -> np.ndarray:
        return np.array([c[i] for c, i, _ in self.slots], dtype=np.float64)

    @property
    def scales(self) -> np.ndarray:
        return np.array([s for *_, s in self.slots], dtype=np.float64)

    def with_vector(self, x: np.ndarray) -> dict:
        """A copy of the spec holding ``x``, repaired into valid membership functions."""
        for (container, index, _), value in zip(self.slots, x):
            container[index] = float(value)
        for params, shape, lo, hi in self.terms:
            if shape in ('tri', 'trap'):
                params[1:] = sorted(min(max(v, lo), hi) for v in params[1:])
            elif shape == 'gauss':
                params[2] = max(abs(params[2]), 1e-3 * (hi - lo))
            elif shape == 'bell':
                params[1] = max(abs(params[1]), 1e-3 * (hi - lo))
                params[2] = max(abs(params[2]), 0.1)
        spec = copy.deepcopy(self.spec)
        return _rounded(spec)


def _rounded(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [_rounded(v) for v in x]
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    return x


def race_times(spec: dict, tracks, lag: int = 0, max_time: float = DNF):
    """Solo race time on each track; a DNF scores ``max_time`` plus the share of the lap left."""
    from fuzzy_torch.envs import SuperTuxKartEnv
    driver = ConfigRaceDriver(spec, **LIMITS)
    times = []
    for track in tracks:
        env = SuperTuxKartEnv(track, max_time=max_time)
        obs, _ = env.reset(seed=0)
        delayed = collections.deque([np.zeros(2, np.float32)] * lag)
        while True:
            delayed.append(driver(obs))
            obs, _, terminated, truncated, info = env.step(delayed.popleft())
            if terminated or truncated:
                break
        left = 1.0 - min(1.0, info['progress'] / env.track.length)
        times.append(info['race_time'] if info['race_time'] is not None else max_time + 100 * left)
        env.close()
    return times


def tune(spec: dict, tracks, budget: int = 40, sigma: float = 0.05, lag: int = 0,
         seed: int = 0, log=print):
    """(1+1)-ES over the spec's numbers; returns ``(best_spec, best_times, start_times)``."""
    rng = np.random.default_rng(seed)
    params = Parameters(spec)
    x, scales = params.vector, params.scales
    best_spec = params.with_vector(x)
    start = best = race_times(best_spec, tracks, lag)
    log(f"start      total {sum(best):7.2f} s   " + _per_track(tracks, best))
    for k in range(1, budget + 1):
        candidate = x + rng.normal(0.0, sigma, size=x.shape) * scales
        try:
            spec_k = params.with_vector(candidate)
            ConfigRaceDriver(spec_k, **LIMITS)
        except (ConfigError, ValueError):
            sigma *= 0.8
            continue
        times = race_times(spec_k, tracks, lag)
        if sum(times) < sum(best):
            x, best, best_spec = params.vector, times, spec_k
            sigma *= 1.5                           # 1/5 success rule: grow on success...
            log(f"try {k:3d}    total {sum(best):7.2f} s   " + _per_track(tracks, best)
                + f"   step {sigma:.3f}")
        else:
            sigma *= 1.5 ** -0.25                  # ...and shrink four times as slowly on failure
            params.with_vector(x)                  # put the best numbers back
        sigma = float(np.clip(sigma, 1e-3, 0.3))
    return best_spec, best, start


def _per_track(tracks, times):
    return '  '.join(f"{t} {v:6.1f}" for t, v in zip(tracks, times))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('driver', type=pathlib.Path, help='your JSON driver')
    ap.add_argument('--tracks', nargs='+', default=['lighthouse', 'zengarden', 'hacienda'],
                    help='tracks to tune on (default: lighthouse zengarden hacienda)')
    ap.add_argument('--check', nargs='*', default=['cornfield_crossing', 'snowmountain', 'scotland'],
                    help='tracks to test the result on afterwards (none: skip)')
    ap.add_argument('--budget', type=int, default=40, help='candidate drivers to try (default: 40)')
    ap.add_argument('--sigma', type=float, default=0.05,
                    help='first step size, as a fraction of each range (default: 0.05)')
    ap.add_argument('--lag', type=int, default=0, metavar='STEPS',
                    help='delay actions by this many 50 ms steps, as over a network')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=pathlib.Path, help='output file (default: <driver>_tuned.json)')
    args = ap.parse_args()

    spec = json.loads(args.driver.read_text())
    try:
        ConfigRaceDriver(spec, **LIMITS)
    except ConfigError as exc:
        raise SystemExit(f"{args.driver}: {exc}")
    print(f"Tuning {len(Parameters(spec).slots)} numbers of {args.driver.name} on "
          f"{', '.join(args.tracks)}" + (f", with {args.lag} steps of lag" if args.lag else ''))
    best, times, start = tune(spec, args.tracks, args.budget, args.sigma, args.lag, args.seed)
    out = args.out or args.driver.with_name(args.driver.stem + '_tuned.json')
    if 'description' in best:
        best['description'] = (str(best['description']) + ' (tuned)')[:200]
    out.write_text(json.dumps(best, indent=1) + '\n')
    gain = 100 * (1 - sum(times) / sum(start))
    print(f"\nwrote {out}: {sum(start):.1f} s -> {sum(times):.1f} s ({gain:.1f}% faster)")
    if args.check:
        before, after = race_times(spec, args.check, args.lag), race_times(best, args.check, args.lag)
        print("on the check tracks (not tuned on):")
        for track, b, a in zip(args.check, before, after):
            print(f"  {track:20s} {b:6.1f} s -> {a:6.1f} s")
        verdict = ('it learnt to drive, not just the tuning tracks' if sum(after) < sum(before)
                   else 'slower here: it fitted the tuning tracks; tune on more tracks')
        print(f"  total {sum(before):.1f} s -> {sum(after):.1f} s: {verdict}")


if __name__ == '__main__':
    main()
