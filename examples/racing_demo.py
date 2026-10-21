"""Run the fuzzy time-trial baseline on all lecture circuits.

    python examples/racing_demo.py --out out/
"""
import argparse
import importlib
import pathlib
import sys

if __package__ is None and 'fuzzy_torch' not in sys.modules:
    try:
        importlib.import_module('fuzzy_torch')
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from fuzzy_torch.envs import RacingEnv
from fuzzy_torch.envs.controllers import FuzzyRaceDriver


def run(track, live=False, backend='auto'):
    env = RacingEnv(track, render_mode='human' if live else None, render_backend=backend)
    driver = FuzzyRaceDriver()
    obs, _ = env.reset(seed=0)
    path = [env._state[0, :2].numpy().copy()]
    while True:
        obs, _, terminated, truncated, info = env.step(driver(obs))
        path.append(env._state[0, :2].numpy().copy())
        if terminated or truncated or not env.is_running:
            break
    return env, np.asarray(path), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out', help='directory for the circuit plot')
    ap.add_argument('--live', action='store_true', help='open a live circuit window')
    ap.add_argument('--backend', choices=['auto', 'mujoco', 'pygame'], default='auto',
                    help='3-D MuJoCo view when available; pygame is the 2-D fallback')
    ap.add_argument('--track', choices=['oval', 'chicane', 'technical'],
                    help='run just one circuit (useful for a live lecture demo)')
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    names = (args.track,) if args.track else ('oval', 'chicane', 'technical')
    for ax, name in zip(axes, names):
        env, path, info = run(name, live=args.live, backend=args.backend)
        centre = env.track.points
        ax.plot(*np.vstack([centre, centre[:1]]).T, '--', color='#777', label='centre line')
        ax.plot(path[:, 0], path[:, 1], color='#c03c3c', label='fuzzy driver')
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        result = f"{info['lap_time']:.2f} s" if info['lap_time'] else 'DNF'
        print(f'{name:10s} {result}; progress={info["progress"]:.1f} m; off-track={info["off_track"]}')
        env.close()
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.suptitle('Fuzzy Grand Prix: identical driver, three circuits')
    fig.tight_layout()
    target = out / 'racing_time_trial.png'
    fig.savefig(target, dpi=130)
    print(f'wrote {target}')


if __name__ == '__main__':
    main()
