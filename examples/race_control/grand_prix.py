#!/usr/bin/env python3
"""The official Fuzzy Grand Prix evaluation, usable on its own.

Race Control calls :func:`evaluate` for every submission.  Run it locally to check a
driver file before submitting, or to watch any entry in the 3-D MuJoCo view::

    python examples/race_control/grand_prix.py my_driver.json
    python examples/race_control/grand_prix.py my_driver.json --live --track chicane

The JSON format is described in this folder's README; ``example_driver.json`` is
the hand-written baseline driver in that format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from fuzzy_torch.config import ConfigError  # noqa: E402
from fuzzy_torch.envs import RacingEnv  # noqa: E402
from fuzzy_torch.envs.controllers import ConfigRaceDriver  # noqa: E402

TRACKS = ('oval', 'chicane', 'technical')
SEED = 2026
DT = 0.02
# Size limits keep one public submission from monopolising the evaluation queue.
LIMITS = {'max_rules': 125, 'max_inputs': 3, 'max_terms': 7}
# Trajectory samples every RECORD_EVERY steps (0.08 s at dt = 0.02) for replays.
RECORD_EVERY = 4
SAMPLE_DT = RECORD_EVERY * DT
# A kart that gains less than STALL_DISTANCE metres in STALL_WINDOW seconds is
# retired early: it would otherwise sit out the full 45 s limit and stall the queue.
STALL_WINDOW, STALL_DISTANCE = 4.0, 0.25


def build_driver(spec: Any) -> ConfigRaceDriver:
    """Validate a spec and build its driver; raises ConfigError with a field path."""
    return ConfigRaceDriver(spec, **LIMITS)


def run_lap(driver, track: str, *, live: bool = False, backend: str = 'auto') -> dict[str, Any]:
    """One timed lap. ``path`` holds ``[x, y, speed, heading]`` samples for replays."""
    env = RacingEnv(track, dt=DT, render_mode='human' if live else None, render_backend=backend)
    try:
        obs, _ = env.reset(seed=SEED)
        state = env._state[0]
        path = [[float(state[0]), float(state[1]), 0.0, float(state[2])]]
        window = int(round(STALL_WINDOW / env.dt))
        progress_log = [0.0]
        steps, detail = 0, ''
        while True:
            action = np.asarray(driver(obs), dtype=np.float32)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError(f"driver produced a non-finite action {action.tolist()} on "
                                 f"{track} at t={env.time:.2f} s")
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if steps % RECORD_EVERY == 0 or terminated or truncated:
                state = env._state[0]
                path.append([float(state[0]), float(state[1]), float(state[3]), float(state[2])])
            progress_log.append(info['progress'])
            if live and not env.is_running:
                detail = 'Window closed'
                break
            if terminated or truncated:
                detail = 'Off track' if info['off_track'] else ('Time limit' if truncated else '')
                break
            if steps >= window and progress_log[-1] - progress_log[-1 - window] < STALL_DISTANCE:
                detail = 'Stalled'
                break
        lap_time = info['lap_time']
        return {
            'time': None if lap_time is None else round(float(lap_time), 3),
            'status': 'FINISHED' if lap_time is not None else 'DNF',
            'detail': detail,
            'progress': round(min(1.0, info['progress'] / env.track.length), 4),
            'top_speed': round(max(p[2] for p in path), 3),
            'path': [[round(x, 3), round(y, 3), round(v, 2), round(psi, 3)] for x, y, v, psi in path],
        }
    finally:
        env.close()


def race_track(spec: Any, track: str) -> dict[str, Any]:
    """One lap from a spec; the unit of work Race Control farms out to worker processes."""
    import torch
    torch.set_num_threads(1)
    return run_lap(build_driver(spec), track)


def score(laps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Championship score: total time when every circuit is finished.

    Entries that do not finish everything rank below those that do, ordered by how
    many circuits they finished and then by average lap completion.
    """
    finished = [lap for lap in laps.values() if lap['time'] is not None]
    total = round(sum(lap['time'] for lap in finished), 3) if len(finished) == len(TRACKS) else None
    return {'total': total, 'finished': len(finished),
            'completion': round(sum(lap['progress'] for lap in laps.values()) / len(TRACKS), 4)}


def evaluate(spec: Any, on_track: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Validate and race a spec on every circuit."""
    driver = build_driver(spec)
    laps = {}
    for track in TRACKS:
        if on_track is not None:
            on_track(track)
        laps[track] = run_lap(driver, track)
    return {'laps': laps, **score(laps)}


def describe(spec: dict) -> dict[str, Any]:
    """A short summary for leaderboard tiles: mode, rule count and inputs per system."""
    out = {}
    for name in ('throttle', 'steering'):
        system = spec[name]
        first = system['rules'][0]['then']
        out[name] = {'mode': 'Mamdani' if isinstance(first, str) else 'TSK',
                     'rules': len(system['rules']), 'inputs': list(system['inputs'])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('driver', type=Path, help='driver JSON file')
    parser.add_argument('--track', choices=TRACKS, help='race one circuit only')
    parser.add_argument('--live', action='store_true', help='watch the lap in a window')
    parser.add_argument('--backend', choices=['auto', 'mujoco', 'pygame'], default='auto')
    args = parser.parse_args()
    try:
        spec = json.loads(args.driver.read_text())
        driver = build_driver(spec)
    except (json.JSONDecodeError, ConfigError) as exc:
        sys.exit(f"invalid driver: {exc}")
    laps = {}
    for track in ([args.track] if args.track else TRACKS):
        lap = laps[track] = run_lap(driver, track, live=args.live, backend=args.backend)
        result = f"{lap['time']:7.3f} s" if lap['time'] is not None else f"    DNF ({lap['detail']})"
        print(f"{track:10s} {result}   completion {lap['progress']:6.1%}   "
              f"top speed {lap['top_speed']:.2f} m/s")
    if not args.track:
        s = score(laps)
        print(f"{'TOTAL':10s} {s['total']:7.3f} s" if s['total'] is not None
              else f"TOTAL      — finished {s['finished']}/{len(TRACKS)} circuits")


if __name__ == '__main__':
    main()
