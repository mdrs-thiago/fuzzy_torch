"""Run the crane environment under the PID/state-feedback baseline and visualise it.

    python examples/crane_demo.py --live          # interactive 3D MuJoCo viewer
    python examples/crane_demo.py --out out/      # write plots + animated GIF

The point of the visualisation is to see the *environment and controller together*:
the trolley tracking its setpoint while the anti-sway terms damp the payload swing
they themselves excite.
"""
import argparse
import importlib
import pathlib
import sys

# Running a script puts *its own* directory on sys.path, not the working directory,
# so `fuzzy_torch` is not importable from here unless the package is installed.
# Fall back to the repo's parent so the example works straight from a checkout.
if __package__ is None and "fuzzy_torch" not in sys.modules:
    try:
        importlib.import_module("fuzzy_torch")
    except ModuleNotFoundError:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from fuzzy_torch.envs import (
    CranePlant, disturbance_metrics, disturbance_rejection, step_response,
    step_response_metrics,
)
from fuzzy_torch.envs.controllers import crane_baseline
from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv
from fuzzy_torch.envs.core.wrappers import ErrorFeatures


def rollout(scenario, controller, render_mode=None, frame_every=0, backend='auto'):
    """Run one episode, returning the recorded trajectory and any captured frames."""
    env = ErrorFeatures(FuzzyControlEnv(CranePlant(), scenario=scenario,
                                        render_mode=render_mode,
                                        render_backend=backend))
    obs, _ = env.reset(seed=0)
    controller.reset()

    log = {k: [] for k in ('t', 'x', 'theta', 'force', 'applied', 'target', 'disturbed')}
    frames = []
    for i in range(scenario.episode_steps):
        action = controller(obs)
        obs, _, _, _, info = env.step(action)
        state = env.unwrapped._state[0]

        log['t'].append(info['time'])
        log['x'].append(float(state[0]))
        log['theta'].append(float(state[2]))
        log['force'].append(float(info['control'][0]))
        log['applied'].append(float(info['applied'][0]))
        log['target'].append(float(info['setpoint'][0]))
        log['disturbed'].append(info['disturbance'] is not None)

        if frame_every and i % frame_every == 0:
            frames.append(env.unwrapped.render())

        # Stop early if the viewer window was closed, rather than running the
        # remaining thousands of steps against a dead renderer.
        renderer = env.unwrapped._renderer
        if renderer is not None and not getattr(renderer, 'is_running', True):
            break

    env.close()
    return {k: np.asarray(v) for k, v in log.items()}, frames


def plot(step_log, dist_log, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), sharex='col')
    fig.suptitle('Gantry crane -- Gymnasium env + state-feedback baseline', fontsize=13)

    for col, (log, title) in enumerate([(step_log, 'Step response: move to 5 m'),
                                        (dist_log, 'Disturbance rejection: 25 N impulse')]):
        hit = log['disturbed']
        axes[0, col].set_title(title, fontsize=11)

        axes[0, col].plot(log['t'], log['target'], '--', color='#888', label='setpoint')
        axes[0, col].plot(log['t'], log['x'], color='#c03c3c', label='trolley x')
        axes[0, col].set_ylabel('position [m]')

        axes[1, col].axhline(0, color='#bbb', lw=0.8)
        axes[1, col].plot(log['t'], np.degrees(log['theta']), color='#2f6fa8')
        axes[1, col].set_ylabel('sway [deg]')

        axes[2, col].axhline(0, color='#bbb', lw=0.8)
        if hit.any():
            axes[2, col].plot(log['t'], log['applied'], color='#e8a33d', lw=1.0,
                              label='total on plant')
        axes[2, col].plot(log['t'], log['force'], color='#3f8f5f',
                          label='commanded by controller')
        axes[2, col].set_ylabel('force [N]')
        axes[2, col].set_xlabel('time [s]')
        axes[2, col].legend(loc='upper right', fontsize=8)

        if hit.any():
            t0, t1 = log['t'][hit][0], log['t'][hit][-1]
            for ax in axes[:, col]:
                ax.axvspan(t0, t1, color='#e8a33d', alpha=0.35,
                           label='disturbance' if ax is axes[0, col] else None)

        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
        axes[0, col].legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


def save_gif(frames, path, fps=25, max_width=640):
    """Write an animated GIF, downscaled and palette-quantised.

    A 1280x720 MuJoCo frame in full colour makes a ~14 MB GIF; one shared adaptive
    palette across all frames brings it under a megabyte with no visible loss.
    """
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames if f is not None]
    if not imgs:
        return
    if imgs[0].width > max_width:
        h = round(imgs[0].height * max_width / imgs[0].width)
        imgs = [im.resize((max_width, h), Image.LANCZOS) for im in imgs]
    palette = imgs[0].quantize(colors=128, method=Image.MEDIANCUT)
    imgs = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for im in imgs]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)
    size_mb = pathlib.Path(path).stat().st_size / 1e6
    print(f"wrote {path} ({len(imgs)} frames, {size_mb:.2f} MB)")


def report(name, metrics):
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k:22s} {v:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true',
                    help='open an interactive pygame window instead of writing files')
    ap.add_argument('--out', default='out', help='directory for plots and the GIF')
    ap.add_argument('--backend', default='auto', choices=['auto', 'mujoco', 'pygame'],
                    help="renderer: mujoco (3D) or pygame (2D, headless-safe)")
    args = ap.parse_args()

    dt = CranePlant.default_dt
    ctrl = crane_baseline()
    step_scn = step_response(1800, setpoint=[5.0, 0.0])
    dist_scn = disturbance_rejection(2500, setpoint=[5.0, 0.0], time=15.0,
                                     magnitude=25.0, duration=0.2)

    if args.live:
        rollout(step_scn, ctrl, render_mode='human', backend=args.backend)
        rollout(dist_scn, ctrl, render_mode='human', backend=args.backend)
        return

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ~8 frames of sim per rendered frame keeps the GIF small but smooth.
    step_log, frames = rollout(step_scn, ctrl, render_mode='rgb_array', frame_every=8,
                               backend=args.backend)
    dist_log, _ = rollout(dist_scn, ctrl)

    plot(step_log, dist_log, out / 'crane_response.png')
    save_gif(frames, out / 'crane.gif')

    report('step response (target 5 m)', step_response_metrics(
        __import__('torch').tensor(step_log['x']), target=5.0, dt=dt, initial=0.0,
        control=__import__('torch').tensor(step_log['force'])).as_dict())
    report('disturbance rejection (25 N at t=15 s)', disturbance_metrics(
        __import__('torch').tensor(dist_log['x']), target=5.0, dt=dt,
        disturbance_step=1500))


if __name__ == '__main__':
    main()
