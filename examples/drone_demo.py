"""Fly the 3-D quadrotor to a hover target with a cascaded baseline controller.

    .venv/bin/python examples/drone_demo.py --out out/
    .venv/bin/python examples/drone_demo.py --live --backend mujoco

The controller intentionally uses the public ``RATE_THRUST`` action interface:
an outer position loop requests a small roll/pitch angle, while the plant's inner
loop turns body-rate requests into physical torque.  It is a compact reference
for a fuzzy or learned controller, not a replacement for either.
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

from fuzzy_torch.envs import ControlMode, QuadrotorPlant, Scenario
from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv


def _euler_from_quaternion(q):
    """Return roll, pitch, yaw from a scalar-first body-to-world quaternion."""
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class HoverController:
    """Small-angle position-to-rate baseline for the rate/thrust action space."""

    def __init__(self, drone):
        self.m, self.g = float(drone.m), float(drone.g)
        self.max_thrust, self.max_rate = 4.0 * drone.max_rotor_thrust, drone.max_rate
        self.kp = np.array([1.25, 1.25, 2.4])
        self.kd = np.array([1.35, 1.35, 1.6])
        self.attitude_gain, self.rate_damping = 5.0, 0.35

    def __call__(self, state, target):
        position, velocity = state[:3], state[3:6]
        attitude, body_rates = _euler_from_quaternion(state[6:10]), state[10:13]
        desired_acceleration = self.kp * (target - position) - self.kd * velocity
        desired_acceleration = np.clip(desired_acceleration, [-3.0, -3.0, -4.0],
                                       [3.0, 3.0, 4.0])

        # For a near-level drone, a pitch creates +x acceleration and a roll
        # creates -y acceleration.  Holding yaw at zero keeps this map intuitive.
        desired_roll = -desired_acceleration[1] / self.g
        desired_pitch = desired_acceleration[0] / self.g
        desired_angles = np.array([desired_roll, desired_pitch, 0.0])
        desired_rates = self.attitude_gain * (desired_angles - attitude) - self.rate_damping * body_rates
        desired_rates = np.clip(desired_rates, -self.max_rate, self.max_rate)

        # Tilt reduces vertical thrust; compensate using the current body-up vector.
        up_z = max(np.cos(attitude[0]) * np.cos(attitude[1]), 0.35)
        collective = self.m * (self.g + desired_acceleration[2]) / up_z
        return np.r_[np.clip(collective, 0.0, self.max_thrust), desired_rates].astype(np.float32)


def run(*, live=False, backend='auto', duration=12.0):
    drone = QuadrotorPlant(mode=ControlMode.RATE_THRUST, target=(1.4, -1.0, 1.8))
    scenario = Scenario(name='hover', episode_steps=round(duration / drone.default_dt),
                        setpoint=list(drone.target), randomize_initial_state=True)
    env = FuzzyControlEnv(drone, scenario=scenario, render_mode='human' if live else None,
                          render_backend=backend)
    controller = HoverController(drone)
    _, _ = env.reset(seed=7)
    log = {key: [] for key in ('time', 'position', 'target', 'thrust', 'rates')}
    for _ in range(scenario.episode_steps):
        state = env._state[0].detach().cpu().numpy()
        target = env.setpoint().detach().cpu().numpy()
        _, _, _, truncated, info = env.step(controller(state, target))
        log['time'].append(info['time'])
        log['position'].append(env._state[0, :3].detach().cpu().numpy())
        log['target'].append(target)
        log['thrust'].append(info['control'][0])
        log['rates'].append(env._state[0, 10:13].detach().cpu().numpy())
        renderer = env._renderer
        if truncated or (renderer is not None and not getattr(renderer, 'is_running', True)):
            break
    env.close()
    return {key: np.asarray(value) for key, value in log.items()}


def plot(log, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col')
    labels, colors = ('x', 'y', 'z'), ('#c74343', '#3574ad', '#45935b')
    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[0, 0].plot(log['time'], log['position'][:, i], color=color, label=label)
        axes[0, 0].axhline(log['target'][0, i], color=color, ls='--', alpha=.55)
        axes[0, 1].plot(log['time'], log['rates'][:, i], color=color, label=label)
    axes[1, 0].plot(log['position'][:, 0], log['position'][:, 1], color='#343c4b')
    axes[1, 0].scatter(*log['target'][0, :2], color='#2688c2', marker='x', s=70, label='target')
    axes[1, 1].plot(log['time'], log['thrust'], color='#d68a28')
    axes[0, 0].set_ylabel('position [m]')
    axes[0, 1].set_ylabel('body rate [rad/s]')
    axes[1, 0].set(xlabel='x [m]', ylabel='y [m]', aspect='equal')
    axes[1, 1].set(xlabel='time [s]', ylabel='collective thrust [N]')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    fig.suptitle('Quadrotor hover: cascaded position and rate control')
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true', help='open the live renderer')
    parser.add_argument('--backend', choices=['auto', 'mujoco', 'pygame'], default='auto')
    parser.add_argument('--duration', type=float, default=12.0, help='flight duration in seconds')
    parser.add_argument('--out', default='out', help='directory for the plot')
    args = parser.parse_args()
    log = run(live=args.live, backend=args.backend, duration=args.duration)
    error = np.linalg.norm(log['position'][-1] - log['target'][-1])
    print(f'final position: {log["position"][-1]}; target: {log["target"][-1]}; error: {error:.3f} m')
    if not args.live:
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        target = out / 'quadrotor_hover.png'
        plot(log, target)
        print(f'wrote {target}')


if __name__ == '__main__':
    main()
