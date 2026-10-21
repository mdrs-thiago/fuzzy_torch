"""Baseline controllers, and what they demonstrate about the plant.

A controller that drives the plant to its setpoint with the response theory predicts
is *behavioural* evidence the dynamics are wired correctly -- complementary to the
analytical invariants in test_envs_crane.py, which test the open-loop physics.
"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")

from fuzzy_torch.envs import (  # noqa: E402
    CranePlant, disturbance_rejection, disturbance_metrics, step_response,
    step_response_metrics,
)
from fuzzy_torch.envs.controllers import PID, StateFeedback, crane_baseline  # noqa: E402
from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv  # noqa: E402
from fuzzy_torch.envs.core.wrappers import ErrorFeatures  # noqa: E402


def run(scenario, controller, plant=None, seed=0):
    """Roll one episode, returning tracked position and applied control."""
    env = ErrorFeatures(FuzzyControlEnv(plant or CranePlant(), scenario=scenario))
    obs, _ = env.reset(seed=seed)
    controller.reset()
    xs, us, thetas = [], [], []
    for _ in range(scenario.episode_steps):
        obs, _, _, _, info = env.step(controller(obs))
        state = env.unwrapped._state[0]
        xs.append(float(state[0]))
        thetas.append(float(state[2]))
        us.append(float(info['control'][0]))
    import torch
    return torch.tensor(xs), torch.tensor(thetas), torch.tensor(us)


# ------------------------------------------------------------------- PID unit

def test_pid_is_proportional_to_error():
    pid = PID(kp=2.0, dt=0.01)
    assert pid(np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32))[0] == pytest.approx(6.0)


def test_pid_derivative_uses_the_measured_rate():
    """Reading e_dot from the observation avoids derivative kick on a setpoint step."""
    pid = PID(kp=0.0, kd=3.0, dt=0.01)
    assert pid(np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32))[0] == pytest.approx(6.0)


def test_pid_integral_accumulates_and_resets():
    pid = PID(kp=0.0, ki=1.0, dt=0.1)
    obs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    outs = [float(pid(obs)[0]) for _ in range(3)]
    assert outs == pytest.approx([0.1, 0.2, 0.3])
    pid.reset()
    assert float(pid(obs)[0]) == pytest.approx(0.1)


def test_pid_clamps_to_the_output_limit():
    pid = PID(kp=100.0, dt=0.01, output_limit=5.0)
    assert abs(float(pid(np.array([10.0, 0, 0, 0], dtype=np.float32))[0])) == 5.0


def test_pid_does_not_wind_up_while_saturated():
    """A saturated integrator that kept accumulating would overshoot badly on release."""
    pid = PID(kp=0.0, ki=10.0, dt=0.1, output_limit=1.0)
    push = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(50):
        pid(push)
    wound = pid._integral
    assert wound * 10.0 < 50.0 * 5.0 * 0.1 * 10.0, "integral wound up while saturated"


def test_pid_acts_on_the_selected_channel():
    pid = PID(kp=1.0, channel=1, dt=0.01)
    assert pid(np.array([9.0, 9.0, 2.0, 0.0], dtype=np.float32))[0] == pytest.approx(2.0)


# ---------------------------------------------------------- StateFeedback unit

def test_state_feedback_is_a_linear_map():
    ctrl = StateFeedback(gains=[[1.0, 2.0, 3.0, 4.0]])
    obs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert ctrl(obs)[0] == pytest.approx(10.0)


def test_state_feedback_clips_per_actuator():
    ctrl = StateFeedback(gains=[[100.0, 0, 0, 0], [0, 100.0, 0, 0]],
                         output_limits=[5.0, 2.0])
    out = ctrl(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert out == pytest.approx([5.0, 2.0])


def test_state_feedback_rejects_a_wrong_sized_observation():
    with pytest.raises(ValueError, match="expects 4 features"):
        StateFeedback(gains=[[1.0, 1.0, 1.0, 1.0]])(np.zeros(3, dtype=np.float32))


def test_state_feedback_rejects_mismatched_limits():
    with pytest.raises(ValueError, match="output_limits"):
        StateFeedback(gains=[[1.0, 1.0]], output_limits=[1.0, 2.0])


# ----------------------------------------------- closed-loop behaviour on the plant

def test_baseline_drives_the_crane_to_its_target():
    xs, thetas, _ = run(step_response(1500, [5.0, 0.0]), crane_baseline())
    assert float(xs[-1]) == pytest.approx(5.0, abs=0.1)
    assert abs(float(thetas[-1])) < 0.02, "payload still swinging at the end"


def test_baseline_damps_the_sway_it_creates():
    """Sway must peak during the move and then decay -- the anti-sway terms working."""
    _, thetas, _ = run(step_response(1500, [5.0, 0.0]), crane_baseline())
    peak = float(thetas.abs().max())
    tail = float(thetas[-200:].abs().max())
    assert peak > 0.02, "no sway excited at all; check the coupling"
    assert tail < peak / 10.0


def test_response_is_causal_and_moves_the_right_way():
    xs, _, us = run(step_response(600, [5.0, 0.0]), crane_baseline())
    assert float(us[0]) > 0, "positive position error must push the trolley +x"
    assert float(xs[10]) > 0


def test_negative_target_mirrors_the_response():
    """The plant is symmetric, so -5 m must mirror +5 m."""
    pos, _, _ = run(step_response(1200, [5.0, 0.0]), crane_baseline())
    neg, _, _ = run(step_response(1200, [-5.0, 0.0]), crane_baseline())
    assert float((pos + neg).abs().max()) < 1e-4


def test_metrics_report_a_sane_step_response():
    xs, _, us = run(step_response(1500, [5.0, 0.0]), crane_baseline())
    m = step_response_metrics(xs, target=5.0, dt=0.01, initial=0.0, control=us)
    assert m.steady_state_error < 0.05
    assert 0.0 < m.rise_time < 6.0
    assert m.overshoot < 60.0
    assert m.peak_control <= CranePlant().max_force + 1e-6


def test_higher_damping_gain_reduces_overshoot():
    """A monotone, physically expected trend: more derivative action, less overshoot."""
    def overshoot(kd):
        ctrl = StateFeedback(gains=[[6.0, kd, -40.0, -25.0]], output_limits=[30.0])
        xs, _, _ = run(step_response(1500, [5.0, 0.0]), ctrl)
        return step_response_metrics(xs, target=5.0, dt=0.01).overshoot
    assert overshoot(14.0) < overshoot(6.0)


def test_controller_recovers_from_a_scripted_disturbance():
    sc = disturbance_rejection(2500, [5.0, 0.0], time=15.0, magnitude=25.0, duration=0.2)
    xs, _, _ = run(sc, crane_baseline())
    d = disturbance_metrics(xs, target=5.0, dt=0.01, disturbance_step=1500)
    assert d['peak_deviation'] > 0.05, "disturbance had no measurable effect"
    assert float(abs(xs[-1] - 5.0)) < 0.15, "never recovered to the setpoint"


def test_uncontrolled_plant_does_not_reach_the_target():
    """Sanity on the test itself: zero control must fail, or the above proves nothing."""
    class Zero:
        def reset(self): pass
        def __call__(self, obs): return np.zeros(1, dtype=np.float32)
    xs, _, _ = run(step_response(1500, [5.0, 0.0]), Zero())
    assert abs(float(xs[-1]) - 5.0) > 4.0
