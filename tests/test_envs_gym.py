"""The Gymnasium layer: API conformance, wrappers, scenarios, metrics."""
import math

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from fuzzy_torch.envs import (  # noqa: E402
    ControlMode, CranePlant, Disturbance, Scenario, disturbance_rejection,
    disturbance_metrics, step_response, step_response_metrics,
)
from fuzzy_torch.envs.core.gym_env import FuzzyControlEnv  # noqa: E402
from fuzzy_torch.envs.core.wrappers import ErrorFeatures  # noqa: E402


def make_env(**kw):
    kw.setdefault('episode_steps', 200)
    plant_kw = {k: kw.pop(k) for k in ('mode', 'hoist') if k in kw}
    return FuzzyControlEnv(CranePlant(**plant_kw), **kw)


# ------------------------------------------------------------- conformance

@pytest.mark.parametrize("mode", [ControlMode.FORCE, ControlMode.VELOCITY])
@pytest.mark.parametrize("hoist", [False, True])
def test_passes_gymnasium_env_checker(mode, hoist):
    check_env(make_env(mode=mode, hoist=hoist), skip_render_check=True)


@pytest.mark.parametrize("hoist", [False, True])
def test_space_shapes_follow_the_plant(hoist):
    env = make_env(hoist=hoist)
    assert env.observation_space.shape == (6 if hoist else 4,)
    assert env.action_space.shape == (2 if hoist else 1,)


def test_step_before_reset_is_an_error():
    with pytest.raises(RuntimeError, match="before reset"):
        make_env().step(np.zeros(1, dtype=np.float32))


def test_episode_truncates_at_fixed_length_and_never_terminates():
    """Fixed horizon, no early termination, so IAE spans the same window for every
    controller. An env that bailed out early would flatter unstable controllers."""
    env = make_env(episode_steps=50)
    env.reset(seed=0)
    for i in range(50):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert not terminated
        assert truncated == (i == 49)


def test_actions_are_clipped_to_the_action_space():
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1e6], dtype=np.float32))
    assert abs(info['control'][0]) <= env.unwrapped.plant.max_force + 1e-6


def test_same_seed_reproduces_the_episode():
    scenario = Scenario(name='rand', episode_steps=50, setpoint=[5.0, 0.0],
                        randomize_initial_state=True)
    outs = []
    for _ in range(2):
        env = FuzzyControlEnv(CranePlant(), scenario=scenario)
        obs, _ = env.reset(seed=1234)
        traj = [obs]
        for _ in range(20):
            traj.append(env.step(np.array([1.0], dtype=np.float32))[0])
        outs.append(np.stack(traj))
    assert np.array_equal(outs[0], outs[1])


def test_different_seeds_give_different_initial_states():
    scenario = Scenario(name='rand', episode_steps=10, randomize_initial_state=True)
    env = FuzzyControlEnv(CranePlant(), scenario=scenario)
    assert not np.allclose(env.reset(seed=1)[0], env.reset(seed=2)[0])


# ----------------------------------------------------------------- wrapper

def test_error_features_expose_errors_not_raw_state():
    env = ErrorFeatures(FuzzyControlEnv(CranePlant(), scenario=step_response(100, [5.0, 0.0])))
    obs, _ = env.reset(seed=0)
    assert env.feature_names == ['x_error', 'x_error_dot', 'theta_error', 'theta_error_dot']
    # Trolley starts at 0 with a target of 5, and hangs still.
    assert obs[0] == pytest.approx(5.0)
    assert obs[1] == pytest.approx(0.0)


def test_error_features_derivative_is_the_negated_state_rate():
    env = ErrorFeatures(FuzzyControlEnv(CranePlant(), scenario=step_response(100, [5.0, 0.0])))
    env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.array([20.0], dtype=np.float32))
    raw = env.unwrapped._state[0]
    assert obs[1] == pytest.approx(-float(raw[1]), rel=1e-4)
    assert obs[3] == pytest.approx(-float(raw[3]), rel=1e-4)


def test_error_features_passes_the_env_checker():
    check_env(ErrorFeatures(make_env()), skip_render_check=True)


def test_wrapper_gives_both_controllers_the_same_observation():
    """The fairness protocol is structural: RL and the FLC read one wrapper."""
    env = ErrorFeatures(make_env())
    a, _ = env.reset(seed=7)
    b, _ = env.reset(seed=7)
    assert np.array_equal(a, b)
    assert env.observation_space.shape == (4,)


# ---------------------------------------------------------------- scenarios

def test_setpoint_steps_at_the_scripted_time():
    sc = step_response(100, setpoint=[5.0, 0.0], step_time=0.2)
    env = FuzzyControlEnv(CranePlant(), scenario=sc)
    env.reset(seed=0)
    assert float(env.setpoint()[0]) == 0.0
    for _ in range(19):
        env.step(np.zeros(1, dtype=np.float32))
    assert float(env.setpoint()[0]) == 0.0     # t = 0.19 s, still before the step
    env.step(np.zeros(1, dtype=np.float32))
    assert float(env.setpoint()[0]) == 5.0     # t = 0.20 s


def test_disturbance_fires_at_the_scripted_time_and_magnitude():
    sc = disturbance_rejection(200, setpoint=[0.0, 0.0], time=0.5,
                               magnitude=25.0, duration=0.1)
    env = FuzzyControlEnv(CranePlant(), scenario=sc)
    env.reset(seed=0)
    seen = []
    for _ in range(100):
        _, _, _, _, info = env.step(np.zeros(1, dtype=np.float32))
        seen.append(info['disturbance'])
    active = [i for i, d in enumerate(seen) if d is not None]
    # Steps are taken at t = 0, dt, ... so the window [0.5, 0.6) covers steps 50..59.
    assert active == list(range(50, 60))
    assert all(seen[i][0] == 25.0 for i in active)


def test_impulse_disturbance_lasts_exactly_one_step():
    sc = Scenario(name='impulse', episode_steps=100,
                  disturbances=[Disturbance(time=0.1, magnitude=10.0)])
    env = FuzzyControlEnv(CranePlant(), scenario=sc)
    env.reset(seed=0)
    n = sum(env.step(np.zeros(1, dtype=np.float32))[4]['disturbance'] is not None
            for _ in range(50))
    assert n == 1


def test_disturbance_actually_moves_the_plant():
    quiet = FuzzyControlEnv(CranePlant(), scenario=step_response(120, [0.0, 0.0]))
    hit = FuzzyControlEnv(CranePlant(),
                          scenario=disturbance_rejection(120, [0.0, 0.0], time=0.2,
                                                         magnitude=30.0, duration=0.1))
    a = b = None
    for env in (quiet, hit):
        env.reset(seed=3)
        for _ in range(120):
            obs = env.step(np.zeros(1, dtype=np.float32))[0]
        if a is None:
            a = obs
        else:
            b = obs
    assert abs(b[0] - a[0]) > 1e-3


# ------------------------------------------------------------------ metrics

def test_step_metrics_on_an_ideal_first_order_response():
    """1 - exp(-t/tau) has no overshoot, zero steady-state error, and a rise time
    of tau*ln(9) between the 10% and 90% points."""
    dt, tau = 1e-3, 0.5
    t = torch.arange(0, 10.0, dt)
    y = 1.0 - torch.exp(-t / tau)
    m = step_response_metrics(y, target=1.0, dt=dt, initial=0.0)
    assert m.overshoot == 0.0
    assert m.steady_state_error == pytest.approx(0.0, abs=1e-3)
    assert m.rise_time == pytest.approx(tau * math.log(9.0), rel=0.02)
    # IAE of exp(-t/tau) over a long horizon is tau.
    assert m.iae == pytest.approx(tau, rel=1e-3)


def test_step_metrics_detect_overshoot():
    dt = 1e-3
    t = torch.arange(0, 6.0, dt)
    y = 1.0 - torch.exp(-t) * torch.cos(6.0 * t)     # ~overshoots above 1
    m = step_response_metrics(y, target=1.0, dt=dt)
    assert m.overshoot > 5.0
    assert m.settling_time > 0.0


def test_settling_time_is_the_last_exit_not_the_first_entry():
    """A response that re-leaves the band later has not settled at the first entry."""
    dt = 0.01
    y = torch.ones(1000)
    y[500:520] = 1.5                       # a late excursion outside the band
    m = step_response_metrics(y, target=1.0, dt=dt)
    assert m.settling_time == pytest.approx(520 * dt, abs=dt)


def test_perfect_tracking_settles_immediately():
    m = step_response_metrics(torch.ones(500), target=1.0, dt=0.01, initial=1.0)
    assert m.settling_time == 0.0
    assert m.iae == pytest.approx(0.0, abs=1e-12)


def test_control_effort_is_integrated():
    m = step_response_metrics(torch.ones(101), target=1.0, dt=0.01,
                              control=torch.full((101,), 2.0))
    assert m.control_effort == pytest.approx(2.0, rel=1e-9)   # 2 N over 1 s
    assert m.peak_control == pytest.approx(2.0)


def test_disturbance_metrics_measure_only_after_the_hit():
    dt = 0.01
    y = torch.zeros(400)
    y[200:250] = 0.4                                    # deviation after the hit
    d = disturbance_metrics(y, target=0.0, dt=dt, disturbance_step=200)
    assert d['peak_deviation'] == pytest.approx(0.4)
    assert d['recovery_time'] == pytest.approx(50 * dt, abs=dt)


def test_integral_metrics_are_batched():
    from fuzzy_torch.envs import integral_metrics
    err = torch.ones(5, 100)
    out = integral_metrics(err, dt=0.01)
    assert out['iae'].shape == (5,)
    assert torch.allclose(out['iae'], torch.full((5,), 0.99))  # trapezoid over 99 gaps


# ----------------------------------------------------------------- rendering

def test_rgb_array_render_produces_an_image():
    env = make_env(render_mode='rgb_array')
    env.reset(seed=0)
    env.step(np.array([5.0], dtype=np.float32))
    frame = env.render()
    # Derived from whichever backend was chosen, not hardcoded, so changing a canvas
    # size is not a test failure -- only a *wrong-shaped* frame is.
    backend = env.unwrapped._renderer
    assert frame.shape == (backend.height, backend.width, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 0        # something was actually drawn
    env.close()


def test_render_is_none_without_a_render_mode():
    env = make_env()
    env.reset(seed=0)
    assert env.render() is None


def test_info_separates_commanded_control_from_disturbance():
    """Control-effort metrics must not bill the controller for the disturbance."""
    sc = disturbance_rejection(200, [0.0, 0.0], time=0.1, magnitude=25.0, duration=0.05)
    env = FuzzyControlEnv(CranePlant(), scenario=sc)
    env.reset(seed=0)
    hit = None
    for _ in range(30):
        _, _, _, _, info = env.step(np.array([4.0], dtype=np.float32))
        if info['disturbance'] is not None:
            hit = info
            break
    assert hit is not None
    assert hit['control'][0] == pytest.approx(4.0)          # only what we commanded
    assert hit['applied'][0] == pytest.approx(29.0)         # commanded + 25 N
    assert hit['applied'][0] == pytest.approx(hit['control'][0] + hit['disturbance'][0])


def test_applied_equals_control_when_undisturbed():
    env = FuzzyControlEnv(CranePlant(), scenario=step_response(50, [1.0, 0.0]))
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([3.0], dtype=np.float32))
    assert info['disturbance'] is None
    assert info['applied'] == pytest.approx(info['control'])


def test_pygame_backend_works_without_a_display():
    """The fallback must render headless -- that is the reason it still exists."""
    env = make_env(render_mode='rgb_array')
    env.unwrapped.render_backend = 'pygame'
    env.reset(seed=0)
    env.step(np.array([5.0], dtype=np.float32))
    frame = env.render()
    from fuzzy_torch.envs.core.render import CranePygameRenderer, PygameRenderer
    assert isinstance(env.unwrapped._renderer, PygameRenderer)
    assert frame.shape == (CranePygameRenderer.height, CranePygameRenderer.width, 3)
    env.close()


def test_unknown_render_backend_is_rejected():
    from fuzzy_torch.envs.core.render import make_renderer
    with pytest.raises(ValueError, match="unknown backend"):
        make_renderer(CranePlant(), 'rgb_array', backend='raytracer')


def test_explicit_mujoco_backend_reports_why_it_cannot_run():
    """Asking for MuJoCo explicitly must fail loudly, not silently downgrade."""
    from fuzzy_torch.envs.core.render import make_renderer, mujoco_available
    ok, _ = mujoco_available(CranePlant())
    if ok:
        r = make_renderer(CranePlant(), 'rgb_array', backend='mujoco')
        from fuzzy_torch.envs.core.render import MujocoRenderer
        assert isinstance(r, MujocoRenderer)
        r.close()
    else:
        with pytest.raises(RuntimeError, match="MuJoCo backend unavailable"):
            make_renderer(CranePlant(), 'rgb_array', backend='mujoco')
