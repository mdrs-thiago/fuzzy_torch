"""Contract tests for the closed-circuit fuzzy racing task."""
import numpy as np
import pytest

pytest.importorskip('gymnasium')

from fuzzy_torch.envs import RacingEnv, make_track  # noqa: E402
from fuzzy_torch.envs.racing import _race_mjcf  # noqa: E402
from fuzzy_torch.envs.controllers import FuzzyRaceDriver  # noqa: E402


@pytest.mark.parametrize('name', ['oval', 'chicane', 'technical'])
def test_circuits_are_closed_and_have_a_positive_length(name):
    track = make_track(name)
    assert track.length > 20.0
    point, _ = track.ahead(track.length)
    assert point == pytest.approx(track.points[0])


def test_reset_starts_on_the_centre_line_with_named_features():
    env = RacingEnv('oval')
    obs, info = env.reset(seed=7)
    assert env.feature_names == ('cross_track_error', 'heading_error', 'speed', 'yaw_rate',
                                 'lookahead_curvature', 'progress')
    assert env.observation_space.contains(obs)
    assert abs(obs[0]) < 1e-5
    assert abs(obs[1]) < 1e-5
    assert info['lap_time'] is None


def test_start_pose_and_finish_geometry_are_normal_to_track_flow():
    track = make_track('oval')
    env = RacingEnv('oval')
    env.reset(seed=7)
    _, _, tangent, _ = track.project(env._state[0, :2].numpy())
    assert env._state[0, 2].item() == pytest.approx(tangent)

    # The MJCF finish tiles and gantry have their short local x-axis along the
    # street and their long local y-axis across it.
    xml = _race_mjcf(track)
    assert f'euler="0 0 {tangent:.6f}" size=".10 ' in xml
    assert f'euler="0 0 {tangent:.6f}" size=".08 ' in xml


def test_action_is_clipped_and_info_reports_race_metrics():
    env = RacingEnv('oval')
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.array([20.0, -20.0]))
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert not truncated
    assert isinstance(terminated, bool)
    assert set(('lap_time', 'progress', 'cross_track_error', 'off_track', 'speed')) <= set(info)
    assert info['speed'] <= env.vehicle.max_speed * 1.2


def test_fuzzy_driver_returns_a_valid_race_action():
    env = RacingEnv('technical')
    obs, _ = env.reset(seed=0)
    action = FuzzyRaceDriver()(obs)
    assert action.shape == (2,)
    assert env.action_space.contains(action)


# ---------------------------------------------------------------- JSON drivers

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

from fuzzy_torch.config import ConfigError  # noqa: E402
from fuzzy_torch.envs.controllers import ConfigRaceDriver  # noqa: E402

RACE_CONTROL = pathlib.Path(__file__).resolve().parents[1] / 'examples' / 'race_control'


def test_json_driver_reproduces_the_baseline_actions():
    baseline = FuzzyRaceDriver()
    driver = ConfigRaceDriver(json.loads(json.dumps(baseline.to_config())))
    rng = np.random.default_rng(0)
    low, high = RacingEnv('oval').observation_space.low, RacingEnv('oval').observation_space.high
    for obs in rng.uniform(low, high, size=(200, 6)).astype(np.float32):
        assert np.allclose(driver(obs), baseline(obs), atol=1e-4)


def test_json_driver_rejects_missing_systems_and_unknown_inputs():
    spec = FuzzyRaceDriver().to_config()
    with pytest.raises(ConfigError, match='steering: missing'):
        ConfigRaceDriver({'throttle': spec['throttle']})
    spec['throttle']['inputs']['x_position'] = spec['throttle']['inputs'].pop('speed')
    with pytest.raises(ConfigError, match='throttle.inputs.x_position: unknown input'):
        ConfigRaceDriver(spec)


@pytest.fixture
def grand_prix():
    sys.path.insert(0, str(RACE_CONTROL))
    try:
        import grand_prix
        yield grand_prix
    finally:
        sys.path.remove(str(RACE_CONTROL))


def test_example_driver_file_finishes_a_lap(grand_prix):
    spec = json.loads((RACE_CONTROL / 'example_driver.json').read_text())
    lap = grand_prix.run_lap(grand_prix.build_driver(spec), 'oval')
    assert lap['status'] == 'FINISHED'
    assert 10.0 < lap['time'] < 25.0
    assert lap['path'][0][:2] == pytest.approx([8.0, 0.0], abs=1e-3)


def test_a_parked_kart_is_retired_early(grand_prix):
    spec = json.loads((RACE_CONTROL / 'example_driver.json').read_text())
    for rule in spec['throttle']['rules']:
        rule['then'] = 'brake'
    spec['throttle']['output']['terms']['brake'] = ['tri', 0, 0, 0.001]
    lap = grand_prix.run_lap(grand_prix.build_driver(spec), 'oval')
    assert lap['status'] == 'DNF' and lap['detail'] == 'Stalled'
    # Retired after the stall window, not the 45 s time limit.
    assert len(lap['path']) * grand_prix.SAMPLE_DT < grand_prix.STALL_WINDOW + 1.0
