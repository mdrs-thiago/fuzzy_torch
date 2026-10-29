"""Contract tests for the SuperTuxKart wrapper (skipped unless pystk2 is installed).

pystk2 starts one game per process, so these tests only ever use it headless.
"""
import json
import pathlib

import numpy as np
import pytest

pytest.importorskip('gymnasium')
pytest.importorskip('pystk2')

from fuzzy_torch.envs import RacingEnv, SuperTuxKartEnv  # noqa: E402
from fuzzy_torch.envs.controllers import ConfigRaceDriver  # noqa: E402

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / 'examples/race_control/example_driver.json'


def test_shares_the_racing_interface():
    env = SuperTuxKartEnv('lighthouse')
    obs, info = env.reset(seed=0)
    assert env.feature_names == RacingEnv.feature_names
    assert env.action_space.shape == (2,)
    assert env.observation_space.contains(obs)
    # The race starts on the road, pointing down it, at the start signal.
    assert abs(obs[0]) < 1.0 and abs(obs[1]) < 0.2 and obs[5] == 0.0
    assert info['lap_time'] is None
    env.close()


def test_positive_steering_turns_left_and_raises_yaw_rate():
    env = SuperTuxKartEnv('lighthouse')
    env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step([1.0, 1.0])
    assert obs[3] > 0.5
    env.close()


def test_the_leaderboard_example_driver_finishes_a_race():
    env = SuperTuxKartEnv('zengarden', opponents=1)
    driver = ConfigRaceDriver(json.loads(EXAMPLE.read_text()))
    obs, _ = env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(driver(obs))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated and info['race_time'] is not None
    assert info['progress'] == pytest.approx(env.track.length, rel=0.05)
    assert info['position'] in (1, 2) and info['num_karts'] == 2
    env.close()


def test_a_second_environment_takes_the_game_over():
    first, second = SuperTuxKartEnv('lighthouse'), SuperTuxKartEnv('zengarden')
    first.reset(seed=0)
    second.reset(seed=0)
    with pytest.raises(RuntimeError, match='took over'):
        first.step([0.0, 0.0])
    second.step([1.0, 0.0])
    second.close()


def test_rejects_unknown_tracks_and_bad_actions():
    with pytest.raises(ValueError, match='unknown SuperTuxKart track'):
        SuperTuxKartEnv('rainbow_road')
    env = SuperTuxKartEnv('lighthouse', extra_actions=True)
    env.reset(seed=0)
    env.step([1.0, 0.0, 0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match='shape'):
        env.step([1.0, 0.0])
    env.close()


def test_kart_race_gives_every_driver_its_own_kart():
    from fuzzy_torch.envs import KartRace
    spec = json.loads(EXAMPLE.read_text())

    def broken(obs):
        raise RuntimeError('oops')

    race = KartRace({'Ana': ConfigRaceDriver(spec), 'Bruno': ConfigRaceDriver(spec),
                     'Broken': broken}, 'lighthouse', ai_karts=1)
    observations = race.reset(seed=0)
    assert set(observations) == {'Ana', 'Bruno', 'Broken'}
    for _ in range(200):
        race.step()
    rows = race.standings()
    assert [r['position'] for r in rows] == [1, 2, 3, 4]
    assert sum(r['ai'] for r in rows) == 1
    by_name = {r['name']: r for r in rows}
    # The two working drivers drive off; the broken one coasts and is reported.
    assert by_name['Ana']['progress'] > 0.05 and by_name['Bruno']['progress'] > 0.05
    assert by_name['Broken']['progress'] < 0.02 and 'oops' in by_name['Broken']['error']
    race.close()


def test_race_director_renders_a_heat(tmp_path):
    # Graphics can only start in a fresh process, so the window is drawn in one.
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[1]
    shot = tmp_path / 'director.png'
    result = subprocess.run(
        [sys.executable, str(root / 'examples/kart_party.py'), '--driver', str(EXAMPLE),
         '--ai', '1', '--screenshot', str(shot), '--steps', '40'],
        capture_output=True, text=True, timeout=600,
        env={**__import__('os').environ, 'PYTHONPATH': str(root.parent)})
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    from PIL import Image
    image = np.asarray(Image.open(shot).convert('RGB'))
    assert image.shape == (720, 1280, 3)
    assert image[:500, :800].std() > 10          # the camera shows the game, not a blank


@pytest.mark.skipif(not (__import__('os').environ.get('DISPLAY')
                         or __import__('os').environ.get('WAYLAND_DISPLAY')),
                    reason='needs a desktop to open a window')
def test_window_process_shows_frames():
    from fuzzy_torch.envs.supertuxkart import Window
    window = Window(64, 48, 'test')
    try:
        for shade in (0, 128, 255):
            window.show(np.full((48, 64, 3), shade, np.uint8))
        assert window.running
    finally:
        window.close()


def test_kart_party_takes_each_students_latest_valid_submission(tmp_path):
    import importlib.util
    path = pathlib.Path(__file__).resolve().parents[1] / 'examples/kart_party.py'
    spec = importlib.util.spec_from_file_location('kart_party', path)
    kart_party = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kart_party)
    store = tmp_path / 'leaderboard.json'
    store.write_text(json.dumps([
        {'name': 'Ana Souza', 'submitted_at': '2026-10-29T10:00:00', 'status': 'done', 'spec': 'old'},
        {'name': 'ana  souza', 'submitted_at': '2026-10-29T10:05:00', 'status': 'done', 'spec': 'new'},
        {'name': 'Bruno', 'submitted_at': '2026-10-29T10:01:00', 'status': 'done', 'spec': 'ok'},
        {'name': 'Bruno', 'submitted_at': '2026-10-29T10:06:00', 'status': 'error', 'spec': 'bad'},
    ]))
    assert kart_party.from_store(store) == {'ana souza': 'new', 'Bruno': 'ok'}
