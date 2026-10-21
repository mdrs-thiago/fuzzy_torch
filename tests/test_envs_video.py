"""Headless episode recording for notebooks."""
import pytest

pytest.importorskip('gymnasium')
pytest.importorskip('pygame')
pytest.importorskip('PIL')

from fuzzy_torch.envs import (  # noqa: E402
    CranePlant, FuzzyControlEnv, ErrorFeatures, RacingEnv, Video, record, step_response,
)
from fuzzy_torch.envs.controllers import FuzzyRaceDriver, crane_baseline  # noqa: E402


def _crane_env(render_mode='rgb_array', steps=40):
    return ErrorFeatures(FuzzyControlEnv(CranePlant(), scenario=step_response(steps, [5.0, 0.0]),
                                         render_mode=render_mode, render_backend='pygame'))


def test_record_keeps_every_nth_frame_and_the_last():
    video = record(_crane_env(steps=40), crane_baseline(), every=8)
    # reset frame + steps 8, 16, 24, 32, 40 (the last one is also the end)
    assert len(video) == 6
    assert len(video.observations) == len(video.actions) == len(video.infos) == 40
    assert video.frames[0].ndim == 3 and video.frames[0].shape[2] == 3
    assert video.fps == pytest.approx(1 / (8 * CranePlant.default_dt))


def test_record_stops_at_max_steps():
    video = record(_crane_env(steps=40), crane_baseline(), every=1, max_steps=5)
    assert len(video.infos) == 5
    assert len(video) == 6


def test_encodings_and_notebook_display(tmp_path):
    video = record(_crane_env(steps=12), crane_baseline(), every=4, width=200)
    webp, gif = video.encode('webp'), video.encode('gif')
    assert webp[:4] == b'RIFF' and webp[8:12] == b'WEBP'
    assert gif[:6] in (b'GIF87a', b'GIF89a')
    assert 'data:image/webp;base64,' in video._repr_html_()
    assert video.save(tmp_path / 'clip.gif').read_bytes() == gif
    with pytest.raises(ValueError):
        video.encode('mp4')


def test_record_race_reports_the_lap():
    env = RacingEnv('oval', render_mode='rgb_array', render_backend='pygame')
    video = record(env, FuzzyRaceDriver(), every=50)
    assert video.info['lap_time'] is not None
    assert not video.info['off_track']


def test_record_needs_a_render_mode():
    with pytest.raises(ValueError, match='render_mode'):
        record(_crane_env(render_mode=None), crane_baseline())


def test_empty_video_cannot_encode():
    with pytest.raises(ValueError, match='no frames'):
        Video([]).encode()


def test_states_and_compare():
    from fuzzy_torch.envs import compare
    a = record(_crane_env(steps=10), crane_baseline(), every=5, width=120)
    assert a.states.shape == (10, 4)
    assert a.states[-1, 0] > 0                       # the trolley moved towards x = 5
    html = compare(**{'state feedback': a, 'again': a})._repr_html_()
    assert html.count('<img') == 2 and 'state feedback' in html
