"""Record an episode headlessly and show it inline in a notebook.

    video = record(RacingEnv('oval', render_mode='rgb_array'), driver)
    video              # last expression of a notebook cell -> animated playback
    video.save('lap.gif')

The environment must be created with ``render_mode='rgb_array'``. Rendering uses
MuJoCo when it can open an OpenGL context and falls back to pygame otherwise (see
:func:`~fuzzy_torch.envs.core.render.make_renderer`), so the same code runs on a
desktop, a headless server or Colab.

Videos are encoded with Pillow, so no ffmpeg is needed. Inline playback uses animated
WebP, about four times smaller than GIF for the textured MuJoCo scenes and supported by
every current browser-based notebook front end; ``save()`` writes GIF or WebP by
extension.
"""
import base64
import io
import pathlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class Video:
    """Frames of one episode, plus what the environment reported along the way.

    Attributes:
        frames: list of ``(H, W, 3)`` uint8 images.
        fps: playback rate of the encoded animation.
        observations, actions, rewards, infos: one entry per environment step.
        states: the plant's raw state after each step, as ``(steps, state_dim)``
            once recorded (empty if the environment does not expose one).
    """

    def __init__(self, frames: List[np.ndarray], fps: float = 25.0, width: int = 640):
        self.frames = [f for f in frames if f is not None]
        self.fps = float(fps)
        self.width = int(width)
        self.observations: List[Any] = []
        self.actions: List[Any] = []
        self.rewards: List[float] = []
        self.infos: List[Dict[str, Any]] = []
        self.states = np.zeros((0, 0))
        self._encoded: Dict[str, bytes] = {}

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def info(self) -> Dict[str, Any]:
        """The ``info`` dict of the last step (e.g. ``lap_time`` for a race)."""
        return self.infos[-1] if self.infos else {}

    def _images(self):
        if not self.frames:
            raise ValueError("no frames recorded -- was the env made with render_mode='rgb_array'?")
        from PIL import Image
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in self.frames]
        if images[0].width > self.width:
            height = round(images[0].height * self.width / images[0].width)
            images = [im.resize((self.width, height), Image.LANCZOS) for im in images]
        return images

    def encode(self, fmt: str = 'webp', quality: int = 70) -> bytes:
        """The animation as ``'webp'`` (lossy, small) or ``'gif'`` bytes."""
        fmt = fmt.lower()
        if fmt in self._encoded:
            return self._encoded[fmt]
        images = self._images()
        duration = max(20, int(round(1000 / self.fps)))
        buffer = io.BytesIO()
        if fmt == 'webp':
            images[0].save(buffer, format='WEBP', save_all=True, append_images=images[1:],
                           duration=duration, loop=0, quality=quality, method=4)
        elif fmt == 'gif':
            from PIL import Image
            # One adaptive palette for the whole clip keeps the file small and stops
            # colours flickering between frames.
            palette = images[len(images) // 2].quantize(colors=192, method=Image.MEDIANCUT)
            images = [im.quantize(palette=palette, dither=Image.NONE) for im in images]
            images[0].save(buffer, format='GIF', save_all=True, append_images=images[1:],
                           duration=duration, loop=0, optimize=True)
        else:
            raise ValueError(f"unknown format '{fmt}' (use 'webp' or 'gif')")
        self._encoded[fmt] = buffer.getvalue()
        return self._encoded[fmt]

    def save(self, path) -> pathlib.Path:
        """Write the animation to ``path``; ``.gif`` or ``.webp`` picks the format."""
        path = pathlib.Path(path)
        path.write_bytes(self.encode(path.suffix.lstrip('.') or 'webp'))
        return path

    def _repr_html_(self) -> str:
        if not self.frames:
            return '<em>(played live -- no frames recorded)</em>'
        data = base64.b64encode(self.encode('webp')).decode('ascii')
        return (f'<img src="data:image/webp;base64,{data}" '
                f'style="max-width:100%; width:{self.width}px"/>')

    def __repr__(self) -> str:
        return f'Video({len(self.frames)} frames @ {self.fps:g} fps)'


def record(env, policy: Callable, *, seed: Optional[int] = 0, max_steps: Optional[int] = None,
           every: int = 4, fps: Optional[float] = None, width: int = 640) -> Video:
    """Run one episode of ``env`` under ``policy`` and capture it as a :class:`Video`.

    Args:
        env: a Gymnasium environment (wrappers are fine) made with
            ``render_mode='rgb_array'`` to capture frames, or ``'human'`` to watch it
            live in a window instead (then no frames are kept).
        policy: ``action = policy(observation)``. If it has a ``reset()`` method,
            that is called first.
        seed: passed to ``env.reset``.
        max_steps: stop after this many steps even if the episode is not over.
        every: keep one frame every ``every`` steps (the first and last frames are
            always kept).
        fps: playback rate; defaults to real time, ``1 / (every * dt)``.
        width: width of the encoded animation in pixels.

    The returned video also holds every observation, action, reward and info dict,
    so you can plot the episode afterwards.
    """
    mode = getattr(env, 'render_mode', None)
    if mode not in ('rgb_array', 'human'):
        raise ValueError("record() needs an environment made with render_mode='rgb_array' "
                         "(or 'human' to watch live)")
    capture = mode == 'rgb_array'
    every = max(1, int(every))
    dt = float(getattr(env.unwrapped, 'dt', 0.02))
    video = Video([], fps=fps or min(50.0, 1.0 / (every * dt)), width=width)

    if hasattr(policy, 'reset'):
        policy.reset()
    obs, _ = env.reset(seed=seed)
    if capture:
        video.frames.append(env.render())
    step, states = 0, []
    while max_steps is None or step < max_steps:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        video.observations.append(obs)
        video.actions.append(np.asarray(action))
        video.rewards.append(float(reward))
        video.infos.append(info)
        state = getattr(env.unwrapped, '_state', None)
        if state is not None:
            states.append(np.asarray(state[0].detach().cpu() if hasattr(state, 'detach')
                                     else state[0], dtype=np.float64))
        done = terminated or truncated
        if capture and (step % every == 0 or done):
            video.frames.append(env.render())
        if done:
            break
        renderer = getattr(env.unwrapped, '_renderer', None)
        if not capture and renderer is not None and not getattr(renderer, 'is_running',
                                                                 getattr(renderer, 'running', True)):
            break                                   # the live window was closed
    env.close()
    if states:
        video.states = np.stack(states)
    return video


class _Row:
    def __init__(self, videos: Dict[str, Video]):
        self.videos = videos

    def _repr_html_(self) -> str:
        cells = ''.join(
            f'<div style="flex:1 1 0; min-width:220px; text-align:center">'
            f'<div style="font-weight:600; margin-bottom:4px">{name}</div>'
            f'{video._repr_html_().replace(f"width:{video.width}px", "width:100%")}</div>'
            for name, video in self.videos.items())
        return f'<div style="display:flex; gap:12px; flex-wrap:wrap">{cells}</div>'

    def __repr__(self) -> str:
        return f'compare({", ".join(self.videos)})'


def compare(**videos: Video):
    """Show several videos side by side in a notebook: ``compare(pid=v1, fuzzy=v2)``.

    Use a dict for titles with spaces: ``compare(**{'state feedback': v1})``.
    """
    return _Row(videos)
