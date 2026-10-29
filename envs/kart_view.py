"""A live "race director" window for :class:`~fuzzy_torch.envs.supertuxkart.KartRace`.

It shows the chase camera of one kart next to that kart's fuzzy driver at work:
every input on its membership functions, the rules firing, and the output set and
crisp action, updated every step.  Underneath are a track map with every kart and
the live standings.  It is meant for a projector: follow any student with the arrow
keys or a click, or let it follow the leader.

Keys: ``Left``/``Right`` previous/next driver, ``L`` follow the leader, ``Space``
pause, ``F`` toggle real-time pacing, ``Enter`` next race (once finished), ``Esc`` quit.
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..system import TSK
from .controllers.racing import race_inputs, system_inputs
from .supertuxkart import Window, _pystk2

WIDTH, HEIGHT = 1280, 720
CAMERA = (800, 500)
BG, PANEL, GRID = (17, 20, 26), (27, 31, 40), (52, 58, 72)
TEXT, DIM, ACCENT, VALUE = (232, 235, 240), (140, 148, 164), (255, 196, 46), (255, 84, 84)
TERMS = [(84, 160, 255), (110, 210, 120), (255, 160, 60), (200, 120, 255), (255, 110, 160),
         (80, 210, 210), (220, 220, 90)]


class FuzzyPanel:
    """Draws one :class:`~fuzzy_torch.FuzzyModule` evaluating one input, live."""

    samples = 64

    def __init__(self, system, title: str):
        self.system, self.title = system, title
        self.inputs = [system.input_variables[n] for n in system.input_order]
        with torch.no_grad():
            self.curves = [self._curves(var, var.range_min, var.range_max) for var in self.inputs]
            out = system.output_variable
            self.output_curves = self._curves(out, out.range_min, out.range_max) if out.terms else None
        self.labels = [self._rule_label(rule) for rule in system.rules]

    def _curves(self, var, lo: float, hi: float):
        xs = torch.linspace(lo, hi, self.samples)
        return xs.numpy(), {term: mf(xs).numpy() for term, mf in var.terms.items()}

    @staticmethod
    def _rule_label(rule) -> str:
        cond = ' & '.join(f"{_short(var.name)} {term}" for var, term in rule.antecedent)
        cons = rule.consequent
        then = cons[1] if isinstance(cons, tuple) else 'f(x)'
        return f"{cond} -> {then}"

    def evaluate(self, values: Dict[str, float]):
        """Memberships, rule strengths, output set and crisp output for these inputs."""
        system = self.system
        x = system_inputs(system, values)
        with torch.no_grad():
            strengths = system.get_rule_activations(x)[0].numpy()
            crisp = float(system(x))
            aggregated = rule_values = None
            if system.mode == TSK:
                X = torch.stack([x[n].reshape(1) for n in system.input_order], dim=1)
                rule_values = np.array([float(r.consequent_fn(X)[0]) for r in system.rules])
            elif len(system.rules):
                cons = system._consequent_matrix()
                w = torch.as_tensor(strengths).unsqueeze(-1)
                aggregated = system._aggregate(system.implication(w, cons).unsqueeze(0))[0].numpy()
        return {'x': {n: float(x[n]) for n in system.input_order}, 'strengths': strengths,
                'crisp': crisp, 'aggregated': aggregated, 'rule_values': rule_values}

    def height(self, rules: int) -> int:
        return 24 + 52 * (len(self.inputs) + 1) + 15 * rules + 6

    def draw(self, surface, fonts, rect, values: Dict[str, float], rules: int = 4):
        pg = _pygame()
        state = self.evaluate(values)
        x0, y = rect[0] + 8, rect[1] + 4
        width = rect[2] - 16
        _text(surface, fonts['bold'], self.title, (x0, y), ACCENT)
        crisp = state['crisp']
        _text(surface, fonts['bold'], f"{crisp:+.2f}" if math.isfinite(crisp) else 'no rule',
              (x0 + width - 60, y), VALUE)
        y += 22
        for var, (xs, curves) in zip(self.inputs, self.curves):
            value = state['x'][var.name]
            _text(surface, fonts['small'], f"{var.name} = {values[var.name]:+.2f}", (x0, y), TEXT)
            plot = pg.Rect(x0, y + 15, width, 34)
            with torch.no_grad():
                mus = {t: float(m(torch.tensor([value]))[0]) for t, m in var.terms.items()}
            self._plot(surface, fonts, plot, xs, curves, var, mus=mus, marker=value)
            y += 52
        _text(surface, fonts['small'], f"output: {self.system.output_variable.name}", (x0, y), TEXT)
        out = self.system.output_variable
        plot = pg.Rect(x0, y + 15, width, 34)
        if self.output_curves is not None:
            xs, curves = self.output_curves
            universe = getattr(self.system.defuzz, 'universe', None) if self.system.defuzz else None
            fill = None
            if state['aggregated'] is not None and universe is not None:
                fill = (universe.numpy(), state['aggregated'])
            self._plot(surface, fonts, plot, xs, curves, out, fill=fill, marker=crisp)
        else:
            self._stems(surface, plot, out, state)
        y += 52
        order = np.argsort(-state['strengths'])[:rules]
        for i in order:
            strength = float(state['strengths'][i])
            if strength < 0.01:
                break
            bar = pg.Rect(x0, y + 3, int(60 * strength), 9)
            pg.draw.rect(surface, ACCENT, bar)
            pg.draw.rect(surface, GRID, (x0, y + 3, 60, 9), 1)
            _text(surface, fonts['tiny'], f"{strength:.2f}  {self.labels[i]}", (x0 + 66, y), TEXT)
            y += 15

    def _plot(self, surface, fonts, plot, xs, curves, var, mus=None, fill=None, marker=None):
        pg = _pygame()
        pg.draw.rect(surface, BG, plot)
        lo, hi = var.range_min, var.range_max

        def px(x, mu):
            return (float(plot.x + (x - lo) / (hi - lo) * plot.w), float(plot.bottom - mu * plot.h))

        def lpx(x, mu):
            a, b = px(x, mu)
            return (a - plot.x, b - plot.y)

        # Translucent fills on their own layer, so overlapping terms stay readable.
        layer = pg.Surface(plot.size, pg.SRCALPHA)
        if fill is not None:
            ux, uy = fill
            points = [lpx(lo, 0.0)] + [lpx(a, b) for a, b in zip(ux, uy)] + [lpx(hi, 0.0)]
            pg.draw.polygon(layer, (255, 196, 46, 110), points)
        for k, (term, ys) in enumerate(curves.items()):
            if mus is not None and mus[term] > 0.01:
                # The fuzzified degree: the term's area cut at its membership.
                clipped = np.minimum(ys, mus[term])
                points = [lpx(xs[0], 0.0)] + [lpx(a, b) for a, b in zip(xs, clipped)] + [lpx(xs[-1], 0.0)]
                pg.draw.polygon(layer, (*TERMS[k % len(TERMS)], 95), points)
        surface.blit(layer, plot.topleft)
        label_end = -1e9
        for k, (term, ys) in enumerate(curves.items()):
            colour = TERMS[k % len(TERMS)]
            pg.draw.lines(surface, colour, False, [px(a, b) for a, b in zip(xs, ys)], 2)
            peak = xs[int(np.argmax(ys))]
            label = fonts['tiny'].render(term[:9], True, colour)
            x = min(max(px(peak, 1.0)[0] - label.get_width() / 2, plot.x + 1, label_end + 4),
                    plot.right - label.get_width())
            surface.blit(label, (x, plot.y - 1))
            label_end = x + label.get_width()
        if marker is not None and math.isfinite(marker):
            x = px(min(max(marker, lo), hi), 0.0)[0]
            pg.draw.line(surface, VALUE, (x, plot.y), (x, plot.bottom), 2)

    def _stems(self, surface, plot, var, state):
        """TSK output: each rule's crisp value as a stem, as high as its strength."""
        pg = _pygame()
        pg.draw.rect(surface, BG, plot)
        lo, hi = var.range_min, var.range_max
        for value, strength in zip(state['rule_values'], state['strengths']):
            x = float(plot.x + (min(max(value, lo), hi) - lo) / (hi - lo) * plot.w)
            pg.draw.line(surface, ACCENT, (x, plot.bottom), (x, float(plot.bottom - strength * plot.h)), 3)
        crisp = state['crisp']
        if math.isfinite(crisp):
            x = plot.x + (min(max(crisp, lo), hi) - lo) / (hi - lo) * plot.w
            pg.draw.line(surface, VALUE, (x, plot.y), (x, plot.bottom), 2)


class RaceDirector:
    """A window that runs :class:`KartRace` races: camera, fuzzy panels, map, standings.

    One window serves a whole session (heats, then a final): pass each race to
    :meth:`run`.  ``headless=True`` draws into an off-screen surface instead, for tests
    and screenshots.
    """

    #: Refresh the camera picture every this many steps (2 = 10 fps at the default step).
    video_every = 2

    def __init__(self, *, realtime: bool = True, headless: bool = False):
        # Start SuperTuxKart's renderer first.  The picture is always drawn off-screen;
        # a live window shows it from its own process (see supertuxkart.Window).
        _pystk2(True, *CAMERA)
        pg = _pygame()
        pg.init()
        self.realtime, self.headless = realtime, headless
        self.surface = pg.Surface((WIDTH, HEIGHT))
        self.window = None if headless else Window(WIDTH, HEIGHT, 'Fuzzy Kart Race')
        mono = 'dejavusansmono,menlo,consolas,monospace'
        self.fonts = {'bold': pg.font.SysFont(mono, 16, bold=True), 'small': pg.font.SysFont(mono, 13),
                      'tiny': pg.font.SysFont(mono, 11), 'big': pg.font.SysFont(mono, 22, bold=True)}
        self.clock = pg.time.Clock()
        self.running = True
        self.race = None
        self._rows: List[Tuple] = []

    # -- loop --------------------------------------------------------------------------
    def run(self, race, seed: int = 0, title: str = '', max_steps: Optional[int] = None):
        """Race ``race`` to the end and return its standings, or None if the window closed.

        In a window, the finished race stays on screen until ``Enter`` is pressed.
        """
        if not race.render:
            raise ValueError("create the KartRace with render=True")
        pg = _pygame()
        self.race, self.title = race, title or race.track_name
        if self.window is not None:
            self.window.set_title(f'Fuzzy Kart Race — {self.title}')
        self.panels = {}
        for name, driver in race.drivers.items():
            systems = [(getattr(driver, n, None), n.upper()) for n in ('steering', 'throttle')]
            self.panels[name] = [FuzzyPanel(s, t) for s, t in systems if s is not None]
        self.follow, self.follow_leader = race.names[0], len(race.names) > 1
        self.paused = self.done = self._next = False
        self._image, self._image_of, self._steps = None, None, 0
        race.reset(seed=seed)
        steps = 0
        while self.running and not self._next:
            if not self.paused and not self.done:
                self.done = race.step()
                steps += 1
                self._steps += 1
            self.draw()
            if self.window is not None:
                self._events(self.window.show(pg.surfarray.pixels3d(self.surface).transpose(1, 0, 2)))
                self.running = self.running and self.window.running
                self.clock.tick(round(1 / race.dt) if self.realtime and not self.done else 30)
            if self.headless and (self.done or max_steps is not None and steps >= max_steps):
                break
        return race.standings() if self.running else None

    def _events(self, events):
        names = self.race.names
        for event in events:
            if event[0] == 'quit' or event == ('key', 'escape'):
                self.running = False
            elif event[0] == 'key':
                key = event[1]
                if key in ('right', 'tab', 'left'):
                    step = -1 if key == 'left' else 1
                    self.follow = names[(names.index(self.follow) + step) % len(names)]
                    self.follow_leader = False
                elif key == 'l':
                    self.follow_leader = True
                elif key == 'space':
                    self.paused = not self.paused
                elif key == 'f':
                    self.realtime = not self.realtime
                elif key == 'return' and self.done:
                    self._next = True
            elif event[0] == 'click':
                for rect, name in self._rows:
                    if rect.collidepoint(event[1]) and name in self.race.drivers:
                        self.follow, self.follow_leader = name, False

    # -- drawing -----------------------------------------------------------------------
    def draw(self):
        pg = _pygame()
        race, surface = self.race, self.surface
        standings = race.standings()
        if self.follow_leader:
            self.follow = next((r['name'] for r in standings if not r['ai']), self.follow)
        surface.fill(BG)
        # Fetching frames reads every camera back from the GPU (~5 ms each), so the
        # video refreshes every `video_every` steps; the fuzzy panels update every step.
        if (self._image is None or self._image_of != self.follow
                or self._steps % self.video_every == 0):
            frame = race.frame(self.follow)
            self._image_of = self.follow
            self._image = None if frame is None else pg.transform.scale(
                pg.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), CAMERA)
        if self._image is None:
            _text(surface, self.fonts['small'], f"{self.follow} has no camera: SuperTuxKart "
                  f"renders at most {race.max_cameras} per race.", (20, 200), DIM)
            _text(surface, self.fonts['small'], "Race in heats to give every kart one.", (20, 220), DIM)
        else:
            surface.blit(self._image, (0, 0))
        _text(surface, self.fonts['bold'], self.title, (12, CAMERA[1] - 96), ACCENT)
        self._driver_badge(standings)
        self._panels()
        self._map(standings, pg.Rect(0, CAMERA[1], 250, HEIGHT - CAMERA[1]))
        self._standings(standings, pg.Rect(250, CAMERA[1], CAMERA[0] - 250, HEIGHT - CAMERA[1]))
        if self.done:
            message = 'RACE OVER' if self.headless else 'RACE OVER  -  Enter: continue'
            _text(surface, self.fonts['big'], message, (CAMERA[0] // 2 - 180, 50), ACCENT)
        elif self.paused:
            _text(surface, self.fonts['big'], 'PAUSED', (CAMERA[0] // 2 - 40, 16), ACCENT)

    def _driver_badge(self, standings):
        pg = _pygame()
        race, name = self.race, self.follow
        row = next(r for r in standings if r['name'] == name)
        badge = pg.Surface((330, 60), pg.SRCALPHA)
        badge.fill((10, 12, 16, 200))
        self.surface.blit(badge, (230, CAMERA[1] - 70))
        x, y = 240, CAMERA[1] - 64
        mode = 'following leader' if self.follow_leader else 'following'
        _text(self.surface, self.fonts['tiny'], f"{mode}  ·  {race.time:5.1f} s", (x, y), DIM)
        _text(self.surface, self.fonts['big'], f"P{row['position']}  {name[:16]}", (x, y + 14), TEXT)
        throttle, steer = race.actions[name]
        _text(self.surface, self.fonts['tiny'], f"throttle {throttle:+.2f}  steer {steer:+.2f}"
              f"{'  ERROR' if row['error'] else ''}", (x, y + 40), VALUE if row['error'] else DIM)

    def _panels(self):
        pg = _pygame()
        rect = pg.Rect(CAMERA[0], 0, WIDTH - CAMERA[0], HEIGHT)
        pg.draw.rect(self.surface, PANEL, rect)
        name = self.follow
        _text(self.surface, self.fonts['bold'], f"Fuzzy driver: {name[:24]}", (rect.x + 8, 8), TEXT)
        panels = self.panels[name]
        if not panels:
            _text(self.surface, self.fonts['small'], "(not a fuzzy driver)", (rect.x + 8, 34), DIM)
            return
        values = race_inputs(self.race.observations[name])
        error = self.race.errors.get(name)
        y = 30
        if error:
            _text(self.surface, self.fonts['tiny'], error[:70], (rect.x + 8, y), VALUE)
            y += 14
        rules = 4 if sum(p.height(4) for p in panels) <= HEIGHT - y else 2
        for panel in panels:
            h = panel.height(rules)
            panel.draw(self.surface, self.fonts, (rect.x, y, rect.w, h), values, rules)
            y += h + 4
            pg.draw.line(self.surface, GRID, (rect.x + 8, y - 3), (rect.right - 8, y - 3))

    def _map(self, standings, rect):
        pg = _pygame()
        track = self.race.track
        pts = np.vstack([track.start, track.start[:1]])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        scale = min((rect.w - 24) / max(hi[0] - lo[0], 1), (rect.h - 24) / max(hi[1] - lo[1], 1))
        centre = (lo + hi) / 2

        def px(p):
            return (float(rect.centerx + (p[0] - centre[0]) * scale),
                    float(rect.centery - (p[1] - centre[1]) * scale))

        pg.draw.rect(self.surface, PANEL, rect)
        pg.draw.lines(self.surface, GRID, True, [px(p) for p in pts], 7)
        pg.draw.lines(self.surface, DIM, True, [px(p) for p in pts], 1)
        for row in reversed(standings):
            point = px(row['ground'][0])
            colour = DIM if row['ai'] else (ACCENT if row['name'] == self.follow else TERMS[0])
            pg.draw.circle(self.surface, colour, point, 6 if row['name'] == self.follow else 4)

    def _standings(self, standings, rect):
        pg = _pygame()
        pg.draw.rect(self.surface, BG, rect)
        rows_per_col = max(1, (rect.h - 12) // 17)
        cols = max(1, math.ceil(len(standings) / rows_per_col))
        col_w = rect.w // cols
        self._rows = []
        for k, row in enumerate(standings):
            col, line = divmod(k, rows_per_col)
            x, y = rect.x + 8 + col * col_w, rect.y + 8 + line * 17
            result = (f"{row['race_time']:.1f}s" if row['race_time'] is not None
                      else f"L{row['lap']} {100 * row['progress']:3.0f}%")
            colour = (DIM if row['ai'] else VALUE if row['error']
                      else ACCENT if row['name'] == self.follow else TEXT)
            chars = max(6, col_w // 8 - 12)
            _text(self.surface, self.fonts['small'],
                  f"{row['position']:2d} {row['name'][:chars]:<{chars}} {result}", (x, y), colour)
            self._rows.append((pg.Rect(x, y, col_w, 17), row['name']))

    def screenshot(self, path):
        _pygame().image.save(self.surface, str(path))

    def close(self):
        if self.window is not None:
            self.window.close()
        _pygame().quit()


def _short(name: str) -> str:
    """Compact input names for rule labels."""
    return {'cross_track_error': 'cte', 'heading_error': 'head', 'lookahead_curvature': 'curv',
            'yaw_rate': 'yaw'}.get(name, name)


def _text(surface, font, text, pos, colour):
    surface.blit(font.render(text, True, colour), pos)


def _pygame():
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the race director needs pygame-ce; install 'torch-fis[envs]'") from exc
    return pygame
