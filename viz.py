"""Matplotlib views of membership functions, variables and the inference process.

The classes expose these as methods -- ``mf.plot()``, ``variable.plot()``,
``system.plot_inference(x)`` and ``system.plot_surface()`` -- so most code never
imports this module directly.

matplotlib is imported lazily: the library itself needs only torch, and plotting
needs ``pip install matplotlib`` (or the ``[plot]`` extra).
"""
import math
from typing import Dict, Optional, Sequence, Tuple

import torch

from .membership import (
    BellMF, GaussianMF, MembershipFunction, SigmoidMF, TrapezoidalMF, TriangularMF,
)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotting needs matplotlib; install with: pip install matplotlib "
            "(or pip install 'torch-fis[plot]')") from exc
    return plt


def _value(t) -> float:
    return float(t.detach()) if torch.is_tensor(t) else float(t)


def default_span(mf: MembershipFunction) -> Tuple[float, float]:
    """A range that shows the whole interesting part of ``mf``."""
    if isinstance(mf, GaussianMF):
        mu, sigma = _value(mf.mu), abs(_value(mf.sigma)) or 1.0
        return mu - 3 * sigma, mu + 3 * sigma
    if isinstance(mf, BellMF):
        c, a = _value(mf.c), abs(_value(mf.a)) or 1.0
        return c - 3 * a, c + 3 * a
    if isinstance(mf, SigmoidMF):
        c, b = _value(mf.c), abs(_value(mf.b)) or 1.0
        return c - 6 / b, c + 6 / b
    if isinstance(mf, (TriangularMF, TrapezoidalMF)):
        points = [_value(p) for p in mf.parameters()]
        lo, hi = min(points), max(points)
        pad = 0.1 * (hi - lo) or 1.0
        return lo - pad, hi + pad
    # Unknown shape: fall back to the spread of its parameters.
    points = [_value(p) for p in mf.parameters() if p.numel() == 1] or [0.0, 1.0]
    lo, hi = min(points), max(points)
    pad = 0.5 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def _grid(lo: float, hi: float, n: int, device=None) -> torch.Tensor:
    return torch.linspace(float(lo), float(hi), n, device=device)


def _device(module) -> torch.device:
    for p in module.parameters():
        return p.device
    return torch.device('cpu')


# --------------------------------------------------------------------- sets

def plot_membership(mf: MembershipFunction, x=None, ax=None, n: int = 400,
                    label: Optional[str] = None, **kwargs):
    """Draw one membership function.

    Args:
        x: ``(lo, hi)`` range or a tensor of points; defaults to a span that
            covers the function (see :func:`default_span`).
        ax: matplotlib axes to draw into; a new figure is made if omitted.
        kwargs: passed to ``ax.plot`` (``color``, ``ls``, ...).

    Returns the axes.
    """
    plt = _plt()
    own = ax is None
    ax = ax or plt.subplots(figsize=(5, 2.8))[1]
    if x is None:
        x = default_span(mf)
    if not torch.is_tensor(x):
        x = _grid(x[0], x[1], n, device=_device(mf))
    with torch.no_grad():
        y = mf(x)
    ax.plot(x.cpu(), y.cpu(), label=label or type(mf).__name__, **kwargs)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('membership')
    if label is not None:
        ax.legend(fontsize=8)
    if own:
        ax.figure.tight_layout()
    return ax


def plot_variable(var, value=None, ax=None, n: int = 400, legend: bool = True, **kwargs):
    """Draw every term of a :class:`~fuzzy_torch.variable.FuzzyVariable`.

    Args:
        value: optional crisp input. It is marked with a vertical line, and each
            term's membership degree at that point is shown in the legend.
        kwargs: passed to ``ax.plot`` for every term.

    Returns the axes.
    """
    plt = _plt()
    own = ax is None
    ax = ax or plt.subplots(figsize=(6, 3))[1]
    x = _grid(var.range_min, var.range_max, n, device=_device(var))
    with torch.no_grad():
        for i, (term, mf) in enumerate(var.terms.items()):
            label = term
            if value is not None:
                degree = _value(mf(torch.as_tensor(float(value), device=x.device)))
                label = f'{term} ({degree:.2f})'
                ax.plot([value], [degree], 'o', color=f'C{i}', ms=5)
            ax.plot(x.cpu(), mf(x).cpu(), label=label, color=f'C{i}', **kwargs)
    if value is not None:
        ax.axvline(float(value), color='k', ls='--', lw=1)
    ax.set_xlim(var.range_min, var.range_max)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(var.name)
    ax.set_ylabel('membership')
    title = var.name if value is None else f'{var.name} = {float(value):.3g}'
    ax.set_title(title)
    if legend and len(var.terms):
        ax.legend(fontsize=8, loc='best')
    if own:
        ax.figure.tight_layout()
    return ax


# ---------------------------------------------------------------- inference

_LABEL = dict(fontsize=7, va='top', bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                              ec='none', alpha=0.8))


def _first_item(system, x: Dict) -> Dict[str, torch.Tensor]:
    prepared, _, _ = system._prepare_inputs(x)
    return {name: t[:1] for name, t in prepared.items()}


def plot_inference(system, x: Dict, n: int = 300, active_only: Optional[bool] = None,
                   threshold: float = 1e-6):
    """Rule-viewer figure: how one input produced the system's output.

    One row per rule and one column per input, plus an output column:

    - antecedent cells show the term's membership function, the crisp input and
      the membership degree it reaches (shaded);
    - Mamdani: the output cell shows the consequent set after implication with
      the rule's firing strength ``w``; the last row shows the aggregated set
      and the defuzzified output.
    - TSK: the output cell marks the rule's crisp value ``f_i(x)`` and its share
      of the total weight; the last row shows every ``f_i`` weighted by ``w_i``
      and the resulting weighted average.

    Only the first item of a batch is drawn.

    Args:
        x: inputs, as for ``system(x)``.
        active_only: draw only rules that fire (``w > threshold``). Defaults to
            True for rule bases larger than 12 rules, False otherwise.

    Returns the matplotlib figure.
    """
    from .system import TSK
    plt = _plt()

    inputs = _first_item(system, x)
    out_var = system.output_variable
    with torch.no_grad():
        w_all = system._activations(inputs, 1)[0]
        crisp = _value(system(inputs)[0])

    indices = list(range(len(system.rules)))
    if active_only is None:
        active_only = len(indices) > 12
    if active_only:
        indices = [i for i in indices if _value(w_all[i]) > threshold]

    names = system.input_order
    n_rows, n_cols = len(indices) + 1, len(names) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols + 0.6, 1.25 * n_rows + 0.9),
                             squeeze=False, sharex='col')

    grids = {name: _grid(system.input_variables[name].range_min,
                         system.input_variables[name].range_max, n, device=w_all.device)
             for name in names}
    y_grid = _grid(out_var.range_min, out_var.range_max, n, device=w_all.device)
    tsk = system.mode == TSK
    total = _value(w_all.sum())

    clipped_sets, tsk_points = [], []
    with torch.no_grad():
        for row, i in enumerate(indices):
            rule = system.rules[i]
            w = _value(w_all[i])
            used = {v.name: (v, t) for v, t in rule.antecedent}

            for col, name in enumerate(names):
                ax = axes[row, col]
                if name not in used:
                    ax.set_yticks([])
                    ax.set_facecolor('#f4f4f4')
                    continue
                var, term = used[name]
                mf = var.terms[term]
                g = grids[name]
                mu = mf(g).cpu()
                value = _value(inputs[name][0])
                degree = _value(mf(inputs[name]))
                ax.plot(g.cpu(), mu, color='C0', lw=1.2)
                ax.fill_between(g.cpu(), torch.minimum(mu, torch.tensor(degree)),
                                color='C0', alpha=0.35)
                ax.axvline(value, color='r', lw=1)
                ax.text(0.02, 0.92, f'{term}: {degree:.2f}', transform=ax.transAxes, **_LABEL)

            ax = axes[row, -1]
            if tsk:
                fi = _value(rule.consequent_fn(
                    torch.stack([inputs[nm] for nm in names], dim=1))[0])
                share = w / total if total > 1e-9 else 0.0
                tsk_points.append((fi, w))
                ax.vlines(fi, 0, max(w, 0.02), color='C2', lw=3)
                ax.text(0.02, 0.92, f'f = {fi:.3g}  ({share:.0%})', transform=ax.transAxes,
                        **_LABEL)
            else:
                _, term = rule.consequent
                mu = out_var.terms[term](y_grid)
                implied = system.implication(torch.tensor(w, device=mu.device), mu)
                clipped_sets.append(implied)
                ax.plot(y_grid.cpu(), mu.cpu(), color='C2', lw=1, ls=':')
                ax.fill_between(y_grid.cpu(), implied.cpu(), color='C2', alpha=0.5)
                ax.text(0.02, 0.92, term, transform=ax.transAxes, **_LABEL)
            ax.set_xlim(out_var.range_min, out_var.range_max)
            axes[row, 0].set_ylabel(f'R{i + 1}\nw={w:.2f}', rotation=0, ha='right',
                                    va='center', fontsize=8)

        for row in range(n_rows - 1):
            for ax in axes[row]:
                ax.set_ylim(0, 1.08)
                ax.tick_params(labelsize=7)

        # Bottom row: every term of each input, and the combined result.
        for col, name in enumerate(names):
            ax = axes[-1, col]
            var = system.input_variables[name]
            value = _value(inputs[name][0])
            for term, mf in var.terms.items():
                ax.plot(grids[name].cpu(), mf(grids[name]).cpu(), color='0.55', lw=0.8)
            ax.axvline(value, color='r', lw=1)
            ax.set_ylim(0, 1.08)
            ax.set_xlabel(f'{name} = {value:.3g}', fontsize=9)
            ax.set_xlim(var.range_min, var.range_max)
            ax.tick_params(labelsize=7)
        axes[-1, 0].set_ylabel('all terms', rotation=0, ha='right', va='center', fontsize=8)
        ax = axes[-1, -1]
        if tsk:
            for fi, w in tsk_points:
                ax.vlines(fi, 0, w, color='C2', lw=3, alpha=0.7)
            ax.set_ylim(0, max([w for _, w in tsk_points] + [0.05]) * 1.15)
            ax.set_title('weighted average', fontsize=8)
        else:
            if clipped_sets:
                agg = clipped_sets[0]
                for s in clipped_sets[1:]:
                    agg = system.aggregation(agg, s)
                ax.fill_between(y_grid.cpu(), agg.cpu(), color='0.4', alpha=0.6)
            ax.set_ylim(0, 1.08)
            ax.set_title(f'aggregated ({type(system.aggregation).__name__})', fontsize=8)
        if math.isfinite(crisp):
            ax.axvline(crisp, color='r', lw=2)
        ax.set_xlim(out_var.range_min, out_var.range_max)
        ax.set_xlabel(f'{out_var.name} = {crisp:.3g}', fontsize=9, color='r')
        ax.tick_params(labelsize=7)

    for col, name in enumerate(names):
        axes[0, col].set_title(name, fontsize=9)
    axes[0, -1].set_title(f'{out_var.name} (output)', fontsize=9)
    hidden = len(system.rules) - len(indices)
    note = f' - {hidden} inactive rules hidden' if hidden else ''
    fig.suptitle(f'{system.mode.upper()} inference: {out_var.name} = {crisp:.4g}{note}',
                 fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ surface

def plot_surface(system, x: Optional[str] = None, y: Optional[str] = None,
                 fixed: Optional[Dict[str, float]] = None, n: int = 60, ax=None,
                 cmap: str = 'viridis'):
    """Output of the system over one or two inputs.

    With one input (``y`` omitted) it draws a curve; with two, a filled contour.
    Any other inputs are held at ``fixed[name]``, or at the middle of their range.
    Regions where no rule fires appear blank (NaN).

    Args:
        x, y: input names. Name only ``x`` for a curve; name neither for the
            first two inputs.
        fixed: values for the inputs that are not plotted.

    Returns the axes.
    """
    plt = _plt()
    names = system.input_order
    if x is None:
        # Nothing named: the first two inputs (or the only one).
        x, y = names[0], (names[1] if len(names) > 1 else None)
    fixed = dict(fixed or {})
    xv = system.input_variables[x]
    device = _device(system)
    gx = _grid(xv.range_min, xv.range_max, n, device=device)

    if y is None:
        batch = {x: gx}
    else:
        yv = system.input_variables[y]
        gy = _grid(yv.range_min, yv.range_max, n, device=device)
        X, Y = torch.meshgrid(gx, gy, indexing='xy')
        batch = {x: X.reshape(-1), y: Y.reshape(-1)}
    size = next(iter(batch.values())).shape[0]
    for name in names:
        if name not in batch:
            v = system.input_variables[name]
            value = fixed.get(name, (v.range_min + v.range_max) / 2)
            batch[name] = torch.full((size,), float(value), device=device)

    with torch.no_grad():
        Z = system(batch).cpu()

    own = ax is None
    others = [f'{k}={_value(batch[k][0]):.3g}' for k in names if k not in (x, y)]
    suffix = f"  ({', '.join(others)})" if others else ''
    if y is None:
        ax = ax or plt.subplots(figsize=(5.5, 3.2))[1]
        ax.plot(gx.cpu(), Z)
        ax.set_xlabel(x)
        ax.set_ylabel(system.output_variable.name)
    else:
        ax = ax or plt.subplots(figsize=(5.5, 4.2))[1]
        im = ax.contourf(X.cpu(), Y.cpu(), Z.reshape(n, n), levels=20, cmap=cmap)
        ax.figure.colorbar(im, ax=ax, label=system.output_variable.name)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    ax.set_title(f'{system.output_variable.name}{suffix}', fontsize=10)
    if own:
        ax.figure.tight_layout()
    return ax
