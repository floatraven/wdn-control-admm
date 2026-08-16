"""Matplotlib figures — port of ``results_plotting.jl``.

The original is PGFPlotsX code with a header warning that it "is very messy and
is not intended to be reproduced".  This module keeps the same four figures but
rebuilds them as reusable functions that take :class:`~wdn_admm.results.SolverResult`
objects.

Colour choices
--------------
The pressure tolerances ``delta`` and the penalty parameters are *ordered*
quantities, so they are encoded with a single-hue sequential ramp (light to
dark) rather than with arbitrary categorical hues.  The few genuinely
categorical roles — distributed solution, centralised reference, uncontrolled
network — use the first three slots of a categorical palette validated for
colour-vision deficiency (worst all-pairs CVD ΔE 9.2, normal-vision ΔE 24.0).
Every series is also named in a legend, so identity is never carried by colour
alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .objectives import pressure_range
from .results import SolverResult

__all__ = [
    "SEQUENTIAL",
    "CATEGORICAL",
    "apply_style",
    "sequential_colors",
    "plot_objective_time_series",
    "plot_residuals",
    "plot_pressure_cdf",
    "plot_penalty_sweep",
    "save_figure",
]

#: Single-hue ramp, light -> dark, for ordered series (delta, gamma, beta).
SEQUENTIAL = ["#9dc3f0", "#5a9ae2", "#2a78d6", "#1d54a0", "#14406f"]

#: Categorical slots for the genuinely nominal roles.
CATEGORICAL = {
    "distributed": "#2a78d6",
    "centralised": "#eb6834",
    "uncontrolled": "#1baf7a",
    "reference": "#52514e",
}

_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8880", "surface": "#fcfcfb"}
_INK_DARK = {"primary": "#ffffff", "secondary": "#c3c2b7", "muted": "#8a8880", "surface": "#1a1a19"}


def apply_style(dark: bool = False, font_size: float = 11.0) -> None:
    """Install a recessive, publication-oriented Matplotlib style."""
    ink = _INK_DARK if dark else _INK
    mpl.rcParams.update(
        {
            "figure.facecolor": ink["surface"],
            "axes.facecolor": ink["surface"],
            "savefig.facecolor": ink["surface"],
            "axes.edgecolor": ink["muted"],
            "axes.labelcolor": ink["primary"],
            "axes.titlecolor": ink["primary"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": ink["muted"],
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "text.color": ink["primary"],
            "xtick.color": ink["secondary"],
            "ytick.color": ink["secondary"],
            "xtick.labelcolor": ink["secondary"],
            "ytick.labelcolor": ink["secondary"],
            "font.size": font_size,
            "axes.labelsize": font_size + 1,
            "legend.fontsize": font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "figure.dpi": 130,
        }
    )


def sequential_colors(n: int) -> list[str]:
    """``n`` evenly spaced steps of :data:`SEQUENTIAL`, light to dark."""
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENTIAL[2]]
    index = np.linspace(0, len(SEQUENTIAL) - 1, n)
    return [SEQUENTIAL[int(round(i))] for i in index]


def _shade_scc_window(ax: plt.Axes, scc_time: np.ndarray, label: str | None = None) -> None:
    """Grey band marking the self-cleaning-capacity time window."""
    if not len(scc_time):
        return
    ax.axvspan(
        scc_time.min() + 1,
        scc_time.max() + 1,
        color=_INK["muted"],
        alpha=0.15,
        lw=0,
        label=label,
        zorder=0,
    )


# ----------------------------------------------------------------------
def plot_objective_time_series(
    results: Sequence[SolverResult],
    labels: Sequence[str] | None = None,
    scc_time: np.ndarray | None = None,
    show_uncontrolled: bool = True,
) -> plt.Figure:
    """AZP and SCC objective time series (``azp_scc`` figure of the paper).

    Two stacked panels sharing the time axis — never a twin y-axis, since AZP
    is in metres and SCC in percent.
    """
    if not results:
        raise ValueError("at least one result is required")
    labels = list(labels) if labels is not None else [f"δ = {r.delta_max:g}" for r in results]
    scc_time = results[0].parameters.get("scc_time") if scc_time is None else scc_time
    scc_time = np.asarray(scc_time if scc_time is not None else [], dtype=int)

    nt = len(results[0].f_azp)
    steps = np.arange(1, nt + 1)
    colors = sequential_colors(len(results))

    fig, (ax_azp, ax_scc) = plt.subplots(2, 1, figsize=(6.5, 6.0), sharex=True)
    _shade_scc_window(ax_azp, scc_time, label="SCC window")
    _shade_scc_window(ax_scc, scc_time)

    if show_uncontrolled:
        ax_azp.plot(
            steps, results[0].f_azp_0, color=CATEGORICAL["uncontrolled"],
            ls="--", lw=1.6, label="no coupling constraint",
        )
        ax_scc.plot(
            steps, results[0].f_scc_0, color=CATEGORICAL["uncontrolled"], ls="--", lw=1.6
        )

    for result, label, color in zip(results, labels, colors):
        ax_azp.plot(steps, result.f_azp, color=color, label=label)
        ax_scc.plot(steps, result.f_scc, color=color)

    ax_azp.set_ylabel("AZP [m]")
    ax_scc.set_ylabel("SCC [%]")
    ax_scc.set_xlabel("Time step")
    ax_azp.set_xlim(1, nt)
    ax_azp.legend(loc="best", ncols=2)
    fig.tight_layout()
    return fig


def plot_residuals(
    results: Sequence[SolverResult],
    labels: Sequence[str] | None = None,
    name: str | None = None,
    tolerance: float | None = None,
) -> plt.Figure:
    """Residual convergence on a log scale, one line per run."""
    if not results:
        raise ValueError("at least one result is required")
    labels = list(labels) if labels is not None else [f"δ = {r.delta_max:g}" for r in results]
    colors = sequential_colors(len(results))

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    longest = 0
    for result, label, color in zip(results, labels, colors):
        key = name or ("primal" if "primal" in result.residual_names else result.residual_names[0])
        series = result.residual(key)
        iterations = np.arange(1, len(series) + 1)
        longest = max(longest, len(series))
        ax.semilogy(iterations, series, color=color, label=label)
        # Direct label at the end of the trace, in ink rather than series colour.
        ax.annotate(
            label,
            xy=(iterations[-1], series[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_INK["secondary"],
        )
    # Room for the direct labels, which sit outside the last data point.
    ax.set_xlim(1, longest * 1.12)

    if tolerance is not None:
        ax.axhline(tolerance, color=_INK["muted"], ls=":", lw=1.2)
        ax.annotate(
            "tolerance",
            xy=(1, tolerance),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=9,
            color=_INK["secondary"],
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|r\| \,/\, \sqrt{n_n n_t}$")
    if len(results) > 1:
        # A single trace is already named by its direct label.
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_pressure_cdf(
    heads: Sequence[np.ndarray],
    labels: Sequence[str],
    elevation: np.ndarray | None = None,
    quantity: str = "range",
    reference: np.ndarray | None = None,
    reference_label: str = "uncontrolled",
) -> plt.Figure:
    """Empirical CDF of the nodal head range (or pressure) across the network.

    Parameters
    ----------
    heads:
        One ``(nn, nt)`` head matrix per series.  These are the *ordered*
        family (one per pressure tolerance) and share the sequential ramp.
    quantity:
        ``"range"`` plots ``max_t h - min_t h`` per node; ``"pressure"`` plots
        every nodal pressure ``h - elevation`` over the whole horizon.
    reference:
        An optional extra head matrix — the uncontrolled network, say — which
        is not part of the ordered family and therefore gets a categorical
        colour and a dashed stroke instead of a step on the ramp.
    """
    if len(heads) != len(labels):
        raise ValueError("heads and labels must have the same length")
    colors = sequential_colors(len(heads))

    def values_of(h: np.ndarray) -> np.ndarray:
        if quantity == "range":
            return pressure_range(np.asarray(h))
        if quantity == "pressure":
            if elevation is None:
                raise ValueError("elevation is required for quantity='pressure'")
            return (np.asarray(h) - np.asarray(elevation)[:, None]).ravel()
        raise ValueError(f"unknown quantity: {quantity!r}")

    def ecdf(ax: plt.Axes, h: np.ndarray, label: str, **style) -> None:
        ordered = np.sort(values_of(h))
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, label=label, **style)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if reference is not None:
        ecdf(ax, reference, reference_label, color=CATEGORICAL["uncontrolled"], ls="--", lw=1.6)
    for h, label, color in zip(heads, labels, colors):
        ecdf(ax, h, label, color=color)

    ax.set_xlabel("Head range [m]" if quantity == "range" else "Pressure head [m]")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    if len(heads) + (reference is not None) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_penalty_sweep(
    penalties: Sequence[float],
    objective: np.ndarray,
    iterations: np.ndarray,
    labels: Sequence[str],
    max_iterations: int | None = None,
) -> plt.Figure:
    """Objective value and iteration count against the penalty parameter.

    ``objective`` and ``iterations`` are ``(n_penalties, n_series)`` arrays;
    each column is one pressure tolerance.  Non-converged runs should be
    ``inf``/``nan`` and are drawn as open markers at the iteration cap.
    """
    penalties = np.asarray(penalties, dtype=float)
    objective = np.atleast_2d(np.asarray(objective, dtype=float))
    iterations = np.atleast_2d(np.asarray(iterations, dtype=float))
    colors = sequential_colors(objective.shape[1])

    fig, (ax_obj, ax_it) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    for j, (label, color) in enumerate(zip(labels, colors)):
        ax_obj.semilogx(penalties, objective[:, j], color=color, marker="o", label=label)
        finite = np.isfinite(iterations[:, j])
        ax_it.loglog(penalties[finite], iterations[finite, j], color=color, marker="o", label=label)
        if max_iterations is not None and np.any(~finite):
            ax_it.loglog(
                penalties[~finite],
                np.full((~finite).sum(), max_iterations),
                color=color,
                marker="o",
                mfc="none",
                ls="none",
            )

    if max_iterations is not None:
        ax_it.axhline(max_iterations, color=_INK["muted"], ls=":", lw=1.2)
        ax_it.annotate(
            "iteration cap (not converged)",
            xy=(penalties[0], max_iterations),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=9,
            color=_INK["secondary"],
        )

    ax_obj.set_ylabel("Objective value")
    ax_it.set_ylabel("Iterations")
    ax_it.set_xlabel("Penalty parameter")
    ax_obj.legend(loc="best")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: str | Path, formats: Iterable[str] = ("pdf", "png")) -> list[Path]:
    """Write ``fig`` next to the Julia ``pgfsave`` outputs (``plots/`` by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = path.with_suffix(f".{fmt}")
        fig.savefig(target, bbox_inches="tight")
        written.append(target)
    return written
