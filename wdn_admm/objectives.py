"""Objective functions: average zone pressure (AZP) and self-cleaning capacity (SCC).

Both objectives appear verbatim in ``admm_standard.jl``, ``admm_two_level.jl``,
``centralized_solver.jl`` and ``sfscp_functions.jl``.  Collecting them here
means the ADMM residual bookkeeping, the NLP objective and the post-processing
all evaluate exactly the same expressions.

Definitions
-----------
AZP at time step ``t``::

    f_azp(t) = sum_i azp_weights[i] * (h[i, t] - elev[i])

SCC at time step ``t`` uses a two-sided sigmoid relaxation of the indicator
"link velocity exceeds ``umin``"::

    psi(q, s) = 1 / (1 + exp(-rho * (s * q/1000 * area - umin)))
    f_scc(t)  = sum_j scc_weights[j] * (psi(q[j, t], +1) + psi(q[j, t], -1))

SCC is *maximised*, so it enters the minimisation objectives with a minus sign.
"""

from __future__ import annotations

import numpy as np

from .data import ProblemData

__all__ = [
    "sigmoid_scc",
    "azp_time_series",
    "scc_time_series",
    "objective_time_series",
    "total_objective",
    "pressure_range",
]


def sigmoid_scc(
    q: np.ndarray,
    area: np.ndarray,
    rho: float,
    umin: float,
    sign: int = 1,
) -> np.ndarray:
    """Smoothed self-cleaning indicator for flows ``q`` [L/s].

    ``area`` is the reciprocal pipe area, so ``q / 1000 * area`` is the flow
    velocity in m/s.  Written with :func:`numpy.logaddexp` so that large
    ``rho`` values do not overflow.
    """
    velocity = sign * q / 1000.0 * area[:, None] if q.ndim == 2 else sign * q / 1000.0 * area
    return np.exp(-np.logaddexp(0.0, -rho * (velocity - umin)))


def azp_time_series(data: ProblemData, h: np.ndarray) -> np.ndarray:
    """AZP objective per time step, shape ``(nt,)``."""
    return data.azp_weights @ (h - data.elev[:, None])


def scc_time_series(data: ProblemData, q: np.ndarray) -> np.ndarray:
    """SCC objective per time step (positive = good), shape ``(nt,)``."""
    area = data.area
    psi_plus = sigmoid_scc(q, area, data.rho, data.umin, sign=1)
    psi_minus = sigmoid_scc(q, area, data.rho, data.umin, sign=-1)
    return data.scc_weights @ (psi_plus + psi_minus)


def objective_time_series(
    data: ProblemData, q: np.ndarray, h: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(f_val, f_azp, f_scc)`` per time step.

    ``f_val`` is the quantity actually minimised: AZP outside ``scc_time`` and
    ``-SCC`` inside it, matching the ``f_val`` array built at the end of each
    Julia driver script.
    """
    f_azp = azp_time_series(data, h)
    f_scc = scc_time_series(data, q)
    f_val = f_azp.copy()
    f_val[data.scc_time] = -f_scc[data.scc_time]
    return f_val, f_azp, f_scc


def total_objective(data: ProblemData, q: np.ndarray, h: np.ndarray, obj_type: str = "azp-scc") -> float:
    """Scalar objective over the whole horizon.

    Mirrors ``objective_function`` in ``src/sfscp_functions.jl``: the single
    objective variants are averaged over the horizon, the mixed variant is a
    plain sum over the two disjoint time windows.
    """
    f_azp = azp_time_series(data, h)
    f_scc = scc_time_series(data, q)

    if obj_type == "azp":
        return float(np.sum(f_azp) / data.nt)
    if obj_type == "scc":
        return float(-np.sum(f_scc) / data.nt)
    if obj_type == "azp-scc":
        return float(np.sum(f_azp[data.azp_time]) - np.sum(f_scc[data.scc_time]))
    raise ValueError(f"unknown obj_type: {obj_type!r}")


def pressure_range(h: np.ndarray) -> np.ndarray:
    """Per-node head range ``max_t h - min_t h``, shape ``(nn,)``."""
    return h.max(axis=1) - h.min(axis=1)
