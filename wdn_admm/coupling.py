"""Time-coupling block updates (the ``z`` / ``h_bar`` blocks).

Ports ``auxiliary_update`` from ``src/admm_functions.jl`` and
``h_bar_update`` / ``z_update`` / ``lambda_update`` from
``src/two_level_functions.jl``.

Both auxiliary updates minimise a separable quadratic over a
pressure-variation set ``Z``, so after completing the square they are *exactly*
a Euclidean projection onto ``Z``:

===============  ==================================================  ============================
``pv_type``      feasible set (per node ``i``)                        method used here
===============  ==================================================  ============================
``"none"``       unrestricted                                         identity
``"range"``      ``max_t z - min_t z <= delta``                        closed-form bisection
``"variation"``  ``|z[t+1] - z[t]| <= delta``                          OSQP (banded QP)
``"variability"`` ``z' A z <= delta^2`` and ``Hmin <= z <= Hmax``      CasADi/IPOPT, one NLP per node
===============  ==================================================  ============================

The Julia code solves the ``range`` and ``variation`` cases with Gurobi and the
``variability`` case with Ipopt.  Nothing here needs a commercial licence, and
the ``range`` case — the one used for every result in the paper — is solved
analytically to machine precision instead of by a 500k-constraint LP/QP.

.. note::
   ``pv_type="none"`` is broken in the Julia original: both
   ``auxiliary_update`` and ``h_bar_update`` reference ``model`` in that branch
   without ever constructing it.  The intended (unconstrained) update has a
   closed form and is implemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from .data import ProblemData

__all__ = [
    "PV_TYPES",
    "project_range",
    "project_variation",
    "project_variability",
    "CouplingProjector",
    "auxiliary_update",
    "h_bar_update",
    "z_update",
    "lambda_update",
    "variability_matrix",
]

PV_TYPES = ("none", "range", "variation", "variability")

# Regularisation added to the diagonal of the variability matrix, matching
# `reg = 1e-12` in the Julia code.
VARIABILITY_REG = 1e-12


# ----------------------------------------------------------------------
# range: projection onto {z : max(z) - min(z) <= delta}
# ----------------------------------------------------------------------
def project_range(target: np.ndarray, delta: float, iterations: int = 64) -> np.ndarray:
    """Row-wise Euclidean projection onto ``{z : max(z) - min(z) <= delta}``.

    For a fixed lower level ``l`` the minimiser is ``clip(target, l, l+delta)``,
    so the problem reduces to the scalar convex program

    .. math:: \\min_l \\sum_t (l - t)_+^2 + (t - l - \\delta)_+^2 .

    Its derivative is nondecreasing in ``l``, which makes a bisection on
    ``[min(target) - delta, max(target)]`` exact up to machine precision.  The
    bisection is run for every row simultaneously.

    Parameters
    ----------
    target:
        Array of shape ``(n_rows, n_cols)``; each row is projected separately.
    delta:
        Allowed range per row.
    """
    target = np.asarray(target, dtype=float)
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if not np.isfinite(delta):
        return target.copy()

    lo = target.min(axis=1, keepdims=True)
    hi = target.max(axis=1, keepdims=True)
    active = (hi - lo > delta).ravel()
    out = target.copy()
    if not np.any(active):
        return out

    t = target[active]
    left = t.min(axis=1, keepdims=True) - delta
    right = t.max(axis=1, keepdims=True)
    for _ in range(iterations):
        mid = 0.5 * (left + right)
        grad = np.sum(np.maximum(mid - t, 0.0) - np.maximum(t - mid - delta, 0.0), axis=1, keepdims=True)
        go_right = grad < 0.0
        left = np.where(go_right, mid, left)
        right = np.where(go_right, right, mid)
    level = 0.5 * (left + right)
    out[active] = np.clip(t, level, level + delta)
    return out


# ----------------------------------------------------------------------
# variation: projection onto {z : |z[t+1] - z[t]| <= delta}
# ----------------------------------------------------------------------
def _difference_matrix(nt: int) -> sp.csc_matrix:
    return sp.diags([-np.ones(nt - 1), np.ones(nt - 1)], offsets=[0, 1], shape=(nt - 1, nt)).tocsc()


class _VariationQP:
    """Cached OSQP program for the ``variation`` projection of one node."""

    def __init__(self, nt: int, delta: float, tol: float = 1e-9) -> None:
        import osqp

        self.nt = nt
        P = sp.eye(nt, format="csc")
        A = _difference_matrix(nt)
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=P,
            q=np.zeros(nt),
            A=A,
            l=np.full(nt - 1, -delta),
            u=np.full(nt - 1, delta),
            verbose=False,
            eps_abs=tol,
            eps_rel=tol,
            polishing=True,
            max_iter=20000,
        )

    def solve(self, target: np.ndarray) -> np.ndarray:
        self.problem.update(q=-target)
        result = self.problem.solve(raise_error=False)
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"OSQP failed on the variation projection: {result.info.status}")
        return np.asarray(result.x)


def project_variation(target: np.ndarray, delta: float) -> np.ndarray:
    """Row-wise projection onto ``{z : |z[t+1] - z[t]| <= delta}``."""
    target = np.asarray(target, dtype=float)
    n_rows, nt = target.shape
    if not np.isfinite(delta) or nt < 2:
        return target.copy()

    qp = _VariationQP(nt, delta)
    out = target.copy()
    for i in range(n_rows):
        if np.max(np.abs(np.diff(target[i]))) <= delta:
            continue  # already feasible
        out[i] = qp.solve(target[i])
    return out


# ----------------------------------------------------------------------
# variability: projection onto {z : z' A z <= delta^2} intersected with a box
# ----------------------------------------------------------------------
def variability_matrix(nt: int, reg: float = VARIABILITY_REG) -> sp.csc_matrix:
    """The matrix ``A`` from the Julia code: ``D' D + reg * I``.

    ``D`` is the first-difference operator, so ``z' A z`` is
    ``sum_t (z[t+1] - z[t])^2 + reg * ||z||^2`` — the squared "variability" of
    the head trajectory.
    """
    D = _difference_matrix(nt)
    return (D.T @ D + reg * sp.eye(nt)).tocsc()


class _VariabilityNLP:
    """Cached CasADi/IPOPT program for the ``variability`` projection."""

    def __init__(self, nt: int, delta: float, reg: float, parallel: str, n_threads: int) -> None:
        import casadi as ca

        self.nt = nt
        self.parallel = parallel
        self.n_threads = n_threads
        A = ca.DM(variability_matrix(nt, reg))

        z = ca.MX.sym("z", nt)
        t = ca.MX.sym("t", nt)
        f = 0.5 * ca.dot(z - t, z - t)
        g = ca.bilin(A, z, z)
        nlp = {"x": z, "p": t, "f": f, "g": g}
        self.solver = ca.nlpsol(
            "variability_projection",
            "ipopt",
            nlp,
            {
                "print_time": False,
                "ipopt": {
                    "print_level": 0,
                    "sb": "yes",
                    "max_iter": 500,
                    "mu_strategy": "adaptive",
                    "linear_solver": "mumps",
                },
            },
        )
        self.ubg = delta**2
        self._mapped: dict[int, Any] = {}

    def _map(self, n: int):
        if n not in self._mapped:
            if self.parallel in ("thread", "openmp"):
                self._mapped[n] = self.solver.map(n, self.parallel, min(self.n_threads, n))
            else:
                self._mapped[n] = self.solver.map(n)
        return self._mapped[n]

    def solve(self, target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        n = target.shape[0]
        solver = self._map(n)
        x0 = np.clip(target, lower, upper)
        result = solver(
            x0=x0.T,
            p=target.T,
            lbx=lower.T,
            ubx=upper.T,
            lbg=np.zeros((1, n)),
            ubg=np.full((1, n), self.ubg),
        )
        return np.asarray(result["x"]).reshape(self.nt, n, order="F").T


def project_variability(
    target: np.ndarray,
    delta: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    reg: float = VARIABILITY_REG,
    parallel: str = "thread",
    n_threads: int = 8,
) -> np.ndarray:
    """Row-wise projection onto ``{z : z' A z <= delta^2}`` inside a box."""
    target = np.asarray(target, dtype=float)
    n_rows, nt = target.shape
    lower = np.full_like(target, -np.inf) if lower is None else np.asarray(lower, dtype=float)
    upper = np.full_like(target, np.inf) if upper is None else np.asarray(upper, dtype=float)

    nlp = _VariabilityNLP(nt, delta, reg, parallel, n_threads)
    return nlp.solve(target, lower, upper)


# ----------------------------------------------------------------------
# projector facade
# ----------------------------------------------------------------------
class CouplingProjector:
    """Projection onto the pressure-variation set for a given ``pv_type``.

    Solver objects (OSQP / CasADi) are built once and reused across ADMM
    iterations, which is where most of the speed-up over the Julia version
    comes from — there a fresh JuMP model was constructed every iteration.
    """

    def __init__(
        self,
        data: ProblemData,
        pv_type: str = "range",
        delta_max: float = 10.0,
        *,
        parallel: str = "thread",
        n_threads: int = 8,
        use_bounds: bool | None = None,
    ) -> None:
        if pv_type not in PV_TYPES:
            raise ValueError(f"pv_type must be one of {PV_TYPES}, got {pv_type!r}")
        self.data = data
        self.pv_type = pv_type
        self.delta_max = float(delta_max)
        self.parallel = parallel
        self.n_threads = n_threads
        # Only the `variability` branch of the Julia code bounds the auxiliary
        # variable by [Hmin, Hmax]; the default keeps that asymmetry.
        self.use_bounds = (pv_type == "variability") if use_bounds is None else use_bounds

        self._variation_qp: _VariationQP | None = None
        self._variability_nlp: _VariabilityNLP | None = None

    def project(self, target: np.ndarray) -> np.ndarray:
        """Project each node's head trajectory onto the feasible set."""
        target = np.asarray(target, dtype=float)
        if self.pv_type == "none":
            out = target.copy()
        elif self.pv_type == "range":
            out = project_range(target, self.delta_max)
        elif self.pv_type == "variation":
            if self._variation_qp is None:
                self._variation_qp = _VariationQP(self.data.nt, self.delta_max)
            out = target.copy()
            for i in range(target.shape[0]):
                if np.max(np.abs(np.diff(target[i]))) > self.delta_max:
                    out[i] = self._variation_qp.solve(target[i])
        else:  # variability
            if self._variability_nlp is None:
                self._variability_nlp = _VariabilityNLP(
                    self.data.nt, self.delta_max, VARIABILITY_REG, self.parallel, self.n_threads
                )
            lower = self.data.h_min if self.use_bounds else np.full_like(target, -np.inf)
            upper = self.data.h_max if self.use_bounds else np.full_like(target, np.inf)
            out = self._variability_nlp.solve(target, lower, upper)

        if self.use_bounds and self.pv_type != "variability":
            out = np.clip(out, self.data.h_min, self.data.h_max)
        return out

    def violation(self, h: np.ndarray) -> float:
        """Worst-case violation of the coupling constraint by ``h``."""
        if self.pv_type == "none":
            return 0.0
        if self.pv_type == "range":
            return float(np.max(h.max(axis=1) - h.min(axis=1)) - self.delta_max)
        if self.pv_type == "variation":
            return float(np.max(np.abs(np.diff(h, axis=1))) - self.delta_max)
        A = variability_matrix(self.data.nt)
        values = np.einsum("ij,ij->i", h @ A.toarray(), h)
        return float(np.sqrt(np.max(values)) - self.delta_max)


# ----------------------------------------------------------------------
# block updates
# ----------------------------------------------------------------------
def auxiliary_update(
    h: np.ndarray,
    lam: np.ndarray,
    gamma: float,
    projector: CouplingProjector,
) -> np.ndarray:
    """Standard-ADMM auxiliary update (``auxiliary_update`` in Julia).

    Minimises ``sum lam*(h - z) + gamma/2 * ||h - z||^2`` over the coupling
    set, which after completing the square is the projection of
    ``h + lam/gamma``.  The scaled and unscaled formulations in the Julia code
    give the same minimiser, so a single implementation covers both.
    """
    if gamma <= 0:
        # Degenerate: the objective is linear in z, matching the (unused)
        # gamma = 0 branch. Fall back to projecting h itself.
        return projector.project(h)
    return projector.project(h + lam / gamma)


def h_bar_update(
    h: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    rho: float,
    projector: CouplingProjector,
) -> np.ndarray:
    """Two-level ``h_bar`` block update (``h̄_update`` in Julia).

    Minimises ``sum y*(h - h_bar + z) + rho/2 * ||h - h_bar + z||^2`` over the
    coupling set, i.e. the projection of ``h + z + y/rho``.
    """
    if rho <= 0:
        return projector.project(h + z)
    return projector.project(h + z + y / rho)


def z_update(
    h: np.ndarray,
    h_bar: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    beta: float,
    rho: float,
) -> np.ndarray:
    """Two-level slack block update (``z_update`` in Julia).

    The block is unconstrained, so the Gurobi model of the original reduces to
    the stationarity condition
    ``lam + beta*z + y + rho*(h - h_bar + z) = 0``.
    """
    denominator = beta + rho
    if denominator <= 0:
        raise ValueError("beta + rho must be positive for the z update")
    return -(lam + y + rho * (h - h_bar)) / denominator


def lambda_update(lam: np.ndarray, z: np.ndarray, beta: float, lam_bound: float) -> np.ndarray:
    """Outer ALM dual update with clipping (``λ_update`` in Julia)."""
    return np.clip(lam + beta * z, -lam_bound, lam_bound)
