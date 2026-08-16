"""Steady-state hydraulic simulation.

The Julia code calls ``hydraulic_simulation`` from ``OpWater``, a private
package that is not published with the repository.  Everything that function
is used for here can be reconstructed from the arrays that *are* stored in the
problem files, so this module provides a self-contained replacement based on
the global gradient algorithm (Todini & Pilati).

For every time step the following system is solved for ``(q, h)``:

.. math::

    r_i\\, q_i |q_i|^{n_i - 1} + (A_{12} h)_i + (A_{10} h_0)_i + \\eta_i = 0
    \\qquad
    A_{12}^\\top q - \\alpha = d

which is exactly the pair of hydraulic constraints imposed by the optimisation
problems in :mod:`wdn_admm.nlp`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .data import ProblemData

__all__ = ["HydraulicResult", "head_loss", "head_loss_gradient", "hydraulic_simulation"]

# Below this |q| the head-loss derivative is frozen, which keeps the Newton
# system non-singular for links carrying (almost) no flow.
_Q_FLOOR = 1e-6


@dataclass
class HydraulicResult:
    """Result of a multi-period hydraulic simulation."""

    q: np.ndarray  # (np, nt) flows [L/s]
    h: np.ndarray  # (nn, nt) heads [m]
    residual: np.ndarray  # (nt,) final infinity-norm residual per time step
    iterations: np.ndarray  # (nt,) Newton iterations used per time step
    converged: np.ndarray  # (nt,) bool

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))


def head_loss(q: np.ndarray, r: np.ndarray, nexp: np.ndarray) -> np.ndarray:
    """Head loss ``r q |q|^{nexp-1}`` [m]."""
    return r * q * np.abs(q) ** (nexp - 1.0)


def head_loss_gradient(q: np.ndarray, r: np.ndarray, nexp: np.ndarray) -> np.ndarray:
    """Derivative of :func:`head_loss` with respect to ``q``."""
    return nexp * r * np.abs(q) ** (nexp - 1.0)


def _solve_one_step(
    A12: sp.csc_matrix,
    A10: sp.csc_matrix,
    r: np.ndarray,
    nexp: np.ndarray,
    d: np.ndarray,
    h0: np.ndarray,
    eta: np.ndarray,
    alpha: np.ndarray,
    q0: np.ndarray,
    h_guess: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, int, bool]:
    q = q0.astype(float).copy()
    h = h_guess.astype(float).copy()
    A12T = A12.T.tocsc()
    source_head = A10 @ h0
    demand = d + alpha

    residual = np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        f = head_loss(q, r, nexp)
        g = np.maximum(head_loss_gradient(q, r, nexp), head_loss_gradient(np.full_like(q, _Q_FLOOR), r, nexp))

        energy = f + A12 @ h + source_head + eta
        mass = A12T @ q - demand
        residual = max(np.max(np.abs(energy)), np.max(np.abs(mass)))
        if residual < tol:
            converged = True
            break

        g_inv = sp.diags(1.0 / g)
        schur = (A12T @ g_inv @ A12).tocsc()
        rhs = mass - A12T @ (g_inv @ energy)
        dh = spla.spsolve(schur, rhs)
        dq = -(energy + A12 @ dh) / g

        # Damped step: full Newton first, backtrack if the residual grows.
        step = 1.0
        for _ in range(20):
            q_new = q + step * dq
            h_new = h + step * dh
            new_residual = max(
                np.max(np.abs(head_loss(q_new, r, nexp) + A12 @ h_new + source_head + eta)),
                np.max(np.abs(A12T @ q_new - demand)),
            )
            if new_residual < residual or step < 1e-3:
                break
            step *= 0.5
        q, h = q + step * dq, h + step * dh
    else:
        f = head_loss(q, r, nexp)
        residual = max(
            np.max(np.abs(f + A12 @ h + source_head + eta)),
            np.max(np.abs(A12T @ q - demand)),
        )
        converged = residual < tol

    return q, h, float(residual), it, bool(converged)


def hydraulic_simulation(
    data: ProblemData,
    eta: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    h0_guess: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> HydraulicResult:
    """Simulate every time step of ``data`` under valve/actuator settings.

    Parameters
    ----------
    eta:
        Valve head losses [m], shape ``(np, nt)``.  Zero (no control) by default.
    alpha:
        Actuator discharges [L/s], shape ``(nn, nt)``.  Zero by default.
    q0, h0_guess:
        Starting iterates; the stored uncontrolled solution by default.
    """
    np_, nn, nt = data.np_, data.nn, data.nt
    eta = np.zeros((np_, nt)) if eta is None else np.asarray(eta, dtype=float)
    alpha = np.zeros((nn, nt)) if alpha is None else np.asarray(alpha, dtype=float)
    q0 = data.q_init if q0 is None else np.asarray(q0, dtype=float)
    h0_guess = data.h_init if h0_guess is None else np.asarray(h0_guess, dtype=float)

    q = np.zeros((np_, nt))
    h = np.zeros((nn, nt))
    residual = np.zeros(nt)
    iterations = np.zeros(nt, dtype=int)
    converged = np.zeros(nt, dtype=bool)

    for t in range(nt):
        q[:, t], h[:, t], residual[t], iterations[t], converged[t] = _solve_one_step(
            data.A12,
            data.A10,
            data.r,
            data.nexp,
            data.d[:, t],
            data.h0[:, t],
            eta[:, t],
            alpha[:, t],
            q0[:, t],
            h0_guess[:, t],
            max_iter,
            tol,
        )

    return HydraulicResult(q=q, h=h, residual=residual, iterations=iterations, converged=converged)
