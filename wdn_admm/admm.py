"""ADMM drivers.

Ports the two main scripts of the repository:

* :class:`StandardADMM` — ``admm_standard.jl`` / ``admm_standard_threaded.jl``
* :class:`TwoLevelADMM` — ``admm_two_level.jl``

The scripts are turned into classes so a run can be configured, repeated and
tested without editing globals.  Two behavioural changes are worth calling out:

* The iterate history is no longer stored in full by default.  ``x_hist`` in the
  Julia code is ``(2np+2nn)*nt`` floats per iteration, which is 8.5 GB for a
  1000-iteration BWFL run; only what the residuals need is kept unless
  ``store_history=True``.
* The primal update runs through one threaded CasADi ``map`` rather than
  ``addprocs(7)`` plus ``@distributed``, so nothing has to be serialised to
  worker processes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .coupling import CouplingProjector, auxiliary_update, h_bar_update, lambda_update, z_update
from .data import ProblemData
from .nlp import CouplingTerm, TimeStepNLP
from .results import SolverResult

__all__ = ["StandardADMM", "TwoLevelADMM", "ADMMOptions", "TwoLevelOptions"]

logger = logging.getLogger(__name__)


@dataclass
class ADMMOptions:
    """Parameters of the standard ADMM algorithm.

    Defaults follow ``admm_standard.jl``.

    Attributes
    ----------
    gamma: penalty parameter ``gamma_k``.
    gamma_0: penalty used in the very first iteration (0 = solve the uncoupled
        problem, which is what produces the ``x_0`` reference iterate).
    max_iter: iteration cap ``kmax``.
    eps_primal, eps_dual: convergence tolerances.
    stopping: ``"primal"`` stops on the primal residual alone (the active
        branch in ``admm_standard.jl``); ``"boyd"`` uses the adaptive
        absolute/relative test from ``admm_standard_threaded.jl``.
    scale_residuals: divide residuals by ``sqrt(nn*nt)`` as in
        ``admm_standard.jl``.
    scaled: use the scaled dual variable form of the augmented Lagrangian.
    """

    gamma: float = 0.01
    gamma_0: float = 0.0
    max_iter: int = 1000
    eps_primal: float = 1e-2
    eps_dual: float = 1e-5
    stopping: str = "primal"
    scale_residuals: bool = True
    scaled: bool = False
    eps_abs: float = 1e-2
    eps_rel: float = 1e-3
    store_history: bool = False


@dataclass
class TwoLevelOptions:
    """Parameters of the two-level ALM/ADMM algorithm (``admm_two_level.jl``).

    Attributes
    ----------
    beta: initial ALM penalty ``beta_m``.
    gamma: penalty growth factor.
    omega: slack-decrease factor deciding whether ``beta`` grows.
    eps_1, eps_2: inner-level tolerances on residuals (14a) and (14b).
    eps_3: outer-level tolerance on ``||h - h_bar||``.
    lam_bound: clipping bound of the outer dual variable.
    rho_multiplier: ``rho = rho_multiplier * beta`` (2 in the Julia code).
    """

    beta: float = 0.1
    gamma: float = 1.25
    omega: float = 0.75
    max_iter: int = 1000
    eps_1: float = 1e-5
    eps_2: float = 1e-5
    eps_3: float = 1e-2
    lam_bound: float = 1e6
    rho_multiplier: float = 2.0
    store_history: bool = False


@dataclass
class _History:
    """Optional full iterate history."""

    enabled: bool
    x: list[np.ndarray] = field(default_factory=list)
    z: list[np.ndarray] = field(default_factory=list)

    def push(self, x: np.ndarray | None = None, z: np.ndarray | None = None) -> None:
        if not self.enabled:
            return
        if x is not None:
            self.x.append(x.copy())
        if z is not None:
            self.z.append(z.copy())


class _BaseADMM:
    """Shared plumbing: the local NLP, the projector and iteration logging."""

    algorithm = "admm"

    def __init__(
        self,
        data: ProblemData,
        pv_type: str = "range",
        delta_max: float = 10.0,
        *,
        nlp: TimeStepNLP | None = None,
        projector: CouplingProjector | None = None,
        parallel: str = "thread",
        n_threads: int = 8,
        linear_solver: str = "mumps",
        callback: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        self.data = data
        self.pv_type = pv_type
        self.delta_max = float(delta_max)
        self.nlp = nlp or TimeStepNLP(
            data, parallel=parallel, n_threads=n_threads, linear_solver=linear_solver
        )
        self.projector = projector or CouplingProjector(
            data, pv_type, delta_max, parallel=parallel, n_threads=n_threads
        )
        self.callback = callback

    def _scale(self, scale_residuals: bool) -> float:
        return float(np.sqrt(self.data.nn * self.data.nt)) if scale_residuals else 1.0

    def _log(
        self,
        iteration: int,
        values: dict[str, float],
        max_iter: int,
        note: str = "",
    ) -> None:
        body = ", ".join(f"{k} = {v:.3e}" for k, v in values.items())
        logger.info(
            "%s iteration %d of %d. %s%s", self.algorithm, iteration, max_iter, body,
            f" -- {note}" if note else "",
        )
        if self.callback is not None:
            self.callback(iteration, values)


class StandardADMM(_BaseADMM):
    """Standard two-block ADMM with a time-coupled auxiliary variable.

    Variables (using the names of the manuscript)::

        x = [q; h; eta; alpha]   local, one block per time step
        z                        auxiliary/coupling copy of h
        lambda                   dual variable of  h - z = 0
        gamma                    penalty parameter

    Each iteration solves ``nt`` independent NLPs for ``x``, projects onto the
    pressure-variation set for ``z`` and takes a dual ascent step.
    """

    algorithm = "standard-admm"

    def solve(self, options: ADMMOptions | None = None) -> SolverResult:
        options = options or ADMMOptions()
        data = self.data
        nn, nt = data.nn, data.nt
        scale = self._scale(options.scale_residuals)

        x = data.x0()
        x_first: np.ndarray | None = None
        z = data.h_init.copy()
        lam = np.zeros((nn, nt))

        history = _History(options.store_history)
        history.push(x=x, z=z)

        objective_history: list[np.ndarray] = []
        residuals: list[tuple[float, float]] = []
        converged = False
        iteration = 0

        start = time.perf_counter()
        for iteration in range(1, options.max_iter + 1):
            gamma = options.gamma_0 if iteration == 1 else options.gamma

            # --- x block: nt independent NLPs, solved in parallel -----------
            coupling = (
                CouplingTerm.none(nn, nt)
                if gamma == 0
                else CouplingTerm.standard_admm(z, lam, gamma, scaled=options.scaled)
            )
            report = self.nlp.solve(x, coupling)
            if not report.all_ok:
                raise RuntimeError(
                    f"IPOPT did not converge at time steps {report.failed_steps.tolist()} "
                    f"(ADMM iteration {iteration})"
                )
            x = report.x
            if x_first is None:
                x_first = x.copy()
            objective_history.append(report.objective)

            # --- z block: projection onto the coupling set ------------------
            h = data.split(x)[1]
            z_prev = z
            z = auxiliary_update(h, lam, options.gamma, self.projector)

            # --- dual ascent ------------------------------------------------
            lam = lam + options.gamma * (h - z)

            primal = float(np.linalg.norm(h - z) / scale)
            dual = float(np.linalg.norm(options.gamma * (z - z_prev)) / scale)
            residuals.append((primal, dual))
            history.push(x=x, z=z)

            converged = self._converged(options, primal, dual, h, z, lam, scale)
            self._log(
                iteration,
                {"primal residual": primal, "dual residual": dual},
                options.max_iter,
                note="converged" if converged else "",
            )
            if converged:
                break

        cpu_time = time.perf_counter() - start

        result = SolverResult.from_iterates(
            data=data,
            algorithm=self.algorithm,
            x=x,
            x0=x_first if x_first is not None else data.x0(),
            objective_history=np.array(objective_history),
            residuals=np.array(residuals),
            residual_names=("primal", "dual"),
            iterations=iteration,
            cpu_time=cpu_time,
            converged=converged,
            pv_type=self.pv_type,
            delta_max=self.delta_max,
            parameters={
                "gamma": options.gamma,
                "gamma_0": options.gamma_0,
                "scaled": options.scaled,
                "stopping": options.stopping,
                "eps_primal": options.eps_primal,
                "eps_dual": options.eps_dual,
            },
        )
        self.history = history
        return result

    def _converged(
        self,
        options: ADMMOptions,
        primal: float,
        dual: float,
        h: np.ndarray,
        z: np.ndarray,
        lam: np.ndarray,
        scale: float,
    ) -> bool:
        if options.stopping == "primal":
            return primal <= options.eps_primal
        if options.stopping == "primal-dual":
            return primal <= options.eps_primal and dual <= options.eps_dual
        if options.stopping == "boyd":
            # `admm_standard_threaded.jl`: unscaled residuals against adaptive
            # tolerances (Boyd et al. 2010, Section 3.3.1).
            n = np.sqrt(h.size)
            eps_p = n * options.eps_abs + options.eps_rel * max(
                np.linalg.norm(h), np.linalg.norm(z)
            )
            eps_d = n * options.eps_abs + options.eps_rel * np.linalg.norm(lam)
            return primal * scale <= eps_p and dual * scale <= eps_d
        raise ValueError(f"unknown stopping rule: {options.stopping!r}")


class TwoLevelADMM(_BaseADMM):
    """Two-level augmented-Lagrangian / ADMM algorithm.

    Follows Sun & Sun (2023) as implemented in ``admm_two_level.jl``.  The
    inner ADMM level has three blocks — ``x``, ``h_bar`` and a slack ``z`` —
    coupled by ``h - h_bar + z = 0`` with dual ``y``; the outer ALM level
    drives ``z`` to zero using dual ``lambda`` and penalty ``beta``.
    """

    algorithm = "two-level-admm"

    def solve(self, options: TwoLevelOptions | None = None) -> SolverResult:
        options = options or TwoLevelOptions()
        data = self.data
        nn, nt = data.nn, data.nt
        scale = float(np.sqrt(nn * nt))

        beta = options.beta
        rho = 0.0
        lam = np.zeros((nn, nt))

        x = data.x0()
        x_first: np.ndarray | None = None
        h_bar = data.h_init.copy()
        z = np.zeros((nn, nt))
        y = np.zeros((nn, nt))

        # The Julia code compares against the *stored* history, which is not
        # refreshed by the ALM restart in step 9; these mirror that.
        h_bar_stored = h_bar.copy()
        z_stored = z.copy()

        history = _History(options.store_history)
        history.push(x=x, z=h_bar)

        objective_history: list[np.ndarray] = []
        residuals: list[tuple[float, ...]] = []
        res_z_prev = 0.0
        outer = 1
        converged = False
        iteration = 0

        start = time.perf_counter()
        for iteration in range(1, options.max_iter + 1):
            # --- Step 1: x block, nt independent NLPs ----------------------
            coupling = CouplingTerm.two_level(h_bar, z, y, lam, beta, rho)
            report = self.nlp.solve(x, coupling)
            if not report.all_ok:
                raise RuntimeError(
                    f"IPOPT did not converge at time steps {report.failed_steps.tolist()} "
                    f"(two-level iteration {iteration})"
                )
            x = report.x
            if x_first is None:
                x_first = x.copy()
            objective_history.append(report.objective)

            if iteration == 1:
                rho = options.rho_multiplier * beta

            h = data.split(x)[1]

            # --- Step 2: h_bar block (couples time steps) ------------------
            h_bar = h_bar_update(h, z, y, rho, self.projector)

            # --- Step 3: slack block (unconstrained, closed form) ----------
            z_new = z_update(h, h_bar, y, lam, beta, rho)

            # --- Step 4: inner dual ----------------------------------------
            y = y + rho * (h - h_bar + z_new)

            # --- Step 5: residuals (14a)-(14c) of Sun & Sun (2023) ---------
            res_a = float(np.linalg.norm(rho * (h_bar - h_bar_stored + z_stored - z_new)) / scale)
            res_b = float(np.linalg.norm(rho * (z_new - z_stored)) / scale)
            res_c = float(np.linalg.norm(h - h_bar + z_new) / scale)
            res_outer = float(np.linalg.norm(h - h_bar) / scale)
            res_z = float(np.linalg.norm(z_new))
            residuals.append((res_a, res_b, res_c, res_outer, res_z))

            h_bar_stored, z_stored, z = h_bar.copy(), z_new.copy(), z_new
            history.push(x=x, z=h_bar)

            # --- Step 6: inner stopping test -------------------------------
            inner_done = (res_c <= 1.0 / (100.0 * outer)) or (
                (res_b <= options.eps_2 or res_a <= options.eps_1) and iteration > 1
            )
            self._log(
                iteration,
                {"res (14a)": res_a, "res (14b)": res_b, "res (14c)": res_c, "outer res": res_outer},
                options.max_iter,
                note=f"inner ADMM level finished, ALM iteration {outer}" if inner_done else "",
            )

            if inner_done:
                if res_outer <= options.eps_3:
                    converged = True
                    logger.info(
                        "two-level algorithm converged: outer residual = %.3e at ALM iteration %d",
                        res_outer,
                        outer,
                    )
                    break

                # --- Step 7: outer dual ------------------------------------
                lam = lambda_update(lam, z, beta, options.lam_bound)

                # --- Step 8: penalty update --------------------------------
                if res_z > options.omega * res_z_prev and beta * options.gamma <= 1e6 and outer > 1:
                    beta *= options.gamma
                    rho = options.rho_multiplier * beta

                outer += 1
                res_z_prev = res_z

                # --- Step 9: restart the inner ADMM level ------------------
                z = np.zeros((nn, nt))
                y = -lam
                h_bar = h

        cpu_time = time.perf_counter() - start

        result = SolverResult.from_iterates(
            data=data,
            algorithm=self.algorithm,
            x=x,
            x0=x_first if x_first is not None else data.x0(),
            objective_history=np.array(objective_history),
            residuals=np.array(residuals),
            residual_names=("inner_a", "inner_b", "inner_c", "outer", "slack"),
            iterations=iteration,
            cpu_time=cpu_time,
            converged=converged,
            pv_type=self.pv_type,
            delta_max=self.delta_max,
            parameters={
                "beta_0": options.beta,
                "beta_final": beta,
                "gamma": options.gamma,
                "omega": options.omega,
                "eps_1": options.eps_1,
                "eps_2": options.eps_2,
                "eps_3": options.eps_3,
            },
            outer_iterations=outer,
        )
        self.history = history
        return result
