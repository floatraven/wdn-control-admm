"""CasADi/IPOPT formulation of the control problem.

This replaces the JuMP models in ``src/admm_functions.jl`` (``primal_update``),
``src/two_level_functions.jl`` (``x_update``) and ``centralized_solver.jl``.

Two things are done differently from the Julia original, both of which leave
the optimal solution unchanged:

``psi`` elimination
    JuMP declares ``ψ⁺``/``ψ⁻`` as variables pinned by nonlinear equality
    constraints.  Since each is an explicit function of a single flow, they are
    substituted directly into the objective here, removing ``2*np`` variables
    and ``2*np`` constraints per time step.

control reduction
    ``make_object_data`` zeroes the valve bounds away from ``v_loc`` and the
    actuator bounds away from ``y_loc``, so all but ``n_v`` entries of ``eta``
    and all but ``n_f`` entries of ``alpha`` are fixed at zero.  Only the free
    entries become decision variables (``reduce_controls=True``).

The coupling term shared by both ADMM variants is written once, in the generic
form

.. math::

    \\text{lin}^\\top (h - \\text{target})
      + \\tfrac{\\text{quad}}{2}\\,\\lVert h - \\text{target}\\rVert^2
      + \\text{const}

so a single parameterised NLP serves the standard ADMM, its scaled variant and
the two-level algorithm.  See :meth:`CouplingTerm` for the mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import casadi as ca
import numpy as np

from .data import ProblemData

__all__ = [
    "CouplingTerm",
    "TimeStepNLP",
    "SolveReport",
    "default_ipopt_options",
    "logistic",
]

# Offset used by the Julia code to keep `d/dq |q|^(n-1)` finite at q = 0.
HEAD_LOSS_REG = 1e-8
# Lower bound of the bilinear valve-direction constraint `eta * q >= EPS_BILINEAR`.
EPS_BILINEAR = 0.0


def default_ipopt_options(
    print_level: int = 0,
    max_iter: int = 3000,
    linear_solver: str = "mumps",
    resto: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """IPOPT options mirroring the ``set_optimizer_attribute`` calls in Julia.

    The only intentional difference is ``linear_solver``: the Julia code asks
    for HSL's ``ma57``, which needs a separate licensed library.  ``mumps``
    ships with CasADi and needs nothing extra; pass ``linear_solver="ma57"`` if
    you do have HSL installed.
    """
    ipopt: dict[str, Any] = {
        "max_iter": max_iter,
        "print_level": print_level,
        "warm_start_init_point": "yes",
        "linear_solver": linear_solver,
        "mu_strategy": "adaptive",
        "mu_oracle": "quality-function",
        "fixed_variable_treatment": "make_parameter",
        "sb": "yes",
    }
    if resto:
        ipopt["start_with_resto"] = "yes"
        ipopt["expect_infeasible_problem"] = "yes"
    if extra:
        ipopt.update(extra)
    return {"ipopt": ipopt, "print_time": False}


def logistic(x):
    """Overflow-safe logistic ``1/(1+exp(-x)) == 0.5*(1+tanh(x/2))``.

    Algebraically identical to the expression in the Julia code, but ``tanh``
    saturates instead of overflowing for the large ``rho`` values used here.
    """
    return 0.5 * (1.0 + ca.tanh(0.5 * x))


@dataclass
class CouplingTerm:
    """Per-time-step coupling penalty added to the local objective.

    ``lin @ (h - target) + 0.5 * quad * ||h - target||^2 + const``

    Attributes
    ----------
    lin, target:
        Arrays of shape ``(nn, nt)``.
    quad:
        Scalar penalty weight.
    const:
        Constant offset per time step, shape ``(nt,)``; it does not affect the
        minimiser but keeps the reported objective values comparable with the
        Julia output.
    """

    lin: np.ndarray
    target: np.ndarray
    quad: float
    const: np.ndarray

    @staticmethod
    def none(nn: int, nt: int) -> "CouplingTerm":
        """No coupling (used for the very first ADMM iteration, ``gamma = 0``)."""
        return CouplingTerm(np.zeros((nn, nt)), np.zeros((nn, nt)), 0.0, np.zeros(nt))

    @staticmethod
    def standard_admm(
        z: np.ndarray, lam: np.ndarray, gamma: float, scaled: bool = False
    ) -> "CouplingTerm":
        """``lambda' (h - z) + gamma/2 ||h - z||^2`` (``primal_update``).

        With ``scaled=True`` the scaled dual ``u = lambda / gamma`` is used
        instead, i.e. ``gamma/2 ||h - z + u||^2`` — Section 3.1.1 of
        Boyd et al. (2010), matching the ``scaled`` branch in Julia.
        """
        nn, nt = z.shape
        if scaled:
            u = np.zeros_like(lam) if not np.any(lam) else lam / gamma
            return CouplingTerm(np.zeros((nn, nt)), z - u, gamma, np.zeros(nt))
        return CouplingTerm(lam, z, gamma, np.zeros(nt))

    @staticmethod
    def two_level(
        h_bar: np.ndarray, z: np.ndarray, y: np.ndarray, lam: np.ndarray, beta: float, rho: float
    ) -> "CouplingTerm":
        """``lambda' z + beta/2 ||z||^2 + y'(h - h_bar + z) + rho/2 ||h - h_bar + z||^2``.

        Matches ``x_update`` in ``src/two_level_functions.jl``.  The terms that
        do not involve ``h`` go into ``const``.
        """
        const = np.sum(lam * z + 0.5 * beta * z**2, axis=0)
        return CouplingTerm(y, h_bar - z, rho, const)


@dataclass
class SolveReport:
    """Outcome of a batch of time-step solves."""

    x: np.ndarray  # (2*np + 2*nn, nt) stacked [q; h; eta; alpha]
    objective: np.ndarray  # (nt,)
    status: np.ndarray  # (nt,) 0 = accepted, 1 = failed
    messages: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(np.all(self.status == 0))

    @property
    def failed_steps(self) -> np.ndarray:
        return np.flatnonzero(self.status != 0)


_ACCEPTED = {
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Search_Direction_Becomes_Too_Small",
    "Feasible_Point_Found",
}


class TimeStepNLP:
    """Parameterised NLP for one time step of the control problem.

    A single CasADi ``nlpsol`` object is built once and then evaluated for
    every time step.  ``parallel="thread"`` evaluates all time steps
    concurrently through :meth:`casadi.Function.map`, which is what replaces
    Julia's ``@sync @distributed for t in 1:nt``.
    """

    def __init__(
        self,
        data: ProblemData,
        *,
        reduce_controls: bool = True,
        linear_solver: str = "mumps",
        max_iter: int = 3000,
        print_level: int = 0,
        parallel: str = "thread",
        n_threads: int | None = None,
        feasibility_tol: float = 1e-4,
        ipopt_options: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.reduce_controls = reduce_controls
        self.parallel = parallel
        self.n_threads = n_threads or min(8, max(1, data.nt))
        self.feasibility_tol = feasibility_tol

        np_, nn, n0 = data.np_, data.nn, data.n0

        if reduce_controls:
            eta_free = np.flatnonzero(np.any(data.eta_min != 0, axis=1) | np.any(data.eta_max != 0, axis=1))
            alpha_free = np.flatnonzero(np.any(data.alpha_max != 0, axis=1))
        else:
            eta_free = np.arange(np_)
            alpha_free = np.arange(nn)
        self.eta_free = eta_free
        self.alpha_free = alpha_free

        # ---------------- decision variables ----------------
        q = ca.MX.sym("q", np_)
        h = ca.MX.sym("h", nn)
        eta_v = ca.MX.sym("eta", len(eta_free))
        alpha_v = ca.MX.sym("alpha", len(alpha_free))
        x = ca.vertcat(q, h, eta_v, alpha_v)
        self.nx = int(x.numel())

        eta = _scatter(eta_v, eta_free, np_)
        alpha = _scatter(alpha_v, alpha_free, nn)

        # ---------------- parameters ----------------
        p_d = ca.MX.sym("d", nn)
        p_h0 = ca.MX.sym("h0", n0)
        p_lin = ca.MX.sym("lin", nn)
        p_target = ca.MX.sym("target", nn)
        p_quad = ca.MX.sym("quad", 1)
        p_const = ca.MX.sym("const", 1)
        p_wazp = ca.MX.sym("w_azp", 1)
        p_wscc = ca.MX.sym("w_scc", 1)
        p = ca.vertcat(p_d, p_h0, p_lin, p_target, p_quad, p_const, p_wazp, p_wscc)

        A12 = ca.DM(data.A12)
        A10 = ca.DM(data.A10)
        r = ca.DM(data.r)
        nexp = ca.DM(data.nexp)
        area = ca.DM(data.area)

        # ---------------- constraints ----------------
        q_reg = q + HEAD_LOSS_REG
        head_loss = r * q_reg * ca.power(ca.fabs(q_reg), nexp - 1.0)
        g_energy = head_loss + A12 @ h + A10 @ p_h0 + eta
        g_mass = A12.T @ q - alpha - p_d

        g_list = [g_energy, g_mass]
        lbg = [np.zeros(np_), np.zeros(nn)]
        ubg = [np.zeros(np_), np.zeros(nn)]

        if len(data.v_loc):
            # Bilinear direction constraint at the control valves: eta * q >= 0.
            g_valve = eta[data.v_loc.tolist()] * q[data.v_loc.tolist()]
            g_list.append(g_valve)
            lbg.append(np.full(len(data.v_loc), EPS_BILINEAR))
            ubg.append(np.full(len(data.v_loc), np.inf))

        g = ca.vertcat(*g_list)
        self._lbg = np.concatenate(lbg)
        self._ubg = np.concatenate(ubg)

        # ---------------- objective ----------------
        velocity = q / 1000.0 * area
        psi_plus = logistic(data.rho * (velocity - data.umin))
        psi_minus = logistic(data.rho * (-velocity - data.umin))

        f_azp = ca.dot(ca.DM(data.azp_weights), h - ca.DM(data.elev))
        f_scc = -ca.dot(ca.DM(data.scc_weights), psi_plus + psi_minus)

        residual = h - p_target
        f_couple = ca.dot(p_lin, residual) + 0.5 * p_quad * ca.dot(residual, residual) + p_const

        f = p_wazp * f_azp + p_wscc * f_scc + f_couple

        nlp = {"x": x, "p": p, "f": f, "g": g}
        options = ipopt_options or default_ipopt_options(
            print_level=print_level, max_iter=max_iter, linear_solver=linear_solver
        )
        self._options = options
        self.solver = ca.nlpsol("primal_update", "ipopt", nlp, options)
        self._nlp = nlp
        self._resto_solver: ca.Function | None = None
        self._mapped: dict[tuple[str, int, int], ca.Function] = {}
        self._g_fun_cache: ca.Function | None = None

    # ------------------------------------------------------------------
    # packing helpers
    # ------------------------------------------------------------------
    def pack(self, x_full: np.ndarray) -> np.ndarray:
        """``[q; h; eta; alpha]`` (full) -> the reduced decision vector."""
        q, h, eta, alpha = self.data.split(np.atleast_2d(x_full))
        return np.vstack([q, h, eta[self.eta_free], alpha[self.alpha_free]])

    def unpack(self, x_reduced: np.ndarray) -> np.ndarray:
        """The reduced decision vector -> full ``[q; h; eta; alpha]``."""
        x_reduced = np.atleast_2d(x_reduced)
        np_, nn = self.data.np_, self.data.nn
        ncol = x_reduced.shape[1]
        q = x_reduced[:np_]
        h = x_reduced[np_ : np_ + nn]
        eta = np.zeros((np_, ncol))
        alpha = np.zeros((nn, ncol))
        n_eta = len(self.eta_free)
        eta[self.eta_free] = x_reduced[np_ + nn : np_ + nn + n_eta]
        alpha[self.alpha_free] = x_reduced[np_ + nn + n_eta :]
        return np.vstack([q, h, eta, alpha])

    def bounds(self, time_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Variable bounds for the given time steps, shape ``(nx, len(t))``."""
        d = self.data
        lb = np.vstack(
            [
                d.q_min[:, time_steps],
                d.h_min[:, time_steps],
                d.eta_min[np.ix_(self.eta_free, time_steps)],
                np.zeros((len(self.alpha_free), len(time_steps))),
            ]
        )
        ub = np.vstack(
            [
                d.q_max[:, time_steps],
                d.h_max[:, time_steps],
                d.eta_max[np.ix_(self.eta_free, time_steps)],
                d.alpha_max[np.ix_(self.alpha_free, time_steps)],
            ]
        )
        return lb, ub

    def parameters(self, coupling: CouplingTerm, time_steps: np.ndarray) -> np.ndarray:
        """Assemble the parameter matrix, one column per time step."""
        d = self.data
        w_azp, w_scc = d.objective_weights()
        return np.vstack(
            [
                d.d[:, time_steps],
                d.h0[:, time_steps],
                coupling.lin[:, time_steps],
                coupling.target[:, time_steps],
                np.full((1, len(time_steps)), coupling.quad),
                coupling.const[time_steps][None, :],
                w_azp[time_steps][None, :],
                w_scc[time_steps][None, :],
            ]
        )

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------
    def _mapped_solver(self, n: int, resto: bool) -> ca.Function:
        key = (self.parallel, n, int(resto))
        if key not in self._mapped:
            solver = self._resto_for(resto)
            if self.parallel in ("thread", "openmp"):
                self._mapped[key] = solver.map(n, self.parallel, min(self.n_threads, n))
            else:
                self._mapped[key] = solver.map(n)
        return self._mapped[key]

    def _resto_for(self, resto: bool) -> ca.Function:
        if not resto:
            return self.solver
        if self._resto_solver is None:
            options = {
                "print_time": False,
                "ipopt": {**self._options["ipopt"], "start_with_resto": "yes", "expect_infeasible_problem": "yes"},
            }
            self._resto_solver = ca.nlpsol("primal_update_resto", "ipopt", self._nlp, options)
        return self._resto_solver

    def solve(
        self,
        x_start: np.ndarray,
        coupling: CouplingTerm,
        time_steps: np.ndarray | None = None,
        resto: bool = False,
        retry_with_resto: bool = True,
    ) -> SolveReport:
        """Solve the local problem for every requested time step.

        Parameters
        ----------
        x_start:
            Full ``[q; h; eta; alpha]`` starting point, shape ``(2np+2nn, nt)``.
        coupling:
            Coupling penalty; see :class:`CouplingTerm`.
        time_steps:
            Subset of time steps to solve; all of them by default.  The
            returned arrays always have one column/entry per requested step.
        resto, retry_with_resto:
            Mirror the Julia fallback: on failure the step is re-solved with
            IPOPT's restoration phase started up front.
        """
        d = self.data
        time_steps = np.arange(d.nt) if time_steps is None else np.asarray(time_steps, dtype=int)
        n = len(time_steps)

        x0 = self.pack(x_start[:, time_steps])
        lbx, ubx = self.bounds(time_steps)
        x0 = np.clip(x0, lbx, ubx)
        p = self.parameters(coupling, time_steps)
        lbg = np.tile(self._lbg[:, None], (1, n))
        ubg = np.tile(self._ubg[:, None], (1, n))

        solver = self._mapped_solver(n, resto)
        result = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        x_sol = np.asarray(result["x"]).reshape(self.nx, n, order="F")
        obj = np.asarray(result["f"]).ravel().astype(float)

        status = np.array([0 if self._accepted(x_sol[:, i], p[:, i]) else 1 for i in range(n)], dtype=int)
        messages: list[str] = []

        if retry_with_resto and not resto and np.any(status != 0):
            failed = np.flatnonzero(status != 0)
            messages.append(f"restoration retry for time steps {time_steps[failed].tolist()}")
            retry = self.solve(
                x_start,
                coupling,
                time_steps=time_steps[failed],
                resto=True,
                retry_with_resto=False,
            )
            x_sol[:, failed] = self.pack(retry.x)
            obj[failed] = retry.objective
            status[failed] = retry.status

        x_full = self.unpack(x_sol)
        x_full[:, status != 0] = np.nan
        obj[status != 0] = np.inf
        return SolveReport(x=x_full, objective=obj, status=status, messages=messages)

    def _accepted(self, x: np.ndarray, p: np.ndarray) -> bool:
        """Accept a solve on primal feasibility rather than on solver flags.

        ``nlpsol`` inside a ``map`` does not expose per-call return codes, so
        the constraint residual is checked directly.  This plays the role of
        the ``LOCALLY_SOLVED``/``ALMOST_LOCALLY_SOLVED`` check in Julia, but is
        computed explicitly and therefore also rejects a "converged" point that
        is not actually feasible.
        """
        if not np.all(np.isfinite(x)):
            return False
        g = np.asarray(self._g_fun(x, p)).ravel()
        violation = np.maximum(self._lbg - g, g - self._ubg)
        return bool(np.max(violation) <= self.feasibility_tol)

    @property
    def _g_fun(self) -> ca.Function:
        if self._g_fun_cache is None:
            self._g_fun_cache = ca.Function("g_fun", [self._nlp["x"], self._nlp["p"]], [self._nlp["g"]])
        return self._g_fun_cache


def _scatter(values: ca.MX, index: np.ndarray, size: int) -> ca.MX:
    """Place ``values`` at ``index`` inside a zero vector of length ``size``."""
    if len(index) == size and np.array_equal(index, np.arange(size)):
        return values
    out = ca.MX.zeros(size)
    if len(index):
        out[index.tolist()] = values
    return out
