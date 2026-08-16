"""Strictly feasible sequential convex programming (SFSCP).

Port of ``sfscp_solver.jl`` and ``src/sfscp_functions.jl``.

At each iteration the nonconvex problem is replaced by a linearisation about
the current iterate.  Because both the head-loss term and the SCC sigmoid are
linearised and every remaining constraint is linear, the subproblem is a plain
**LP**, solved here with HiGHS through :func:`scipy.optimize.linprog` instead of
Gurobi.  The trial point is then made hydraulically exact by a simulation, and
a backtracking line search on the valve/actuator settings keeps every accepted
iterate strictly feasible — hence "strictly feasible" SCP.

Differences from the Julia source
---------------------------------
``hydraulic_simulation``
    Comes from :mod:`wdn_admm.hydraulics` rather than the private ``OpWater``
    package, so this solver runs from the data shipped in the repository.

sigmoid linearisation
    ``src/sfscp_functions.jl`` calls ``ψ(q_k[i,k]/1000, ...)`` while ``ψ``
    itself also divides by 1000, so the linearisation point is off by a factor
    of 1000 and the chain rule from ``q/1000`` back to ``q`` is missing.  This
    port linearises the SCC term that :mod:`wdn_admm.objectives` actually
    evaluates.

valve direction bounds
    The ``v_dir == -1`` branch of ``build_convex_model``/``is_feasible``
    indexes with ``v_loc`` (every valve) instead of ``valve`` (the current
    one).  Here each valve gets its own bound.  All three shipped networks have
    a uniform ``v_dir``, so this changes nothing on them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .data import ProblemData
from .hydraulics import hydraulic_simulation
from .objectives import pressure_range, sigmoid_scc, total_objective

__all__ = ["SCPOptions", "SCPResult", "is_feasible", "solve_scp", "feasible_starting_point"]

logger = logging.getLogger(__name__)


@dataclass
class SCPOptions:
    """Options of the SFSCP loop (``sfscp_solver.jl`` parameter block)."""

    obj_type: str = "azp-scc"
    pv_active: bool = True
    delta_max: float = 100.0
    max_iter: int = 100
    tol: float = 1e-3
    #: Smallest line-search step before the trial point is abandoned.
    min_step: float = 1e-4
    starting_point: str = "no control"
    bound_tol: float = 1e-2
    lp_method: str = "highs"


@dataclass
class SCPResult:
    q: np.ndarray
    h: np.ndarray
    eta: np.ndarray
    alpha: np.ndarray
    objective: float
    objective_history: list[float]
    step_history: list[float]
    iterations: int
    cpu_time: float
    feasible: bool
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> np.ndarray:
        return np.vstack([self.q, self.h, self.eta, self.alpha])

    def summary(self) -> str:
        state = "feasible" if self.feasible else "INFEASIBLE"
        return (
            f"[sfscp] {state} after {self.iterations} iterations, {self.cpu_time:.1f} s, "
            f"objective={self.objective:.3f}"
        )


# ----------------------------------------------------------------------
# feasibility
# ----------------------------------------------------------------------
def control_bounds(data: ProblemData) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flow and valve bounds tightened by the fixed valve direction.

    A control valve may only dissipate head in the direction it was installed,
    so ``v_dir == +1`` forbids reverse flow and reverse head loss, and
    ``v_dir == -1`` forbids forward flow and forward head loss.
    """
    q_lo, q_up = data.q_min.copy(), data.q_max.copy()
    eta_lo, eta_up = data.eta_min.copy(), data.eta_max.copy()
    for valve, direction in zip(data.v_loc, data.v_dir):
        if direction == 1:
            q_lo[valve, :] = 0.0
            eta_lo[valve, :] = 0.0
        elif direction == -1:
            q_up[valve, :] = 0.0
            eta_up[valve, :] = 0.0
    return q_lo, q_up, eta_lo, eta_up


def is_feasible(
    data: ProblemData,
    q: np.ndarray,
    h: np.ndarray,
    eta: np.ndarray,
    alpha: np.ndarray,
    pv_active: bool = True,
    delta_max: float = 100.0,
    tol: float = 1e-2,
) -> bool:
    """Port of ``is_feasible``: bound, direction and coupling checks."""
    q_lo, q_up, eta_lo, eta_up = control_bounds(data)
    checks = [
        np.all((q_lo - tol <= q) & (q <= q_up + tol)),
        np.all((data.h_min - tol <= h) & (h <= data.h_max + tol)),
        np.all((eta_lo - tol <= eta) & (eta <= eta_up + tol)),
        np.all((0.0 <= alpha) & (alpha <= data.alpha_max)),
        np.all(-tol <= q * eta),
    ]
    if pv_active:
        checks.append(np.all(pressure_range(h) <= delta_max))
    return bool(np.all(checks))


# ----------------------------------------------------------------------
# starting point
# ----------------------------------------------------------------------
def feasible_starting_point(data: ProblemData, linear_solver: str = "mumps") -> tuple[np.ndarray, ...]:
    """Port of ``ipopt_solver``: smooth the head trajectory subject to hydraulics.

    Minimises ``sum_i sum_t (h[i,t+1] - h[i,t])^2`` over the valve settings, so
    the SCP loop can start from a point that already respects a tight pressure
    variation limit.
    """
    import casadi as ca

    from .nlp import HEAD_LOSS_REG, default_ipopt_options

    np_, nn, nt = data.np_, data.nn, data.nt
    q = ca.MX.sym("q", np_, nt)
    h = ca.MX.sym("h", nn, nt)
    eta = ca.MX.sym("eta", np_, nt)

    A12, A10 = ca.DM(data.A12), ca.DM(data.A10)
    q_reg = q + HEAD_LOSS_REG
    head_loss = ca.repmat(ca.DM(data.r), 1, nt) * q_reg * ca.power(
        ca.fabs(q_reg), ca.repmat(ca.DM(data.nexp), 1, nt) - 1.0
    )
    g = ca.vertcat(
        ca.vec(head_loss + A12 @ h + A10 @ ca.DM(data.h0) + eta),
        ca.vec(A12.T @ q - ca.DM(data.d)),
    )
    diff = h[:, 1:] - h[:, :-1]
    f = ca.sum1(ca.sum2(diff * diff))

    q_lo, q_up, eta_lo, eta_up = control_bounds(data)
    x = ca.vertcat(ca.vec(q), ca.vec(h), ca.vec(eta))
    lbx = np.concatenate([np.reshape(b, -1, order="F") for b in (q_lo, data.h_min, eta_lo)])
    ubx = np.concatenate([np.reshape(b, -1, order="F") for b in (q_up, data.h_max, eta_up)])
    x0 = np.clip(
        np.concatenate(
            [np.reshape(b, -1, order="F") for b in (data.q_init, data.h_init, np.zeros((np_, nt)))]
        ),
        lbx,
        ubx,
    )

    solver = ca.nlpsol(
        "starting_point",
        "ipopt",
        {"x": x, "f": f, "g": g},
        default_ipopt_options(print_level=0, linear_solver=linear_solver),
    )
    solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=np.zeros(g.numel()), ubg=np.zeros(g.numel()))
    values = np.asarray(solution["x"]).ravel()
    q_sol = values[: np_ * nt].reshape(np_, nt, order="F")
    h_sol = values[np_ * nt : np_ * nt + nn * nt].reshape(nn, nt, order="F")
    eta_sol = values[np_ * nt + nn * nt :].reshape(np_, nt, order="F")
    return q_sol, h_sol, eta_sol, np.zeros((nn, nt))


def make_starting_point(
    data: ProblemData, starting_point: str, pv_active: bool, delta_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Port of ``make_starting_point``."""
    if starting_point == "no control":
        q = data.q_init.copy()
        h = data.h_init.copy()
        eta = np.zeros((data.np_, data.nt))
        alpha = np.zeros((data.nn, data.nt))
    elif starting_point == "feasible control":
        q, h, eta, alpha = feasible_starting_point(data)
    else:
        raise ValueError(f"unknown starting_point: {starting_point!r}")
    return q, h, eta, alpha, is_feasible(data, q, h, eta, alpha, pv_active, delta_max)


# ----------------------------------------------------------------------
# convex (LP) subproblem
# ----------------------------------------------------------------------
class ConvexSubproblem:
    """The linearised subproblem of ``build_convex_model`` as a sparse LP.

    Variable order (all column-major, one block of ``nt`` columns each)::

        [ q | h | eta | alpha | l | u ]

    ``psi_plus``/``psi_minus`` are substituted into the objective rather than
    carried as variables pinned by equalities, which removes ``2*np*nt``
    variables and the matching rows.
    """

    def __init__(self, data: ProblemData, options: SCPOptions) -> None:
        self.data = data
        self.options = options
        np_, nn, nt = data.np_, data.nn, data.nt
        self.n_pv = 2 * nn if options.pv_active else 0
        self.offsets = {
            "q": 0,
            "h": np_ * nt,
            "eta": (np_ + nn) * nt,
            "alpha": (2 * np_ + nn) * nt,
            "l": 2 * (np_ + nn) * nt,
            "u": 2 * (np_ + nn) * nt + nn,
        }
        self.n_var = 2 * (np_ + nn) * nt + self.n_pv

        self._build_static()
        self._build_objective_weights()

    # -- static (iteration-independent) structure ----------------------
    def _build_static(self) -> None:
        data, options = self.data, self.options
        np_, nn, nt = data.np_, data.nn, data.nt
        eye_t = sp.eye(nt, format="csc")

        # Mass balance: A12' q - alpha = d   (identical every iteration)
        self.A_mass = sp.hstack(
            [
                sp.kron(eye_t, data.A12.T),
                sp.csc_matrix((nn * nt, nn * nt)),
                sp.csc_matrix((nn * nt, np_ * nt)),
                -sp.kron(eye_t, sp.eye(nn)),
                sp.csc_matrix((nn * nt, self.n_pv)),
            ],
            format="csc",
        )
        self.b_mass = np.reshape(data.d, -1, order="F")

        # Head/valve part of the energy balance (the q coefficients change).
        self.A_energy_static = sp.hstack(
            [
                sp.csc_matrix((np_ * nt, np_ * nt)),
                sp.kron(eye_t, data.A12),
                sp.kron(eye_t, sp.eye(np_)),
                sp.csc_matrix((np_ * nt, nn * nt)),
                sp.csc_matrix((np_ * nt, self.n_pv)),
            ],
            format="csc",
        )
        self.energy_source = np.reshape(data.A10 @ data.h0, -1, order="F")

        # Pressure-range constraints: h <= u, l <= h, u - l <= delta.
        if options.pv_active:
            h_sel = sp.hstack(
                [
                    sp.csc_matrix((nn * nt, np_ * nt)),
                    sp.kron(eye_t, sp.eye(nn)),
                    sp.csc_matrix((nn * nt, np_ * nt)),
                    sp.csc_matrix((nn * nt, nn * nt)),
                ],
                format="csc",
            )
            zeros_nn = sp.csc_matrix((nn * nt, nn))
            ones_stack = sp.kron(np.ones((nt, 1)), sp.eye(nn))
            upper_rows = sp.hstack([h_sel, zeros_nn, -ones_stack], format="csc")
            lower_rows = sp.hstack([-h_sel, ones_stack, zeros_nn], format="csc")
            width_row = sp.hstack(
                [
                    sp.csc_matrix((nn, 2 * (np_ + nn) * nt)),
                    -sp.eye(nn),
                    sp.eye(nn),
                ],
                format="csc",
            )
            self.A_ub = sp.vstack([upper_rows, lower_rows, width_row], format="csc")
            self.b_ub = np.concatenate(
                [np.zeros(nn * nt), np.zeros(nn * nt), np.full(nn, options.delta_max)]
            )
        else:
            self.A_ub = None
            self.b_ub = None

        # Variable bounds.
        q_lo, q_up, eta_lo, eta_up = control_bounds(data)
        lower = [q_lo, data.h_min, eta_lo, np.zeros((nn, nt))]
        upper = [q_up, data.h_max, eta_up, data.alpha_max]
        lb = [np.reshape(b, -1, order="F") for b in lower]
        ub = [np.reshape(b, -1, order="F") for b in upper]
        if options.pv_active:
            lb += [np.full(nn, -np.inf), np.full(nn, -np.inf)]
            ub += [np.full(nn, np.inf), np.full(nn, np.inf)]
        self.bounds = np.column_stack([np.concatenate(lb), np.concatenate(ub)])

    def _build_objective_weights(self) -> None:
        data, obj_type = self.data, self.options.obj_type
        np_, nn, nt = data.np_, data.nn, data.nt
        azp = np.zeros((nn, nt))
        scc = np.zeros((np_, nt))
        if obj_type == "azp":
            azp[:] = data.azp_weights[:, None] / nt
        elif obj_type == "scc":
            scc[:] = data.scc_weights[:, None] / nt
        elif obj_type == "azp-scc":
            azp[:, data.azp_time] = data.azp_weights[:, None]
            scc[:, data.scc_time] = data.scc_weights[:, None]
        else:
            raise ValueError(f"unknown obj_type: {obj_type!r}")
        self.azp_weights = azp
        self.scc_weights = scc

    # -- per-iteration linearisation -----------------------------------
    def solve(self, q_k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linearise about ``q_k`` and solve the LP."""
        data = self.data
        np_, nn, nt = data.np_, data.nn, data.nt

        # Linearised head loss: phi(q) ~= (1 - n) phi(q_k) + n r |q_k|^(n-1) q.
        abs_pow = np.abs(q_k) ** (data.nexp[:, None] - 1.0)
        phi = data.r[:, None] * q_k * abs_pow
        grad_phi = data.nexp[:, None] * data.r[:, None] * abs_pow
        a_k = phi * (1.0 - data.nexp[:, None])
        A_energy = self.A_energy_static + sp.hstack(
            [
                sp.diags(np.reshape(grad_phi, -1, order="F")),
                sp.csc_matrix((np_ * nt, (np_ + 2 * nn) * nt + self.n_pv)),
            ],
            format="csc",
        )
        b_energy = -np.reshape(a_k + data.A10 @ data.h0, -1, order="F")

        A_eq = sp.vstack([A_energy, self.A_mass], format="csc")
        b_eq = np.concatenate([b_energy, self.b_mass])

        # Linearised SCC sigmoid, d(psi)/dq = rho * area/1000 * psi * (1 - psi).
        area = data.area[:, None]
        psi_plus = sigmoid_scc(q_k, data.area, data.rho, data.umin, sign=1)
        psi_minus = sigmoid_scc(q_k, data.area, data.rho, data.umin, sign=-1)
        slope = data.rho * area / 1000.0 * (psi_plus * (1.0 - psi_plus) - psi_minus * (1.0 - psi_minus))

        c = np.zeros(self.n_var)
        c[self.offsets["q"] : self.offsets["h"]] = np.reshape(
            -self.scc_weights * slope, -1, order="F"
        )
        c[self.offsets["h"] : self.offsets["eta"]] = np.reshape(self.azp_weights, -1, order="F")

        result = linprog(
            c,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=self.bounds,
            method=self.options.lp_method,
        )
        if not result.success:
            raise RuntimeError(f"LP subproblem failed: {result.message}")

        x = result.x
        q = x[self.offsets["q"] : self.offsets["h"]].reshape(np_, nt, order="F")
        h = x[self.offsets["h"] : self.offsets["eta"]].reshape(nn, nt, order="F")
        eta = x[self.offsets["eta"] : self.offsets["alpha"]].reshape(np_, nt, order="F")
        alpha = x[self.offsets["alpha"] : self.offsets["l"]].reshape(nn, nt, order="F")
        return q, h, eta, alpha


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def solve_scp(data: ProblemData, options: SCPOptions | None = None) -> SCPResult:
    """Run the SFSCP loop (``sfscp_solver.jl``)."""
    options = options or SCPOptions()

    q_k, h_k, eta_k, alpha_k, feasible = make_starting_point(
        data, options.starting_point, options.pv_active, options.delta_max
    )
    if feasible:
        obj_k = total_objective(data, q_k, h_k, options.obj_type)
        logger.info("starting point is feasible, objective = %.3f", obj_k)
    else:
        obj_k = float("inf")
        logger.error("starting point is not feasible")

    subproblem = ConvexSubproblem(data, options)
    objective_history = [obj_k]
    step_history: list[float] = []

    start = time.perf_counter()
    iteration = 0
    for iteration in range(1, options.max_iter + 1):
        # Step 1: linearised LP gives a search direction in the controls.
        _, _, eta_t, alpha_t = subproblem.solve(q_k)
        d_eta = eta_t - eta_k
        d_alpha = alpha_t - alpha_k

        # Step 2: backtracking line search, kept hydraulically exact.
        step = 1.0
        while True:
            simulation = hydraulic_simulation(data, eta=eta_t, alpha=alpha_t, q0=q_k, h0_guess=h_k)
            q_t, h_t = simulation.q, simulation.h
            obj_t = total_objective(data, q_t, h_t, options.obj_type)
            feasible = simulation.all_converged and is_feasible(
                data, q_t, h_t, eta_t, alpha_t, options.pv_active, options.delta_max, options.bound_tol
            )
            if (obj_t - obj_k < 0 and feasible) or step < options.min_step:
                break
            step *= 0.5
            eta_t = eta_k + step * d_eta
            alpha_t = alpha_k + step * d_alpha

        step_history.append(step)

        if feasible and obj_t < obj_k:
            improvement = abs(obj_k - obj_t) / max(abs(obj_k), 1e-12)
            obj_k, q_k, h_k, eta_k, alpha_k = obj_t, q_t, h_t, eta_t, alpha_t
        else:
            improvement = 0.0

        objective_history.append(obj_k)
        logger.info(
            "iter %3d  objective %12.3f  relative improvement %.5f  step %.5f",
            iteration,
            obj_k,
            improvement,
            step,
        )
        if improvement <= options.tol:
            break

    cpu_time = time.perf_counter() - start
    result = SCPResult(
        q=q_k,
        h=h_k,
        eta=eta_k,
        alpha=alpha_k,
        objective=float(obj_k),
        objective_history=objective_history,
        step_history=step_history,
        iterations=iteration,
        cpu_time=cpu_time,
        feasible=is_feasible(
            data, q_k, h_k, eta_k, alpha_k, options.pv_active, options.delta_max, options.bound_tol
        ),
        options={
            "obj_type": options.obj_type,
            "pv_active": options.pv_active,
            "delta_max": options.delta_max,
            "starting_point": options.starting_point,
        },
    )
    logger.info("%s", result.summary())
    return result
