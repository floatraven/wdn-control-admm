"""Centralised (single monolithic NLP) solver — port of ``centralized_solver.jl``.

Solves the whole horizon in one shot: all ``nt`` time steps, all hydraulic
constraints and the pressure-variation constraint appear in a single NLP.  It
is the reference the distributed algorithms are compared against, and it is
also what becomes intractable as the network grows — which is the point the
manuscript makes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np

from .coupling import variability_matrix
from .data import ProblemData
from .nlp import EPS_BILINEAR, HEAD_LOSS_REG, default_ipopt_options, logistic
from .objectives import objective_time_series, pressure_range

__all__ = ["CentralizedOptions", "CentralizedResult", "solve_centralized"]

logger = logging.getLogger(__name__)


@dataclass
class CentralizedOptions:
    """Options mirroring the parameter block of ``centralized_solver.jl``."""

    obj_type: str = "azp-scc"
    pv_type: str = "range"
    pv_active: bool = True
    delta_max: float = 10.0
    #: Extra slack added to the coupling bound; the Julia script sets this from
    #: the constraint violation observed in the ADMM results.
    delta_viol: float = 0.0
    time_steps: np.ndarray | None = None
    max_iter: int = 3000
    time_limit: float = 6 * 60 * 60
    print_level: int = 5
    linear_solver: str = "mumps"
    constr_viol_tol: float = 1e-2
    resto: bool = False
    #: ``centralized_solver.jl`` raises the actuator bound to 25 L/s during the
    #: SCC window before solving; keep that behaviour off by default.
    alpha_override: float | None = None


@dataclass
class CentralizedResult:
    x: np.ndarray
    objective: float
    cpu_time: float
    status: str
    converged: bool
    f_val: np.ndarray
    f_azp: np.ndarray
    f_scc: np.ndarray
    max_violation: float
    options: dict[str, Any]

    def summary(self) -> str:
        state = "solved" if self.converged else f"FAILED ({self.status})"
        return (
            f"[centralized] {state} in {self.cpu_time:.1f} s, objective={self.objective:.3f}, "
            f"sum(f_val)={float(np.sum(self.f_val)):.3f}, max violation={self.max_violation:.3g}"
        )


def solve_centralized(data: ProblemData, options: CentralizedOptions | None = None) -> CentralizedResult:
    """Build and solve the full space-time NLP."""
    options = options or CentralizedOptions()
    steps = np.arange(data.nt) if options.time_steps is None else np.asarray(options.time_steps, dtype=int)
    nt = len(steps)
    np_, nn, n0 = data.np_, data.nn, data.n0

    scc_time = np.array([i for i, t in enumerate(steps) if t in set(data.scc_time.tolist())], dtype=int)
    azp_time = np.setdiff1d(np.arange(nt), scc_time)

    # ---------------- variables ----------------
    q = ca.MX.sym("q", np_, nt)
    h = ca.MX.sym("h", nn, nt)
    eta = ca.MX.sym("eta", np_, nt)
    alpha = ca.MX.sym("alpha", nn, nt)
    variables = [q, h, eta, alpha]

    A12 = ca.DM(data.A12)
    A10 = ca.DM(data.A10)
    r = ca.DM(data.r)
    nexp = ca.DM(data.nexp)
    area = ca.DM(data.area)

    # ---------------- hydraulic constraints ----------------
    q_reg = q + HEAD_LOSS_REG
    head_loss = ca.repmat(r, 1, nt) * q_reg * ca.power(ca.fabs(q_reg), ca.repmat(nexp, 1, nt) - 1.0)
    g_energy = head_loss + A12 @ h + A10 @ ca.DM(data.h0[:, steps]) + eta
    g_mass = A12.T @ q - alpha - ca.DM(data.d[:, steps])

    constraints = [ca.vec(g_energy), ca.vec(g_mass)]
    lbg = [np.zeros(np_ * nt), np.zeros(nn * nt)]
    ubg = [np.zeros(np_ * nt), np.zeros(nn * nt)]

    if len(data.v_loc):
        valves = data.v_loc.tolist()
        g_valve = ca.vec(eta[valves, :] * q[valves, :])
        constraints.append(g_valve)
        lbg.append(np.full(len(valves) * nt, EPS_BILINEAR))
        ubg.append(np.full(len(valves) * nt, np.inf))

    # ---------------- pressure-variation constraint ----------------
    bound = options.delta_max + options.delta_viol
    if options.pv_active and options.pv_type != "none":
        if options.pv_type == "range":
            lower = ca.MX.sym("l", nn)
            upper = ca.MX.sym("u", nn)
            variables += [lower, upper]
            constraints.append(ca.vec(h - ca.repmat(upper, 1, nt)))
            lbg.append(np.full(nn * nt, -np.inf))
            ubg.append(np.zeros(nn * nt))
            constraints.append(ca.vec(ca.repmat(lower, 1, nt) - h))
            lbg.append(np.full(nn * nt, -np.inf))
            ubg.append(np.zeros(nn * nt))
            constraints.append(upper - lower)
            lbg.append(np.full(nn, -np.inf))
            ubg.append(np.full(nn, bound))
        elif options.pv_type == "variation":
            diff = ca.vec(h[:, 1:] - h[:, :-1])
            constraints.append(diff)
            lbg.append(np.full(nn * (nt - 1), -bound))
            ubg.append(np.full(nn * (nt - 1), bound))
        elif options.pv_type == "variability":
            A = ca.DM(variability_matrix(nt))
            constraints.append(ca.sum2((h @ A) * h))
            lbg.append(np.full(nn, -np.inf))
            ubg.append(np.full(nn, bound**2))
        else:  # pragma: no cover - guarded by CentralizedOptions
            raise ValueError(f"unknown pv_type: {options.pv_type!r}")

    # ---------------- objective ----------------
    velocity = q / 1000.0 * ca.repmat(area, 1, nt)
    psi_plus = logistic(data.rho * (velocity - data.umin))
    psi_minus = logistic(data.rho * (-velocity - data.umin))
    azp = ca.DM(data.azp_weights).T @ (h - ca.repmat(ca.DM(data.elev), 1, nt))
    scc = ca.DM(data.scc_weights).T @ (psi_plus + psi_minus)

    if options.obj_type == "azp":
        f = ca.sum2(azp) / nt
    elif options.obj_type == "scc":
        f = -ca.sum2(scc) / nt
    elif options.obj_type == "azp-scc":
        f = ca.sum2(azp[:, azp_time.tolist()]) - ca.sum2(scc[:, scc_time.tolist()])
    else:
        raise ValueError(f"unknown obj_type: {options.obj_type!r}")

    x = ca.vertcat(*[ca.vec(v) for v in variables])
    g = ca.vertcat(*constraints)

    # ---------------- bounds and starting point ----------------
    alpha_max = data.alpha_max[:, steps].copy()
    if options.alpha_override is not None and len(data.y_loc) and len(scc_time):
        alpha_max[np.ix_(data.y_loc, scc_time)] = options.alpha_override

    lbx = [data.q_min[:, steps], data.h_min[:, steps], data.eta_min[:, steps], np.zeros((nn, nt))]
    ubx = [data.q_max[:, steps], data.h_max[:, steps], data.eta_max[:, steps], alpha_max]
    x0 = [data.q_init[:, steps], data.h_init[:, steps], np.zeros((np_, nt)), np.zeros((nn, nt))]
    if options.pv_active and options.pv_type == "range":
        lbx += [np.full((nn, 1), -np.inf), np.full((nn, 1), -np.inf)]
        ubx += [np.full((nn, 1), np.inf), np.full((nn, 1), np.inf)]
        x0 += [data.h_init[:, steps].min(axis=1)[:, None], data.h_init[:, steps].max(axis=1)[:, None]]

    lbx_v = np.concatenate([np.reshape(b, -1, order="F") for b in lbx])
    ubx_v = np.concatenate([np.reshape(b, -1, order="F") for b in ubx])
    x0_v = np.clip(np.concatenate([np.reshape(b, -1, order="F") for b in x0]), lbx_v, ubx_v)

    solver_options = default_ipopt_options(
        print_level=options.print_level,
        max_iter=options.max_iter,
        linear_solver=options.linear_solver,
        resto=options.resto,
        extra={"constr_viol_tol": options.constr_viol_tol, "max_wall_time": float(options.time_limit)},
    )
    solver = ca.nlpsol("centralized", "ipopt", {"x": x, "f": f, "g": g}, solver_options)

    logger.info(
        "centralised NLP: %d variables, %d constraints (nt=%d)", int(x.numel()), int(g.numel()), nt
    )
    start = time.perf_counter()
    solution = solver(
        x0=x0_v,
        lbx=lbx_v,
        ubx=ubx_v,
        lbg=np.concatenate(lbg),
        ubg=np.concatenate(ubg),
    )
    cpu_time = time.perf_counter() - start

    status = str(solver.stats().get("return_status", "unknown"))
    converged = bool(solver.stats().get("success", False)) or status in {
        "Solve_Succeeded",
        "Solved_To_Acceptable_Level",
    }

    x_opt = np.asarray(solution["x"]).ravel()
    offset = 0
    blocks = []
    for shape in [(np_, nt), (nn, nt), (np_, nt), (nn, nt)]:
        size = shape[0] * shape[1]
        blocks.append(x_opt[offset : offset + size].reshape(shape, order="F"))
        offset += size
    q_sol, h_sol, eta_sol, alpha_sol = blocks
    x_sol = np.vstack(blocks)

    f_val, f_azp, f_scc = objective_time_series(data, q_sol, h_sol)
    result = CentralizedResult(
        x=x_sol,
        objective=float(solution["f"]),
        cpu_time=cpu_time,
        status=status,
        converged=converged,
        f_val=f_val,
        f_azp=f_azp,
        f_scc=f_scc,
        max_violation=float(np.max(pressure_range(h_sol)) - options.delta_max)
        if options.pv_active
        else float("nan"),
        options={
            "obj_type": options.obj_type,
            "pv_type": options.pv_type,
            "pv_active": options.pv_active,
            "delta_max": options.delta_max,
            "delta_viol": options.delta_viol,
            "n_time_steps": nt,
        },
    )
    logger.info("%s", result.summary())
    return result
