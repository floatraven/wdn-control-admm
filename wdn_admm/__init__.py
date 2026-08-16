"""Distributed nonconvex optimisation for water-network control.

Python port of the Julia code accompanying *"Distributed nonconvex optimization
for control of water networks with time-coupling constraints"*
(https://doi.org/10.48550/arXiv.2311.05180).

Typical use::

    from wdn_admm import load_problem_data, StandardADMM, ADMMOptions

    data = load_problem_data("modena")
    solver = StandardADMM(data, pv_type="range", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01))
    print(result.summary())
"""

from __future__ import annotations

__version__ = "0.1.0"

from .admm import ADMMOptions, StandardADMM, TwoLevelADMM, TwoLevelOptions
from .centralized import CentralizedOptions, CentralizedResult, solve_centralized
from .coupling import CouplingProjector
from .data import ProblemData, load_problem_data, problem_data_path
from .hydraulics import hydraulic_simulation
from .nlp import CouplingTerm, TimeStepNLP
from .objectives import azp_time_series, objective_time_series, scc_time_series, total_objective
from .results import SolverResult, load_result, result_path, save_result
from .scp import SCPOptions, SCPResult, solve_scp

__all__ = [
    "__version__",
    "ADMMOptions",
    "CentralizedOptions",
    "CentralizedResult",
    "CouplingProjector",
    "CouplingTerm",
    "ProblemData",
    "SCPOptions",
    "SCPResult",
    "SolverResult",
    "StandardADMM",
    "TimeStepNLP",
    "TwoLevelADMM",
    "TwoLevelOptions",
    "azp_time_series",
    "hydraulic_simulation",
    "load_problem_data",
    "load_result",
    "objective_time_series",
    "problem_data_path",
    "result_path",
    "save_result",
    "scc_time_series",
    "solve_centralized",
    "solve_scp",
    "total_objective",
]
