"""Result containers and ``.npz`` persistence.

The Julia scripts end with ``@save "...jld2" nt np nn x_k x_0 obj_hist ...``.
The Python port stores the same quantities in compressed ``.npz`` archives,
which need no extra dependency and are readable from anywhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import ProblemData
from .objectives import objective_time_series, pressure_range

__all__ = ["SolverResult", "save_result", "load_result", "result_path"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data"


def result_path(
    kind: str,
    net_name: str,
    pv_type: str,
    delta_max: float,
    penalty: float | None = None,
    results_dir: str | Path | None = None,
    suffix: str = ".npz",
) -> Path:
    """File name mirroring the Julia convention, e.g.
    ``data/admm_results/modena_range_10_beta_0.01.npz``."""
    directory = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR / f"{kind}_results"
    stem = f"{net_name}_{pv_type}_{_fmt(delta_max)}"
    if penalty is not None:
        stem += f"_beta_{_fmt(penalty)}"
    return directory / (stem + suffix)


def _fmt(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value)


@dataclass
class SolverResult:
    """Everything the drivers produce, in one place.

    Attributes
    ----------
    x: final stacked iterate ``[q; h; eta; alpha]``, shape ``(2np+2nn, nt)``.
    x0: first iterate (the uncoupled solution), same shape.
    f_val, f_azp, f_scc: objective time series at ``x``.
    f_azp_0, f_scc_0: the same at ``x0``, for the "no pressure-variation
        constraint" reference curves plotted in the paper.
    residuals: iteration history; column meaning depends on the algorithm and
        is documented in ``residual_names``.
    """

    algorithm: str
    net_name: str
    pv_type: str
    delta_max: float

    x: np.ndarray
    x0: np.ndarray

    f_val: np.ndarray
    f_azp: np.ndarray
    f_scc: np.ndarray
    f_azp_0: np.ndarray
    f_scc_0: np.ndarray

    objective_history: np.ndarray
    residuals: np.ndarray
    residual_names: tuple[str, ...]

    iterations: int
    cpu_time: float
    converged: bool
    max_violation: float
    parameters: dict[str, Any] = field(default_factory=dict)
    outer_iterations: int = 0

    # ------------------------------------------------------------------
    @property
    def total_objective(self) -> float:
        return float(np.sum(self.f_val))

    def residual(self, name: str) -> np.ndarray:
        return self.residuals[:, self.residual_names.index(name)]

    def summary(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        line = (
            f"[{self.algorithm}] {self.net_name} pv={self.pv_type} delta={self.delta_max}: "
            f"{status} in {self.iterations} iterations, {self.cpu_time:.1f} s, "
            f"sum(f_val)={self.total_objective:.3f}, max violation={self.max_violation:.3g}"
        )
        if self.outer_iterations:
            line += f", outer iterations={self.outer_iterations}"
        return line

    @classmethod
    def from_iterates(
        cls,
        data: ProblemData,
        algorithm: str,
        x: np.ndarray,
        x0: np.ndarray,
        objective_history: np.ndarray,
        residuals: np.ndarray,
        residual_names: tuple[str, ...],
        iterations: int,
        cpu_time: float,
        converged: bool,
        pv_type: str,
        delta_max: float,
        parameters: dict[str, Any] | None = None,
        outer_iterations: int = 0,
    ) -> "SolverResult":
        q, h, _, _ = data.split(x)
        q0, h0, _, _ = data.split(x0)
        f_val, f_azp, f_scc = objective_time_series(data, q, h)
        _, f_azp_0, f_scc_0 = objective_time_series(data, q0, h0)
        return cls(
            algorithm=algorithm,
            net_name=data.net_name,
            pv_type=pv_type,
            delta_max=float(delta_max),
            x=x,
            x0=x0,
            f_val=f_val,
            f_azp=f_azp,
            f_scc=f_scc,
            f_azp_0=f_azp_0,
            f_scc_0=f_scc_0,
            objective_history=objective_history,
            residuals=residuals,
            residual_names=tuple(residual_names),
            iterations=int(iterations),
            cpu_time=float(cpu_time),
            converged=bool(converged),
            max_violation=float(np.max(pressure_range(h)) - delta_max),
            parameters=parameters or {},
            outer_iterations=int(outer_iterations),
        )


def save_result(result: SolverResult, path: str | Path) -> Path:
    """Write ``result`` to a compressed ``.npz`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    meta = {
        key: payload.pop(key)
        for key in ["algorithm", "net_name", "pv_type", "residual_names", "parameters", "converged"]
    }
    np.savez_compressed(path, _meta=np.array(json.dumps(meta, default=str)), **payload)
    return path


def load_result(path: str | Path) -> SolverResult:
    """Read a ``.npz`` archive written by :func:`save_result`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        meta = json.loads(str(archive["_meta"]))
        fields = {key: archive[key] for key in archive.files if key != "_meta"}
    scalars = {
        "delta_max": float,
        "iterations": int,
        "cpu_time": float,
        "max_violation": float,
        "outer_iterations": int,
    }
    for key, cast in scalars.items():
        if key in fields:
            fields[key] = cast(fields[key])
    meta["residual_names"] = tuple(meta["residual_names"])
    return SolverResult(**meta, **fields)
