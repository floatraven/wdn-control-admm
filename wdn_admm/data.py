"""Problem data container for the water-network control problem.

The Julia code passes a raw ``Dict`` around (``data["A12"]``, ``data["nt"]``,
...).  Here the same content is wrapped in a dataclass so that shapes, units
and index conventions are checked once, on load, instead of being rediscovered
at every call site.

Index conventions
-----------------
Julia is 1-based; Python is 0-based.  Every index vector read from the
``.jld2`` files (``v_loc``, ``y_loc``, ``scc_time``) is converted to 0-based on
load, and stays 0-based everywhere in this package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .jld2 import load_jld2

__all__ = ["ProblemData", "load_problem_data", "problem_data_path"]

# Repository layout: <repo>/wdn_admm/data.py -> <repo>/data/problem_data
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "problem_data"


def problem_data_path(
    net_name: str,
    n_v: int = 3,
    n_f: int = 4,
    data_dir: str | Path | None = None,
) -> Path:
    """Reproduce the file-naming scheme used by ``make_problem_data.jl``."""
    directory = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return directory / f"{net_name}_nv_{n_v}_nf_{n_f}.jld2"


@dataclass
class ProblemData:
    """Network, hydraulic and optimisation data for one control problem.

    Attributes
    ----------
    elev, azp_weights:
        Node elevation [m] and average-zone-pressure weights, shape ``(nn,)``.
    nexp, r, D, scc_weights:
        Link head-loss exponent, resistance coefficient, diameter [m] and
        self-cleaning-capacity weights, shape ``(np,)``.
    d:
        Nodal demand [L/s], shape ``(nn, nt)``.
    h0:
        Fixed head at source nodes [m], shape ``(n0, nt)``.
    A12, A10:
        Link-to-junction and link-to-source incidence matrices, shapes
        ``(np, nn)`` and ``(np, n0)``.
    q_min, q_max:
        Flow bounds [L/s], shape ``(np, nt)``.
    h_min, h_max:
        Head bounds [m], shape ``(nn, nt)``.
    eta_min, eta_max:
        Valve head-loss bounds [m], shape ``(np, nt)``.
    alpha_max:
        Air-flow / discharge-valve actuator bounds [L/s], shape ``(nn, nt)``.
    v_loc, y_loc:
        0-based indices of controllable valve links and actuator nodes.
    scc_time:
        0-based time steps at which the self-cleaning-capacity objective
        replaces the average-zone-pressure objective.
    rho, umin:
        Sigmoid steepness and minimum self-cleaning velocity [m/s] used by the
        smoothed SCC objective.
    q_init, h_init:
        Uncontrolled hydraulic simulation used as the ADMM starting point.
    """

    net_name: str

    elev: np.ndarray
    nexp: np.ndarray
    d: np.ndarray
    h0: np.ndarray
    A10: sp.csc_matrix
    A12: sp.csc_matrix
    r: np.ndarray
    D: np.ndarray

    q_min: np.ndarray
    q_max: np.ndarray
    h_min: np.ndarray
    h_max: np.ndarray
    eta_min: np.ndarray
    eta_max: np.ndarray
    alpha_max: np.ndarray

    azp_weights: np.ndarray
    scc_weights: np.ndarray

    v_loc: np.ndarray
    y_loc: np.ndarray
    scc_time: np.ndarray

    rho: float
    umin: float

    q_init: np.ndarray
    h_init: np.ndarray

    v_dir: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------
    @property
    def np_(self) -> int:
        """Number of links (``np`` in the Julia code; renamed to avoid NumPy)."""
        return self.A12.shape[0]

    @property
    def nn(self) -> int:
        """Number of junction nodes."""
        return self.A12.shape[1]

    @property
    def n0(self) -> int:
        """Number of fixed-head (source) nodes."""
        return self.A10.shape[1]

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return self.d.shape[1]

    @property
    def area(self) -> np.ndarray:
        """``1 / cross-sectional area`` per link, matching ``A`` in the Julia code.

        The Julia scripts write ``A = 1 ./ ((π/4).*D.^2)``, i.e. the reciprocal
        of the pipe area.  The name is kept for traceability even though it is
        really an inverse area; velocity is then ``q / 1000 * area``.
        """
        return 1.0 / ((math.pi / 4.0) * self.D**2)

    @property
    def azp_time(self) -> np.ndarray:
        """Time steps that use the average-zone-pressure objective."""
        return np.setdiff1d(np.arange(self.nt), self.scc_time)

    def objective_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-time-step on/off switches for the AZP and SCC objective terms."""
        w_azp = np.ones(self.nt)
        w_scc = np.zeros(self.nt)
        w_azp[self.scc_time] = 0.0
        w_scc[self.scc_time] = 1.0
        return w_azp, w_scc

    def x0(self) -> np.ndarray:
        """Stacked starting point ``x = [q; h; eta; alpha]`` of shape ``(2np+2nn, nt)``."""
        return np.vstack(
            [
                self.q_init,
                self.h_init,
                np.zeros((self.np_, self.nt)),
                np.zeros((self.nn, self.nt)),
            ]
        )

    def split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a stacked ``x`` into ``(q, h, eta, alpha)`` blocks."""
        np_, nn = self.np_, self.nn
        return (
            x[:np_],
            x[np_ : np_ + nn],
            x[np_ + nn : 2 * np_ + nn],
            x[2 * np_ + nn :],
        )

    def summary(self) -> str:
        return (
            f"{self.net_name}: np={self.np_}, nn={self.nn}, n0={self.n0}, nt={self.nt}, "
            f"nv={len(self.v_loc)}, nf={len(self.y_loc)}, "
            f"scc_time={self.scc_time.tolist()}"
        )


def _as_2d(value: Any, rows: int, cols: int, name: str) -> np.ndarray:
    array = np.atleast_2d(np.asarray(value, dtype=float))
    if array.shape == (cols, rows) and rows != cols:
        array = array.T
    if array.shape != (rows, cols):
        raise ValueError(f"{name}: expected shape {(rows, cols)}, got {array.shape}")
    return np.ascontiguousarray(array)


def load_problem_data(
    net_name: str,
    n_v: int = 3,
    n_f: int = 4,
    data_dir: str | Path | None = None,
    path: str | Path | None = None,
) -> ProblemData:
    """Load a ``*_nv_*_nf_*.jld2`` problem file into a :class:`ProblemData`.

    Parameters
    ----------
    net_name:
        One of ``"modena"``, ``"L_town"``, ``"bwfl_2022_05_hw"``.
    n_v, n_f:
        Number of control valves and actuators encoded in the file name.
    data_dir:
        Directory holding the ``.jld2`` files; defaults to ``data/problem_data``.
    path:
        Explicit file path, overriding ``net_name``/``data_dir``.
    """
    file_path = Path(path) if path is not None else problem_data_path(net_name, n_v, n_f, data_dir)
    raw = load_jld2(file_path)

    A12 = raw["A12"].tocsc()
    A10 = raw["A10"].tocsc()
    np_, nn = A12.shape
    n0 = A10.shape[1]
    nt = int(raw["nt"])

    if int(raw["np"]) != np_ or int(raw["nn"]) != nn:
        raise ValueError(
            f"Inconsistent sizes in {file_path}: A12 is {A12.shape} but "
            f"np={raw['np']}, nn={raw['nn']}"
        )

    # 1-based Julia indices -> 0-based Python indices.
    v_loc = np.atleast_1d(np.asarray(raw["v_loc"], dtype=int)).ravel() - 1
    y_loc = np.atleast_1d(np.asarray(raw["y_loc"], dtype=int)).ravel() - 1
    scc_time = np.atleast_1d(np.asarray(raw["scc_time"], dtype=int)).ravel() - 1

    q_init = _as_2d(raw["q_init"], np_, nt, "q_init")
    h_init = _as_2d(raw["h_init"], nn, nt, "h_init")

    # `make_problem_data.jl` derives the valve direction from the sign of the
    # uncontrolled flow at the first time step.  It is not stored in the
    # `.jld2` file, so it is recomputed here the same way.
    v_dir = np.sign(q_init[v_loc, 0]).astype(int) if v_loc.size else np.zeros(0, dtype=int)

    data = ProblemData(
        net_name=net_name,
        elev=np.asarray(raw["elev"], dtype=float).ravel(),
        nexp=np.asarray(raw["nexp"], dtype=float).ravel(),
        d=_as_2d(raw["d"], nn, nt, "d"),
        h0=_as_2d(raw["h0"], n0, nt, "h0"),
        A10=A10,
        A12=A12,
        r=np.asarray(raw["r"], dtype=float).ravel(),
        D=np.asarray(raw["D"], dtype=float).ravel(),
        q_min=_as_2d(raw["Qmin"], np_, nt, "Qmin"),
        q_max=_as_2d(raw["Qmax"], np_, nt, "Qmax"),
        h_min=_as_2d(raw["Hmin"], nn, nt, "Hmin"),
        h_max=_as_2d(raw["Hmax"], nn, nt, "Hmax"),
        eta_min=_as_2d(raw["ηmin"], np_, nt, "ηmin"),
        eta_max=_as_2d(raw["ηmax"], np_, nt, "ηmax"),
        alpha_max=_as_2d(raw["αmax"], nn, nt, "αmax"),
        azp_weights=np.asarray(raw["azp_weights"], dtype=float).ravel(),
        scc_weights=np.asarray(raw["scc_weights"], dtype=float).ravel(),
        v_loc=v_loc,
        y_loc=y_loc,
        scc_time=scc_time,
        rho=float(raw["ρ"]),
        umin=float(raw["umin"]),
        q_init=q_init,
        h_init=h_init,
        v_dir=v_dir,
    )
    return data
