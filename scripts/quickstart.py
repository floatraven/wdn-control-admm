#!/usr/bin/env python3
"""A five-minute tour of the package — run this first.

Loads the Modena network, checks the hydraulic model against the stored
uncontrolled solution, runs a short ADMM, and writes the objective time-series
figure.  Nothing here needs a commercial solver.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wdn_admm import ADMMOptions, StandardADMM, hydraulic_simulation, load_problem_data  # noqa: E402
from wdn_admm.objectives import objective_time_series, pressure_range  # noqa: E402
from wdn_admm.plotting import apply_style, plot_objective_time_series, save_figure  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    print("1. loading problem data")
    data = load_problem_data("modena")
    print("   ", data.summary())

    print("2. hydraulic model check (replaces the private OpWater package)")
    cold = hydraulic_simulation(
        data,
        q0=np.full_like(data.q_init, 1.0),
        h0_guess=np.tile(data.h0.mean(axis=0), (data.nn, 1)),
    )
    print(f"    converged: {cold.all_converged}")
    print(f"    max |h - h_init| = {np.abs(cold.h - data.h_init).max():.2e} m")

    print("3. uncontrolled network")
    _, f_azp, f_scc = objective_time_series(data, data.q_init, data.h_init)
    print(f"    AZP {f_azp.mean():.2f} m, SCC {f_scc.mean():.2f} %,"
          f" worst head range {pressure_range(data.h_init).max():.2f} m")

    print("4. standard ADMM -- deliberately stopped after 25 iterations")
    solver = StandardADMM(data, pv_type="range", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01, max_iter=25))
    print("   ", result.summary())
    print("    'NOT converged' is expected here: this run is a 15-second taste.")
    print("    A full run needs ~375 iterations -- see docs/使用说明.md, step 3.")

    print("5. figure")
    apply_style()
    figure = plot_objective_time_series([result], scc_time=data.scc_time)
    for path in save_figure(figure, Path("plots") / "quickstart_modena"):
        print(f"    saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
