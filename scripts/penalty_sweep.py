#!/usr/bin/env python3
"""Sweep the penalty parameter for several pressure tolerances.

This is the experiment behind the "objective value / number of iterations
versus penalty parameter" figure of the manuscript: for every pressure
tolerance ``delta`` and every penalty in ``gamma_range`` the algorithm is run to
convergence (or to the iteration cap) and the results are written to
``data/<algorithm>_results/``.

Usage::

    python scripts/penalty_sweep.py --net modena --deltas 20 15 10
    python scripts/penalty_sweep.py --net modena --algorithm two-level --max-iter 300
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wdn_admm import load_problem_data, save_result  # noqa: E402
from wdn_admm.admm import ADMMOptions, StandardADMM, TwoLevelADMM, TwoLevelOptions  # noqa: E402
from wdn_admm.plotting import apply_style, plot_penalty_sweep, save_figure  # noqa: E402
from wdn_admm.results import result_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", default="modena", choices=("modena", "L_town", "bwfl_2022_05_hw"))
    parser.add_argument("--algorithm", default="standard", choices=("standard", "two-level"))
    parser.add_argument("--pv-type", default="range")
    parser.add_argument("--deltas", type=float, nargs="+", default=[20.0, 15.0, 10.0])
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        help="gamma (standard ADMM) or beta (two-level)",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)
    data = load_problem_data(args.net)
    print(data.summary())

    objective = np.full((len(args.penalties), len(args.deltas)), np.nan)
    iterations = np.full((len(args.penalties), len(args.deltas)), np.inf)

    for j, delta in enumerate(args.deltas):
        for i, penalty in enumerate(args.penalties):
            if args.algorithm == "standard":
                solver = StandardADMM(data, args.pv_type, delta, n_threads=args.threads)
                result = solver.solve(ADMMOptions(gamma=penalty, max_iter=args.max_iter))
                kind = "admm"
            else:
                solver = TwoLevelADMM(data, args.pv_type, delta, n_threads=args.threads)
                result = solver.solve(TwoLevelOptions(beta=penalty, max_iter=args.max_iter))
                kind = "two_level"

            print(result.summary())
            objective[i, j] = result.total_objective
            if result.converged:
                iterations[i, j] = result.iterations
            save_result(result, result_path(kind, args.net, args.pv_type, delta, penalty))

    apply_style()
    figure = plot_penalty_sweep(
        args.penalties,
        objective,
        iterations,
        labels=[f"δ = {d:g}" for d in args.deltas],
        max_iterations=args.max_iter,
    )
    for path in save_figure(figure, Path(args.plot_dir) / f"{args.net}_{args.pv_type}_penalty_sweep"):
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
