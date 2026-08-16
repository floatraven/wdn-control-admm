"""Command-line entry points.

Replaces the "edit the parameter block, then run the script" workflow of the
Julia drivers::

    python -m wdn_admm admm        --net modena --pv-type range --delta 10 --gamma 0.01
    python -m wdn_admm two-level   --net modena --pv-type range --delta 20 --beta 0.1
    python -m wdn_admm centralized --net modena --pv-type range --delta 10
    python -m wdn_admm scp         --net bwfl_2022_05_hw --delta 100
    python -m wdn_admm plot        --net modena --results data/admm_results/*.npz
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np

from .admm import ADMMOptions, StandardADMM, TwoLevelADMM, TwoLevelOptions
from .centralized import CentralizedOptions, solve_centralized
from .coupling import PV_TYPES
from .data import load_problem_data
from .objectives import objective_time_series
from .results import SolverResult, load_result, result_path, save_result
from .scp import SCPOptions, solve_scp

NETWORKS = ("modena", "L_town", "bwfl_2022_05_hw")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--net", default="modena", choices=NETWORKS, help="network name")
    parser.add_argument("--n-v", type=int, default=3, help="number of control valves in the file name")
    parser.add_argument("--n-f", type=int, default=4, help="number of actuators in the file name")
    parser.add_argument("--data-dir", default=None, help="directory holding the .jld2 problem files")
    parser.add_argument("--pv-type", default="range", choices=PV_TYPES, help="time-coupling constraint")
    parser.add_argument("--delta", type=float, default=10.0, help="pressure-variation tolerance [m]")
    parser.add_argument("--threads", type=int, default=8, help="threads for the parallel time-step solves")
    parser.add_argument(
        "--linear-solver", default="mumps", help="IPOPT linear solver (ma57 if you have HSL)"
    )
    parser.add_argument("--out", default=None, help="output .npz path (default: data/<kind>_results/...)")
    parser.add_argument("--no-save", action="store_true", help="do not write a result file")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every iteration")


def _load(args: argparse.Namespace):
    return load_problem_data(args.net, args.n_v, args.n_f, args.data_dir)


def _store(result: SolverResult, args: argparse.Namespace, kind: str, penalty: float | None) -> None:
    if args.no_save:
        return
    path = Path(args.out) if args.out else result_path(kind, args.net, args.pv_type, args.delta, penalty)
    save_result(result, path)
    print(f"saved: {path}")


def cmd_admm(args: argparse.Namespace) -> int:
    data = _load(args)
    solver = StandardADMM(
        data,
        pv_type=args.pv_type,
        delta_max=args.delta,
        n_threads=args.threads,
        linear_solver=args.linear_solver,
    )
    result = solver.solve(
        ADMMOptions(
            gamma=args.gamma,
            gamma_0=args.gamma_0,
            max_iter=args.max_iter,
            eps_primal=args.eps_primal,
            eps_dual=args.eps_dual,
            stopping=args.stopping,
            scaled=args.scaled,
        )
    )
    print(result.summary())
    _store(result, args, "admm", args.gamma)
    return 0 if result.converged else 1


def cmd_two_level(args: argparse.Namespace) -> int:
    data = _load(args)
    solver = TwoLevelADMM(
        data,
        pv_type=args.pv_type,
        delta_max=args.delta,
        n_threads=args.threads,
        linear_solver=args.linear_solver,
    )
    result = solver.solve(
        TwoLevelOptions(
            beta=args.beta,
            gamma=args.gamma,
            omega=args.omega,
            max_iter=args.max_iter,
            eps_3=args.eps_outer,
        )
    )
    print(result.summary())
    _store(result, args, "two_level", args.beta)
    return 0 if result.converged else 1


def cmd_centralized(args: argparse.Namespace) -> int:
    data = _load(args)
    result = solve_centralized(
        data,
        CentralizedOptions(
            obj_type=args.obj_type,
            pv_type=args.pv_type,
            pv_active=not args.no_pv,
            delta_max=args.delta,
            delta_viol=args.delta_viol,
            max_iter=args.max_iter,
            time_limit=args.time_limit,
            print_level=5 if args.verbose else 0,
            linear_solver=args.linear_solver,
        ),
    )
    print(result.summary())
    if not args.no_save:
        path = Path(args.out) if args.out else result_path(
            "centralised", args.net, args.pv_type, args.delta
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=result.x,
            objective=result.objective,
            cpu_time=result.cpu_time,
            f_val=result.f_val,
            f_azp=result.f_azp,
            f_scc=result.f_scc,
            max_violation=result.max_violation,
        )
        print(f"saved: {path}")
    return 0 if result.converged else 1


def cmd_scp(args: argparse.Namespace) -> int:
    data = _load(args)
    result = solve_scp(
        data,
        SCPOptions(
            obj_type=args.obj_type,
            pv_active=not args.no_pv,
            delta_max=args.delta,
            max_iter=args.max_iter,
            tol=args.tol,
            starting_point=args.starting_point,
        ),
    )
    print(result.summary())
    if not args.no_save:
        path = Path(args.out) if args.out else result_path("scp", args.net, "pv", args.delta)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=result.q,
            h=result.h,
            eta=result.eta,
            alpha=result.alpha,
            objective=result.objective,
            objective_history=np.array(result.objective_history),
            cpu_time=result.cpu_time,
        )
        print(f"saved: {path}")
    return 0 if result.feasible else 1


def cmd_plot(args: argparse.Namespace) -> int:
    from . import plotting

    paths = sorted({p for pattern in args.results for p in glob.glob(pattern)})
    if not paths:
        print("no result files matched", file=sys.stderr)
        return 1
    results = [load_result(p) for p in paths]
    data = _load(args)
    for result in results:
        result.parameters.setdefault("scc_time", data.scc_time.tolist())

    plotting.apply_style(dark=args.dark)
    out = Path(args.plot_dir)
    labels = [f"δ = {r.delta_max:g}" for r in results]

    figures = {
        f"{args.net}_{args.pv_type}_azp_scc": plotting.plot_objective_time_series(
            results, labels=labels, scc_time=data.scc_time
        ),
        f"{args.net}_{args.pv_type}_residuals": plotting.plot_residuals(
            results, labels=labels, tolerance=args.eps_primal
        ),
        f"{args.net}_{args.pv_type}_cdf": plotting.plot_pressure_cdf(
            [data.split(r.x)[1] for r in results],
            labels,
            reference=data.h_init,
        ),
    }
    for name, fig in figures.items():
        for written in plotting.save_figure(fig, out / name):
            print(f"saved: {written}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wdn_admm", description="Distributed control of water distribution networks"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    admm = sub.add_parser("admm", help="standard ADMM (admm_standard.jl)")
    _common(admm)
    admm.add_argument("--gamma", type=float, default=0.01, help="penalty parameter")
    admm.add_argument("--gamma-0", type=float, default=0.0, help="penalty in the first iteration")
    admm.add_argument("--max-iter", type=int, default=1000)
    admm.add_argument("--eps-primal", type=float, default=1e-2)
    admm.add_argument("--eps-dual", type=float, default=1e-5)
    admm.add_argument("--stopping", default="primal", choices=("primal", "primal-dual", "boyd"))
    admm.add_argument("--scaled", action="store_true", help="use the scaled dual variable form")
    admm.set_defaults(func=cmd_admm)

    two = sub.add_parser("two-level", help="two-level ALM/ADMM (admm_two_level.jl)")
    _common(two)
    two.add_argument("--beta", type=float, default=0.1, help="initial ALM penalty")
    two.add_argument("--gamma", type=float, default=1.25, help="penalty growth factor")
    two.add_argument("--omega", type=float, default=0.75)
    two.add_argument("--max-iter", type=int, default=1000)
    two.add_argument("--eps-outer", type=float, default=1e-2)
    two.set_defaults(func=cmd_two_level)

    central = sub.add_parser("centralized", help="monolithic NLP (centralized_solver.jl)")
    _common(central)
    central.add_argument("--obj-type", default="azp-scc", choices=("azp", "scc", "azp-scc"))
    central.add_argument("--no-pv", action="store_true", help="drop the coupling constraint")
    central.add_argument("--delta-viol", type=float, default=0.0, help="slack added to the bound")
    central.add_argument("--max-iter", type=int, default=3000)
    central.add_argument("--time-limit", type=float, default=6 * 60 * 60)
    central.set_defaults(func=cmd_centralized)

    scp = sub.add_parser("scp", help="strictly feasible SCP (sfscp_solver.jl)")
    _common(scp)
    scp.add_argument("--obj-type", default="azp-scc", choices=("azp", "scc", "azp-scc"))
    scp.add_argument("--no-pv", action="store_true")
    scp.add_argument("--max-iter", type=int, default=100)
    scp.add_argument("--tol", type=float, default=1e-3)
    scp.add_argument(
        "--starting-point", default="no control", choices=("no control", "feasible control")
    )
    scp.set_defaults(func=cmd_scp)

    plot = sub.add_parser("plot", help="figures (results_plotting.jl)")
    _common(plot)
    plot.add_argument("--results", nargs="+", required=True, help="result .npz files or globs")
    plot.add_argument("--plot-dir", default="plots")
    plot.add_argument("--dark", action="store_true", help="render for a dark background")
    plot.add_argument("--eps-primal", type=float, default=1e-2)
    plot.set_defaults(func=cmd_plot)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stdout,
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
