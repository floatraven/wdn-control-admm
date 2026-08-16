# wdn_control_admm

Code for the manuscript *"Distributed nonconvex optimization for control of
water networks with time-coupling constraints"*
([10.48550/arXiv.2311.05180](https://doi.org/10.48550/arXiv.2311.05180)).

The repository now contains **two implementations**:

* `wdn_admm/` — a Python package (this is the one to use)
* `*.jl`, `src/*.jl` — the original Julia code, kept unchanged as the reference

本仓库现在包含两套实现：Python 包 `wdn_admm/`（推荐使用）与原始 Julia 代码
（保持原样，作为对照参考）。

---

## Why a Python port / 为什么要有 Python 版本

The Julia code needs three things that are hard to get hold of:

| Dependency | Problem |
|---|---|
| Gurobi | commercial licence |
| HSL `ma57` (IPOPT linear solver) | separate licence, must be compiled in |
| `OpWater` | private package, never published |

The Python port needs none of them. IPOPT comes bundled with CasADi, the
LP/QP blocks are solved by closed-form projections, OSQP and HiGHS, and the
hydraulic simulation is reimplemented in `wdn_admm/hydraulics.py` — verified
against the `q_init`/`h_init` that `OpWater` itself produced and that ship
inside the `.jld2` problem files (agreement to 1e-8 m).

Julia 版依赖 Gurobi、HSL `ma57` 和未公开的 `OpWater` 包。Python 版全部不需要：
IPOPT 随 CasADi 一起安装，LP/QP 子问题改用解析投影、OSQP 和 HiGHS，水力仿真在
`wdn_admm/hydraulics.py` 中重新实现，并与数据文件中 `OpWater` 生成的
`q_init`/`h_init` 逐点核对（误差 1e-8 m 量级）。

## Install / 安装

```bash
pip install -r requirements.txt      # or: pip install -e .
python scripts/quickstart.py         # 5-minute tour
```

Python ≥ 3.10. No compiler, no licence server, no solver installation.

## Quick start / 快速开始

```python
from wdn_admm import load_problem_data, StandardADMM, ADMMOptions

data = load_problem_data("modena")            # modena | L_town | bwfl_2022_05_hw
solver = StandardADMM(data, pv_type="range", delta_max=10.0)
result = solver.solve(ADMMOptions(gamma=0.01, max_iter=1000))

print(result.summary())
# [standard-admm] modena pv=range delta=10.0: converged in 375 iterations,
# 189.0 s, sum(f_val)=399.218, max violation=0.498
```

From the command line:

```bash
python -m wdn_admm admm        --net modena --pv-type range --delta 10 --gamma 0.01 -v
python -m wdn_admm two-level   --net modena --pv-type range --delta 20 --beta 0.1 -v
python -m wdn_admm centralized --net modena --pv-type range --delta 10
python -m wdn_admm scp         --net modena --delta 100
python -m wdn_admm plot        --net modena --results "data/admm_results/*.npz"

python scripts/penalty_sweep.py --net modena --deltas 20 15 10
```

## What is being solved / 问题描述

Over a horizon of `nt` time steps, choose valve head losses `eta` and actuator
discharges `alpha` to

* minimise **average zone pressure** (AZP) — less pressure, less leakage — at
  most time steps, and
* maximise **self-cleaning capacity** (SCC) — enough velocity to resuspend
  sediment — during a short window `scc_time`,

subject to the nonlinear hydraulics

```
r_i q_i |q_i|^(n_i-1) + (A12 h)_i + (A10 h0)_i + eta_i = 0     energy
A12' q - alpha = d                                             mass
```

and a **time-coupling constraint** on the head trajectory of every node, which
is what prevents the problem from separating across time:

| `pv_type` | constraint on each node's head trajectory |
|---|---|
| `range` | `max_t h - min_t h <= delta` |
| `variation` | `\|h[t+1] - h[t]\| <= delta` |
| `variability` | `sum_t (h[t+1]-h[t])^2 + reg\|h\|^2 <= delta^2` |
| `none` | — |

ADMM breaks the coupling by copying `h` into an auxiliary variable `z`: the
`x` block splits into `nt` independent NLPs (solved in parallel), and the `z`
block is a projection onto the coupling set.

ADMM 通过把 `h` 复制成辅助变量 `z` 来解耦：`x` 块分裂成 `nt` 个可并行求解的
独立 NLP，`z` 块则是到耦合可行集上的投影。

## Package layout / 包结构

| Module | Contents |
|---|---|
| `wdn_admm/jld2.py` | reads Julia `.jld2` files (column-major, `SparseMatrixCSC`, union types) |
| `wdn_admm/data.py` | `ProblemData` — shapes, units and 0-based indices checked on load |
| `wdn_admm/hydraulics.py` | global-gradient hydraulic simulation (replaces `OpWater`) |
| `wdn_admm/objectives.py` | AZP, SCC, and the `f_val` time series |
| `wdn_admm/nlp.py` | the per-time-step NLP (CasADi/IPOPT), shared by both ADMM variants |
| `wdn_admm/coupling.py` | `z` / `h_bar` / slack / dual updates; the four projections |
| `wdn_admm/admm.py` | `StandardADMM`, `TwoLevelADMM` |
| `wdn_admm/centralized.py` | the monolithic space-time NLP |
| `wdn_admm/scp.py` | strictly feasible sequential convex programming |
| `wdn_admm/results.py` | `SolverResult`, `.npz` save/load |
| `wdn_admm/plotting.py` | Matplotlib figures |
| `wdn_admm/cli.py` | `python -m wdn_admm ...` |

## Networks / 测试网络

| Name | Links | Junctions | Time steps | SCC window |
|---|---|---|---|---|
| `modena` | 317 | 268 | 24 | steps 7–8 |
| `L_town` | 797 | 688 | 96 | steps 38–42 |
| `bwfl_2022_05_hw` | 2816 | 2745 | 96 | steps 38–42 |

Problem data lives in `data/problem_data/*.jld2` and is read directly — no
conversion step, and the Julia code keeps working on the same files.

## Tests / 测试

```bash
pytest                 # fast checks (~10 s)
pytest --run-slow      # also solves full NLPs (~50 s)
```

78 tests covering the JLD2 decoding, the hydraulic model against the stored
reference solution, every projection against the QP it replaces, the objective
functions against a literal transcription of the Julia loops, and the ADMM
drivers end to end.

## Porting notes / 移植说明

**[`docs/PORTING.md`](docs/PORTING.md)** is the audit trail: the file-by-file
map, the solver substitutions, the mathematically equivalent reformulations
(each with the test that pins it), six bugs found in the Julia source, and the
cross-checks that were run.  Read it before modifying the algorithms.

移植细节、求解器替换、等价重写、原代码中发现的六处缺陷以及交叉验证结果，
全部记录在 **[`docs/PORTING.md`](docs/PORTING.md)**。修改算法前请先阅读。

## Original Julia code / 原始 Julia 代码

Instantiate `Project.toml` to reproduce the original environment. The main
scripts are `admm_two_level.jl` and `admm_standard.jl`. The `OpWater` package
used by `make_problem_data.jl` is private and not included; the problem data it
produced is precompiled in `data/problem_data`.
