# Julia → Python porting notes / Julia → Python 移植说明

This document is the audit trail of the port: what maps to what, what changed,
and why.  Read it before you start modifying the algorithms.

本文件记录移植过程：对应关系、改动内容以及改动原因。修改算法前请先阅读。

---

## 1. File map / 文件对应表

| Julia | Python | Notes |
|---|---|---|
| `src/admm_functions.jl` → `primal_update` | `wdn_admm/nlp.py` → `TimeStepNLP` | JuMP+Ipopt → CasADi+IPOPT |
| `src/admm_functions.jl` → `auxiliary_update` | `wdn_admm/coupling.py` → `auxiliary_update`, `CouplingProjector` | Gurobi/Ipopt → projection |
| `src/admm_functions.jl` → `make_object_data` | *(not ported)* | Needs the private `OpWater` package; the `.jld2` files it produced ship with the repo |
| `src/two_level_functions.jl` → `x_update` | `wdn_admm/nlp.py` → `TimeStepNLP` + `CouplingTerm.two_level` | Same NLP, different coupling term |
| `src/two_level_functions.jl` → `h̄_update` | `wdn_admm/coupling.py` → `h_bar_update` | |
| `src/two_level_functions.jl` → `z_update` | `wdn_admm/coupling.py` → `z_update` | Gurobi QP → closed form |
| `src/two_level_functions.jl` → `λ_update` | `wdn_admm/coupling.py` → `lambda_update` | |
| `admm_standard.jl` | `wdn_admm/admm.py` → `StandardADMM` | |
| `admm_standard_threaded.jl` | `wdn_admm/admm.py` → `StandardADMM` with `stopping="boyd"` | The two scripts differ only in the parallel backend and the stopping rule |
| `admm_two_level.jl` | `wdn_admm/admm.py` → `TwoLevelADMM` | |
| `centralized_solver.jl` | `wdn_admm/centralized.py` | |
| `sfscp_solver.jl` + `src/sfscp_functions.jl` | `wdn_admm/scp.py` | Gurobi LP → HiGHS via `scipy.optimize.linprog` |
| `make_problem_data.jl` | *(not portable)* | Depends on `OpWater` |
| `results_plotting.jl` | `wdn_admm/plotting.py` | PGFPlotsX → Matplotlib |
| — | `wdn_admm/jld2.py`, `wdn_admm/data.py` | Reads the Julia `.jld2` problem files |
| — | `wdn_admm/hydraulics.py` | Replacement for `OpWater.hydraulic_simulation` |
| — | `wdn_admm/objectives.py`, `wdn_admm/results.py`, `wdn_admm/cli.py` | New: shared objective evaluation, result I/O, command line |

## 2. Solver substitutions / 求解器替换

| Julia | Licence | Python | Licence |
|---|---|---|---|
| Ipopt + HSL `ma57` | free + **licensed** linear solver | CasADi's IPOPT + MUMPS | free (`pip install casadi`) |
| Gurobi (LP/QP blocks) | **commercial** | closed-form projection, OSQP, HiGHS | free |
| Mosek / SCS (unused branches) | commercial / free | — | |
| `OpWater` (network + simulation) | **private, unpublished** | `wdn_admm/hydraulics.py` | in-repo |

Nothing in the Python port needs a licence.  If you do have HSL installed, pass
`--linear-solver ma57` to recover the original setting.

Python 版本不需要任何商业许可证。若已安装 HSL，可用 `--linear-solver ma57` 恢复原设置。

## 3. Mathematically equivalent reformulations / 等价重写

These change how the problem is solved, not what is solved.  Each has a test.

以下改动只影响求解方式，不改变问题本身，且每一项都有对应测试。

1. **`psi` elimination** (`nlp.py`, `scp.py`).  JuMP declares `ψ⁺`/`ψ⁻` as
   variables tied down by nonlinear equalities.  Each is an explicit function of
   one flow, so they are substituted into the objective: `2*np` fewer variables
   and `2*np` fewer constraints per time step.
   *Test:* `test_nlp.py::test_reported_objective_equals_the_post_processed_objective`.

2. **Control reduction** (`nlp.py`).  `make_object_data` zeroes the valve bounds
   away from `v_loc` and the actuator bounds away from `y_loc`, so all but
   `n_v + n_f` entries of `eta`/`alpha` are fixed at zero.  Only the free ones
   become variables.  For BWFL this removes 5554 variables per time step.
   *Tests:* `test_data.py::test_controls_are_structurally_zero_outside_their_locations`,
   `test_nlp.py::test_control_reduction_drops_the_fixed_variables`.

3. **`range` projection in closed form** (`coupling.py`).  The Gurobi model
   `min sum (z-t)^2  s.t.  z <= u, l <= z, u - l <= delta` is a Euclidean
   projection.  For a fixed lower level `l` the answer is `clip(t, l, l+delta)`,
   leaving a scalar convex problem in `l` whose derivative is monotone — solved
   by a vectorised bisection to machine precision.  A `2745 x 96` projection
   takes 0.3 s instead of a 527 000-constraint QP.
   *Test:* `test_coupling.py::test_range_projection_matches_the_qp_it_replaces`.

4. **Closed-form `z` and unconstrained `h_bar` updates** (`coupling.py`).  The
   two-level slack block is unconstrained, so the Gurobi model reduces to
   `z = -(lambda + y + rho*(h - h_bar)) / (beta + rho)`.
   *Tests:* `test_coupling.py::test_z_update_matches_its_stationarity_condition`,
   `::test_h_bar_update_matches_its_stationarity_condition`.

5. **Overflow-safe sigmoid** (`objectives.py`, `nlp.py`).  `1/(1+exp(-x))` is
   evaluated as `0.5*(1+tanh(x/2))` (identical) / via `logaddexp` in NumPy, so
   large `rho` cannot overflow.
   *Tests:* `test_objectives.py::test_scc_matches_the_julia_expression`,
   `::test_sigmoid_is_stable_for_extreme_velocities`.

6. **Scaled == unscaled auxiliary update** (`coupling.py`).  Both branches of
   the Julia `auxiliary_update` minimise the same function of `z`, so one
   implementation covers both.
   *Test:* `test_nlp.py::test_scaled_and_unscaled_coupling_agree_when_lambda_is_zero`.

## 4. Bugs in the Julia source, fixed here / 原代码中的缺陷及修正

| Where | Problem | What the port does |
|---|---|---|
| `admm_functions.jl:278-289`, `two_level_functions.jl:266-277` | the `pv_type == "none"` branch builds an objective referring to `model`, which was never constructed — the branch throws | `"none"` is the unconstrained update, which has a closed form (`z = h + lambda/gamma`) |
| `admm_standard.jl:80` | `x_0 = SharedArray(vcat(q_k, h_k, η_k, α_k))` uses undefined names; the line is immediately overwritten on line 81 | dropped |
| `admm_two_level.jl:134` | the restoration retry passes `ρ_k`, which is never defined (only `ρ_m` is) | retry uses the same penalty as the first attempt |
| `sfscp_functions.jl:34-35, 187-188` | the `v_dir == -1` branch writes to `v_loc` (every valve) instead of `valve` (the current one) | each valve gets its own bound.  All three shipped networks have a uniform `v_dir`, so results are unchanged |
| `sfscp_functions.jl:260-261` | `ψ(q_k/1000, ...)` where `ψ` divides by 1000 again, and `∇ψ_q` is a derivative with respect to `q/1000` used as if it were with respect to `q` | linearises the SCC term that `objectives.py` actually evaluates |
| `sfscp_functions.jl:305-306` | `update_convex_model` addresses `convex_model[:linearised_hyd]`, a constraint name that `build_convex_model` never registers | the LP is rebuilt from cached sparse blocks each iteration |

None of these affect the `pv_type="range"` results reported in the manuscript.

以上问题都不影响论文中 `pv_type="range"` 的结果。

## 5. Deliberate behavioural changes / 有意的行为改动

* **Iterate history is off by default.**  `x_hist` in Julia is
  `(2np+2nn)*nt` floats per iteration — 8.5 GB for a 1000-iteration BWFL run.
  Only what the residuals need is kept; pass `store_history=True` for the rest.
* **Parallelism.**  `addprocs(7)` + `@sync @distributed` becomes one threaded
  CasADi `Function.map`, so the problem data is never serialised to workers.
  Set `parallel="serial"` to disable.
* **Acceptance test.**  `nlpsol` inside a `map` does not expose per-call return
  codes, so a solve is accepted on primal feasibility (`feasibility_tol`,
  default `1e-4`) instead of on IPOPT's `LOCALLY_SOLVED` status.  This also
  rejects a "converged" point that is not actually feasible.
* **Index base.**  Every index read from a `.jld2` file (`v_loc`, `y_loc`,
  `scc_time`) is converted from 1-based to 0-based at load time.
* **`v_dir`** is not stored in the problem files; it is recomputed as
  `sign(q_init[v_loc, 0])`, exactly as `make_problem_data.jl` derives it.

## 6. Cross-checks performed / 已完成的交叉验证

| Check | Result |
|---|---|
| Hydraulic solver vs. stored `q_init`/`h_init` (cold start), all three networks | `max abs h error` ≤ 1.1e-8 m |
| `range` projection vs. OSQP solution of the original Gurobi model | agrees to 3e-8; objective never worse |
| SCC/AZP objectives vs. literal transcription of the Julia loops | agree to 1e-12 relative |
| Standard ADMM on Modena, `range`, δ=10, γ=0.01 | converged, 375 iterations, 189 s, `sum(f_val)` = 399.2, max violation 0.50 m |
| Centralised NLP on Modena, same case | solved, `sum(f_val)` = 402.8, max violation 1.2e-7 m |
| SFSCP on Modena, δ=100 | monotone descent 807.2 → 332.3, strictly feasible throughout |

The ADMM objective being slightly below the centralised one while violating the
coupling bound by ~0.5 m is the behaviour the manuscript reports; the
`δviol = 1.24` constant hard-coded in `centralized_solver.jl` is the same
quantity measured from the original Julia runs.

ADMM 的目标值略低于集中式求解，同时耦合约束有约 0.5 m 的违反——这与论文描述一致；
`centralized_solver.jl` 中写死的 `δviol = 1.24` 正是原 Julia 实验测出的同一个量。
