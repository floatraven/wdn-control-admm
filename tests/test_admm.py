"""End-to-end behaviour of the two ADMM drivers, plus result round-tripping."""

from __future__ import annotations

import numpy as np
import pytest

from wdn_admm.admm import ADMMOptions, StandardADMM, TwoLevelADMM, TwoLevelOptions
from wdn_admm.objectives import pressure_range
from wdn_admm.results import load_result, result_path, save_result


@pytest.mark.slow
def test_standard_admm_reduces_the_primal_residual(modena):
    solver = StandardADMM(modena, pv_type="range", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01, max_iter=12))

    primal = result.residual("primal")
    assert len(primal) == 12
    assert primal[-1] < primal[0] / 5
    assert result.residuals.shape == (12, 2)
    assert result.objective_history.shape == (12, modena.nt)


@pytest.mark.slow
def test_standard_admm_drives_the_head_range_towards_the_tolerance(modena):
    solver = StandardADMM(modena, pv_type="range", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01, max_iter=12))

    _, h, _, _ = modena.split(result.x)
    _, h0, _, _ = modena.split(result.x0)
    assert pressure_range(h).max() < pressure_range(h0).max()


@pytest.mark.slow
def test_first_iterate_is_the_uncoupled_solution(modena):
    """gamma_0 = 0 means iteration 1 solves the problem without any coupling."""
    solver = StandardADMM(modena, pv_type="range", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01, max_iter=2))

    from wdn_admm.nlp import CouplingTerm, TimeStepNLP

    nlp = TimeStepNLP(modena, parallel="serial")
    reference = nlp.solve(modena.x0(), CouplingTerm.none(modena.nn, modena.nt))
    _, h_ref, _, _ = modena.split(reference.x)
    _, h_0, _, _ = modena.split(result.x0)
    assert np.max(np.abs(h_ref - h_0)) < 1e-3


@pytest.mark.slow
def test_two_level_admm_runs_and_tracks_five_residuals(modena):
    solver = TwoLevelADMM(modena, pv_type="range", delta_max=20.0)
    result = solver.solve(TwoLevelOptions(beta=0.1, max_iter=8))

    assert result.residuals.shape == (8, 5)
    assert result.residual_names == ("inner_a", "inner_b", "inner_c", "outer", "slack")
    assert result.outer_iterations >= 1
    assert np.all(np.isfinite(result.f_val))


@pytest.mark.slow
def test_no_coupling_constraint_leaves_the_local_solution_alone(modena):
    """pv_type='none' is the branch that is broken in the Julia original."""
    solver = StandardADMM(modena, pv_type="none", delta_max=10.0)
    result = solver.solve(ADMMOptions(gamma=0.01, max_iter=3))
    assert np.all(np.isfinite(result.x))
    # With an unconstrained auxiliary block the primal residual collapses at once.
    assert result.residual("primal")[-1] < 1e-6


def test_result_round_trips_through_npz(tmp_path, modena):
    from wdn_admm.results import SolverResult

    rng = np.random.default_rng(0)
    x = modena.x0() + rng.normal(0, 0.1, modena.x0().shape)
    result = SolverResult.from_iterates(
        data=modena,
        algorithm="standard-admm",
        x=x,
        x0=modena.x0(),
        objective_history=rng.normal(size=(4, modena.nt)),
        residuals=rng.normal(size=(4, 2)) ** 2,
        residual_names=("primal", "dual"),
        iterations=4,
        cpu_time=1.25,
        converged=True,
        pv_type="range",
        delta_max=10.0,
        parameters={"gamma": 0.01},
    )

    path = save_result(result, tmp_path / "run.npz")
    restored = load_result(path)

    assert restored.algorithm == result.algorithm
    assert restored.net_name == result.net_name
    assert restored.residual_names == result.residual_names
    assert restored.parameters == {"gamma": 0.01}
    assert restored.iterations == 4
    assert restored.cpu_time == pytest.approx(1.25)
    assert restored.converged is True
    assert np.allclose(restored.x, result.x)
    assert np.allclose(restored.residual("dual"), result.residual("dual"))
    assert restored.total_objective == pytest.approx(result.total_objective)


def test_result_path_follows_the_julia_naming_scheme(tmp_path):
    path = result_path("admm", "modena", "range", 10.0, 0.01, results_dir=tmp_path)
    assert path.name == "modena_range_10_beta_0.01.npz"


def test_unknown_stopping_rule_is_rejected(modena):
    solver = StandardADMM(modena, pv_type="range", delta_max=10.0)
    with pytest.raises(ValueError, match="stopping"):
        solver.solve(ADMMOptions(max_iter=1, stopping="nonsense"))
