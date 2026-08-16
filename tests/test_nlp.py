"""The local NLP is the piece the ADMM iterations rest on."""

from __future__ import annotations

import numpy as np
import pytest

from wdn_admm.hydraulics import head_loss
from wdn_admm.nlp import CouplingTerm, TimeStepNLP
from wdn_admm.objectives import objective_time_series


@pytest.fixture(scope="module")
def solved(modena):
    nlp = TimeStepNLP(modena, parallel="serial")
    report = nlp.solve(modena.x0(), CouplingTerm.none(modena.nn, modena.nt))
    return nlp, report


def test_control_reduction_drops_the_fixed_variables(modena):
    nlp = TimeStepNLP(modena, parallel="serial")
    full = 2 * modena.np_ + 2 * modena.nn
    assert nlp.nx == modena.np_ + modena.nn + len(modena.v_loc) + len(modena.y_loc)
    assert nlp.nx < full
    assert np.array_equal(np.sort(nlp.eta_free), np.sort(modena.v_loc))
    assert np.array_equal(np.sort(nlp.alpha_free), np.sort(modena.y_loc))


def test_pack_unpack_round_trip(modena):
    nlp = TimeStepNLP(modena, parallel="serial")
    x = modena.x0()
    x[nlp.eta_free[0], :] = 1.5
    x[2 * modena.np_ + modena.nn + nlp.alpha_free[0], :] = 2.5
    assert np.allclose(nlp.unpack(nlp.pack(x)), x)


@pytest.mark.slow
def test_all_time_steps_solve_and_satisfy_the_physics(solved, modena):
    _, report = solved
    assert report.all_ok
    q, h, eta, alpha = modena.split(report.x)

    energy = head_loss(q, modena.r[:, None], modena.nexp[:, None]) + modena.A12 @ h
    energy = energy + modena.A10 @ modena.h0 + eta
    mass = modena.A12.T @ q - alpha - modena.d
    assert np.max(np.abs(energy)) < 1e-4
    assert np.max(np.abs(mass)) < 1e-4


@pytest.mark.slow
def test_solution_respects_bounds_and_valve_direction(solved, modena):
    _, report = solved
    q, h, eta, alpha = modena.split(report.x)
    tol = 1e-6
    assert np.all(q >= modena.q_min - tol) and np.all(q <= modena.q_max + tol)
    assert np.all(h >= modena.h_min - tol) and np.all(h <= modena.h_max + tol)
    assert np.all(eta >= modena.eta_min - tol) and np.all(eta <= modena.eta_max + tol)
    assert np.all(alpha >= -tol) and np.all(alpha <= modena.alpha_max + tol)
    # Bilinear valve constraint eta * q >= 0.
    assert np.all(eta[modena.v_loc] * q[modena.v_loc] >= -1e-4)


@pytest.mark.slow
def test_reported_objective_equals_the_post_processed_objective(solved, modena):
    """With no coupling term the NLP objective must be exactly f_val."""
    _, report = solved
    q, h, _, _ = modena.split(report.x)
    f_val, _, _ = objective_time_series(modena, q, h)
    assert np.allclose(report.objective, f_val, atol=1e-8)


@pytest.mark.slow
def test_coupling_term_pulls_heads_towards_the_target(modena):
    """A large penalty towards a shifted target must move h in that direction."""
    nlp = TimeStepNLP(modena, parallel="serial")
    base = nlp.solve(modena.x0(), CouplingTerm.none(modena.nn, modena.nt))
    h_base = modena.split(base.x)[1]

    target = np.clip(h_base + 5.0, modena.h_min, modena.h_max)
    coupling = CouplingTerm.standard_admm(target, np.zeros_like(target), gamma=50.0)
    pulled = nlp.solve(base.x, coupling)
    assert pulled.all_ok
    h_pulled = modena.split(pulled.x)[1]
    assert np.mean(h_pulled) > np.mean(h_base)
    assert np.mean(np.abs(h_pulled - target)) < np.mean(np.abs(h_base - target))


def test_scaled_and_unscaled_coupling_agree_when_lambda_is_zero():
    nn, nt = 5, 3
    z = np.arange(nn * nt, dtype=float).reshape(nn, nt)
    lam = np.zeros((nn, nt))
    unscaled = CouplingTerm.standard_admm(z, lam, gamma=2.0, scaled=False)
    scaled = CouplingTerm.standard_admm(z, lam, gamma=2.0, scaled=True)
    assert np.allclose(unscaled.target, scaled.target)
    assert np.allclose(unscaled.lin, scaled.lin)


def test_two_level_coupling_expands_to_the_julia_objective():
    """Check the completed square against the literal Julia expression."""
    rng = np.random.default_rng(0)
    nn, nt = 4, 2
    h = rng.normal(size=(nn, nt))
    h_bar = rng.normal(size=(nn, nt))
    z = rng.normal(size=(nn, nt))
    y = rng.normal(size=(nn, nt))
    lam = rng.normal(size=(nn, nt))
    beta, rho = 0.3, 0.7

    term = CouplingTerm.two_level(h_bar, z, y, lam, beta, rho)
    residual = h - term.target
    ours = np.sum(term.lin * residual, axis=0) + 0.5 * term.quad * np.sum(residual**2, axis=0)
    ours = ours + term.const

    julia = np.sum(
        lam * z + 0.5 * beta * z**2 + y * (h - h_bar + z) + 0.5 * rho * (h - h_bar + z) ** 2,
        axis=0,
    )
    assert np.allclose(ours, julia)
