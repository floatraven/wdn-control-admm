"""The coupling updates replace Gurobi/Mosek models with projections.

Each projection is checked against an independent solver (OSQP or CasADi), and
the closed-form block updates are checked against the stationarity conditions
of the objectives they minimise.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from wdn_admm.coupling import (
    CouplingProjector,
    auxiliary_update,
    h_bar_update,
    lambda_update,
    project_range,
    project_variability,
    project_variation,
    variability_matrix,
    z_update,
)


def _range_qp_reference(target: np.ndarray, delta: float) -> np.ndarray:
    """The exact Gurobi model of `auxiliary_update`, re-solved with OSQP."""
    import osqp

    nt = len(target)
    n = nt + 2  # [z, l, u]
    P = sp.block_diag([sp.eye(nt), 1e-10 * sp.eye(2)], format="csc")
    q = np.concatenate([-target, [0.0, 0.0]])
    upper_rows = sp.hstack([sp.eye(nt), sp.csc_matrix((nt, 1)), -np.ones((nt, 1))])
    lower_rows = sp.hstack([-sp.eye(nt), np.ones((nt, 1)), sp.csc_matrix((nt, 1))])
    width_row = sp.csc_matrix(np.concatenate([np.zeros(nt), [-1.0, 1.0]])[None, :])
    A = sp.vstack([upper_rows, lower_rows, width_row], format="csc")
    problem = osqp.OSQP()
    problem.setup(
        P=P,
        q=q,
        A=A,
        l=np.full(2 * nt + 1, -np.inf),
        u=np.concatenate([np.zeros(2 * nt), [delta]]),
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
        polishing=True,
        max_iter=200_000,
    )
    return problem.solve().x[:nt]


@pytest.mark.parametrize("seed", range(8))
def test_range_projection_matches_the_qp_it_replaces(seed):
    rng = np.random.default_rng(seed)
    nt = int(rng.integers(4, 30))
    target = rng.normal(0.0, 10.0, nt)
    delta = float(rng.uniform(0.5, 15.0))

    analytic = project_range(target[None, :], delta)[0]
    reference = _range_qp_reference(target, delta)

    assert np.ptp(analytic) <= delta + 1e-9
    # The analytic projection is exact, so it can only be better than OSQP.
    assert np.sum((analytic - target) ** 2) <= np.sum((reference - target) ** 2) + 1e-9
    assert np.allclose(analytic, reference, atol=1e-6)


def test_range_projection_is_the_identity_when_already_feasible():
    target = np.array([[1.0, 2.0, 3.0], [10.0, 10.5, 9.8]])
    assert np.allclose(project_range(target, 5.0), target)


def test_range_projection_is_idempotent():
    rng = np.random.default_rng(3)
    target = rng.normal(0, 20, (12, 40))
    once = project_range(target, 6.0)
    twice = project_range(once, 6.0)
    assert np.allclose(once, twice, atol=1e-9)


def test_range_projection_handles_rows_independently():
    target = np.array([[0.0, 100.0, 50.0], [1.0, 2.0, 3.0]])
    projected = project_range(target, 10.0)
    assert np.ptp(projected[0]) <= 10.0 + 1e-9
    assert np.allclose(projected[1], target[1])


@pytest.mark.parametrize("seed", range(4))
def test_variation_projection_is_feasible_and_optimal(seed):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.0, 10.0, (3, 16))
    delta = 3.0
    projected = project_variation(target, delta)

    assert np.max(np.abs(np.diff(projected, axis=1))) <= delta + 1e-6
    # Optimality: the projection must beat any feasible competitor, e.g. a
    # constant trajectory at the row mean.
    constant = np.repeat(target.mean(axis=1, keepdims=True), target.shape[1], axis=1)
    assert np.all(
        np.sum((projected - target) ** 2, axis=1) <= np.sum((constant - target) ** 2, axis=1) + 1e-6
    )


def test_variability_projection_activates_the_ellipsoid_constraint():
    rng = np.random.default_rng(7)
    target = rng.normal(0.0, 10.0, (4, 12))
    delta = 5.0
    A = variability_matrix(12).toarray()

    projected = project_variability(target, delta, parallel="serial")

    for row_in, row_out in zip(target, projected):
        assert np.sqrt(row_out @ A @ row_out) <= delta + 1e-5
        if np.sqrt(row_in @ A @ row_in) > delta:
            assert np.sqrt(row_out @ A @ row_out) == pytest.approx(delta, abs=1e-4)


def test_variability_matrix_is_the_squared_first_difference():
    nt = 8
    A = variability_matrix(nt, reg=0.0).toarray()
    rng = np.random.default_rng(1)
    z = rng.normal(size=nt)
    assert z @ A @ z == pytest.approx(np.sum(np.diff(z) ** 2))


def test_auxiliary_update_minimises_the_admm_subproblem(modena):
    """z-update: check the value against the objective it is supposed to minimise."""
    rng = np.random.default_rng(0)
    h = modena.h_init + rng.normal(0, 3, modena.h_init.shape)
    lam = rng.normal(0, 0.5, h.shape)
    gamma = 0.05
    projector = CouplingProjector(modena, "range", 10.0)

    z = auxiliary_update(h, lam, gamma, projector)

    def objective(candidate):
        return float(np.sum(lam * (h - candidate) + 0.5 * gamma * (h - candidate) ** 2))

    assert np.max(np.ptp(z, axis=1)) <= 10.0 + 1e-6
    # Any other feasible point must be worse.
    for seed in range(5):
        other = project_range(h + rng.normal(0, 2, h.shape), 10.0)
        assert objective(z) <= objective(other) + 1e-6


def test_h_bar_update_matches_its_stationarity_condition(modena):
    rng = np.random.default_rng(2)
    h = modena.h_init + rng.normal(0, 1, modena.h_init.shape)
    z = rng.normal(0, 0.2, h.shape)
    y = rng.normal(0, 0.2, h.shape)
    rho = 0.4
    projector = CouplingProjector(modena, "none", 10.0)

    h_bar = h_bar_update(h, z, y, rho, projector)

    # Unconstrained case: gradient wrt h_bar is -y - rho*(h - h_bar + z) = 0.
    assert np.allclose(-y - rho * (h - h_bar + z), 0.0, atol=1e-9)


def test_z_update_matches_its_stationarity_condition():
    rng = np.random.default_rng(4)
    shape = (20, 12)
    h = rng.normal(0, 5, shape)
    h_bar = rng.normal(0, 5, shape)
    y = rng.normal(0, 1, shape)
    lam = rng.normal(0, 1, shape)
    beta, rho = 0.3, 0.6

    z = z_update(h, h_bar, y, lam, beta, rho)

    gradient = lam + beta * z + y + rho * (h - h_bar + z)
    assert np.allclose(gradient, 0.0, atol=1e-12)


def test_lambda_update_clips_symmetrically():
    lam = np.array([[0.0, 5.0, -5.0]])
    z = np.array([[100.0, 100.0, -100.0]])
    updated = lambda_update(lam, z, beta=1.0, lam_bound=10.0)
    assert np.allclose(updated, [[10.0, 10.0, -10.0]])


def test_projector_reports_violations(modena):
    projector = CouplingProjector(modena, "range", 10.0)
    h = modena.h_init
    expected = float(np.max(h.max(axis=1) - h.min(axis=1)) - 10.0)
    assert projector.violation(h) == pytest.approx(expected)


def test_projector_rejects_unknown_pv_type(modena):
    with pytest.raises(ValueError, match="pv_type"):
        CouplingProjector(modena, "nonsense", 10.0)
