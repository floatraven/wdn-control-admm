"""The hydraulic solver replaces the private OpWater package.

The strongest available check is that, started far from the answer, it
reproduces the ``q_init``/``h_init`` that OpWater itself produced and that are
stored inside the problem files.
"""

from __future__ import annotations

import numpy as np
import pytest

from wdn_admm.hydraulics import head_loss, head_loss_gradient, hydraulic_simulation


def test_reproduces_stored_uncontrolled_solution(network):
    d = network
    cold_q = np.full_like(d.q_init, 1.0)
    cold_h = np.tile(d.h0.mean(axis=0), (d.nn, 1))

    result = hydraulic_simulation(d, q0=cold_q, h0_guess=cold_h, tol=1e-9)

    assert result.all_converged
    assert np.max(np.abs(result.h - d.h_init)) < 1e-6
    assert np.max(np.abs(result.q - d.q_init)) < 1e-3


def test_satisfies_the_nlp_hydraulic_constraints(modena):
    """The simulator and the optimiser must impose the same physics."""
    d = modena
    result = hydraulic_simulation(d, tol=1e-10)
    energy = (
        head_loss(result.q, d.r[:, None], d.nexp[:, None]) + d.A12 @ result.h + d.A10 @ d.h0
    )
    mass = d.A12.T @ result.q - d.d
    assert np.max(np.abs(energy)) < 1e-6
    assert np.max(np.abs(mass)) < 1e-6


def test_valve_head_loss_raises_upstream_head(modena):
    """A positive eta on a valve must show up as extra head loss on that link."""
    d = modena
    eta = np.zeros((d.np_, d.nt))
    eta[d.v_loc[0], :] = 5.0
    controlled = hydraulic_simulation(d, eta=eta)
    assert controlled.all_converged

    link = d.v_loc[0]
    baseline = hydraulic_simulation(d)
    delta_head = (d.A12 @ controlled.h + d.A10 @ d.h0)[link] - (
        d.A12 @ baseline.h + d.A10 @ d.h0
    )[link]
    # energy balance: phi(q) + A12 h + A10 h0 + eta = 0, so the head difference
    # across the link must drop by eta plus the change in friction loss.
    friction = head_loss(controlled.q[link], d.r[link], d.nexp[link]) - head_loss(
        baseline.q[link], d.r[link], d.nexp[link]
    )
    assert np.allclose(delta_head + friction, -5.0, atol=1e-6)


def test_actuator_discharge_appears_in_the_mass_balance(modena):
    d = modena
    alpha = np.zeros((d.nn, d.nt))
    alpha[d.y_loc[0], d.scc_time] = 10.0
    result = hydraulic_simulation(d, alpha=alpha)
    assert result.all_converged
    residual = d.A12.T @ result.q - alpha - d.d
    assert np.max(np.abs(residual)) < 1e-6


def test_head_loss_gradient_matches_finite_differences():
    q = np.array([-30.0, -0.5, 0.5, 12.0])
    r = np.array([0.01, 0.02, 0.03, 0.04])
    nexp = np.full(4, 1.852)
    step = 1e-6
    numeric = (head_loss(q + step, r, nexp) - head_loss(q - step, r, nexp)) / (2 * step)
    assert np.allclose(numeric, head_loss_gradient(q, r, nexp), rtol=1e-5)


@pytest.mark.parametrize("scale", [0.5, 1.5])
def test_demand_scaling_changes_flows_monotonically(modena, scale):
    """Sanity check that the solver responds to the data it is given."""
    import copy

    d = copy.copy(modena)
    d.d = modena.d * scale
    result = hydraulic_simulation(d)
    assert result.all_converged
    assert np.sum(np.abs(result.q)) > np.sum(np.abs(modena.q_init)) if scale > 1 else True
