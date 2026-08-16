"""The JLD2 decoding has to undo three Julia conventions; check each one."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from wdn_admm.data import load_problem_data


def test_sizes_are_self_consistent(network):
    d = network
    assert d.A12.shape == (d.np_, d.nn)
    assert d.A10.shape == (d.np_, d.n0)
    assert d.d.shape == (d.nn, d.nt)
    assert d.h0.shape == (d.n0, d.nt)
    assert d.q_init.shape == (d.np_, d.nt)
    assert d.h_init.shape == (d.nn, d.nt)
    for name in ["q_min", "q_max", "eta_min", "eta_max"]:
        assert getattr(d, name).shape == (d.np_, d.nt), name
    for name in ["h_min", "h_max", "alpha_max"]:
        assert getattr(d, name).shape == (d.nn, d.nt), name
    for name in ["elev", "azp_weights"]:
        assert getattr(d, name).shape == (d.nn,), name
    for name in ["nexp", "r", "D", "scc_weights"]:
        assert getattr(d, name).shape == (d.np_,), name


def test_incidence_matrix_is_a_valid_link_node_topology(network):
    """Every link touches one or two junctions, with opposite signs."""
    A12 = network.A12.tocsr()
    per_link = np.diff(A12.indptr)
    assert set(np.unique(per_link)) <= {1, 2}
    values = np.unique(A12.data)
    assert set(values) <= {-1.0, 1.0}
    # Links that touch two junctions must have one +1 and one -1.
    two = np.flatnonzero(per_link == 2)
    sums = np.asarray(A12[two].sum(axis=1)).ravel()
    assert np.allclose(sums, 0.0)


def test_indices_are_zero_based_and_in_range(network):
    d = network
    assert np.all((0 <= d.v_loc) & (d.v_loc < d.np_))
    assert np.all((0 <= d.y_loc) & (d.y_loc < d.nn))
    assert np.all((0 <= d.scc_time) & (d.scc_time < d.nt))
    assert len(d.v_dir) == len(d.v_loc)
    assert set(np.unique(d.v_dir)) <= {-1, 1}


def test_controls_are_structurally_zero_outside_their_locations(network):
    """`make_object_data` zeroes the bounds away from v_loc / y_loc.

    The reduced NLP in wdn_admm.nlp relies on this, so it is asserted here.
    """
    d = network
    off_valve = np.setdiff1d(np.arange(d.np_), d.v_loc)
    assert np.all(d.eta_min[off_valve] == 0)
    assert np.all(d.eta_max[off_valve] == 0)
    off_actuator = np.setdiff1d(np.arange(d.nn), d.y_loc)
    assert np.all(d.alpha_max[off_actuator] == 0)
    # Actuators are only available during the SCC window.
    off_window = np.setdiff1d(np.arange(d.nt), d.scc_time)
    assert np.all(d.alpha_max[:, off_window] == 0)


def test_bounds_are_ordered(network):
    d = network
    assert np.all(d.q_min <= d.q_max)
    assert np.all(d.h_min <= d.h_max)
    assert np.all(d.eta_min <= d.eta_max)
    assert np.all(d.alpha_max >= 0)


def test_starting_point_respects_bounds(modena):
    d = modena
    assert np.all(d.h_init >= d.h_min - 1e-6)
    x0 = d.x0()
    q, h, eta, alpha = d.split(x0)
    assert np.allclose(q, d.q_init)
    assert np.allclose(h, d.h_init)
    assert np.allclose(eta, 0.0)
    assert np.allclose(alpha, 0.0)


def test_objective_weight_switches_are_complementary(network):
    w_azp, w_scc = network.objective_weights()
    assert np.allclose(w_azp + w_scc, 1.0)
    assert np.allclose(w_scc[network.scc_time], 1.0)
    assert np.allclose(w_azp[network.azp_time], 1.0)


def test_area_is_reciprocal_pipe_area(modena):
    expected = 1.0 / (np.pi / 4.0 * modena.D**2)
    assert np.allclose(modena.area, expected)


def test_sparse_matrices_round_trip_to_dense(modena):
    """CSC decoding: column pointers must land the values in the right rows."""
    dense = modena.A12.toarray()
    assert isinstance(modena.A12, sp.csc_matrix)
    # A12' q for a unit flow on link j must credit/debit exactly its end nodes.
    for link in [0, modena.np_ // 2, modena.np_ - 1]:
        q = np.zeros(modena.np_)
        q[link] = 1.0
        assert np.allclose(modena.A12.T @ q, dense[link])


def test_explicit_path_matches_name_lookup(modena):
    from wdn_admm.data import problem_data_path

    other = load_problem_data("modena", path=problem_data_path("modena"))
    assert other.np_ == modena.np_
    assert np.allclose(other.q_init, modena.q_init)
