"""Objective functions must agree with the expressions written in Julia."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wdn_admm.objectives import (
    azp_time_series,
    objective_time_series,
    pressure_range,
    scc_time_series,
    sigmoid_scc,
    total_objective,
)


def _julia_scc(data, q):
    """Literal transcription of the f_scc loop at the end of admm_standard.jl."""
    area = 1.0 / ((math.pi / 4.0) * data.D**2)
    out = np.zeros(data.nt)
    for k in range(data.nt):
        out[k] = sum(
            data.scc_weights[j]
            * (
                (1 + math.exp(-data.rho * ((q[j, k] / 1000 * area[j]) - data.umin))) ** -1
                + (1 + math.exp(-data.rho * (-(q[j, k] / 1000 * area[j]) - data.umin))) ** -1
            )
            for j in range(data.np_)
        )
    return out


def _julia_azp(data, h):
    out = np.zeros(data.nt)
    for k in range(data.nt):
        out[k] = sum(data.azp_weights[i] * (h[i, k] - data.elev[i]) for i in range(data.nn))
    return out


def test_scc_matches_the_julia_expression(modena):
    q = modena.q_init
    assert np.allclose(scc_time_series(modena, q), _julia_scc(modena, q), rtol=1e-12)


def test_azp_matches_the_julia_expression(modena):
    h = modena.h_init
    assert np.allclose(azp_time_series(modena, h), _julia_azp(modena, h), rtol=1e-12)


def test_sigmoid_is_stable_for_extreme_velocities(modena):
    """The tanh form must not overflow where the exp form would."""
    q = np.full((modena.np_, 1), 1e7)
    values = sigmoid_scc(q, modena.area, modena.rho, modena.umin, sign=1)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, 1.0)
    values = sigmoid_scc(-q, modena.area, modena.rho, modena.umin, sign=1)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, 0.0)


def test_sigmoid_equals_the_logistic_definition(modena):
    q = np.linspace(-50, 50, 25)
    area = np.full(25, modena.area[0])
    expected = 1.0 / (1.0 + np.exp(-modena.rho * (q / 1000 * area - modena.umin)))
    assert np.allclose(sigmoid_scc(q, area, modena.rho, modena.umin), expected)


def test_f_val_switches_objective_inside_the_scc_window(modena):
    q, h = modena.q_init, modena.h_init
    f_val, f_azp, f_scc = objective_time_series(modena, q, h)
    assert np.allclose(f_val[modena.scc_time], -f_scc[modena.scc_time])
    assert np.allclose(f_val[modena.azp_time], f_azp[modena.azp_time])


def test_total_objective_variants(modena):
    q, h = modena.q_init, modena.h_init
    f_azp = azp_time_series(modena, h)
    f_scc = scc_time_series(modena, q)

    assert total_objective(modena, q, h, "azp") == pytest.approx(np.sum(f_azp) / modena.nt)
    assert total_objective(modena, q, h, "scc") == pytest.approx(-np.sum(f_scc) / modena.nt)
    assert total_objective(modena, q, h, "azp-scc") == pytest.approx(
        np.sum(f_azp[modena.azp_time]) - np.sum(f_scc[modena.scc_time])
    )


def test_unknown_objective_type_is_rejected(modena):
    with pytest.raises(ValueError, match="obj_type"):
        total_objective(modena, modena.q_init, modena.h_init, "nope")


def test_pressure_range(modena):
    h = np.array([[1.0, 5.0, 3.0], [-2.0, -2.0, -2.0]])
    assert np.allclose(pressure_range(h), [4.0, 0.0])
