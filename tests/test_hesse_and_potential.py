import os

import inflatox
from inflatox.consistency_conditions import GeneralisedAL
import numpy as np
import sympy
import pytest

a = 1 / 600
m_phi = 2e-5
m_chi = m_phi * np.sqrt(9)
args = np.array([a, m_chi, m_phi])


@pytest.fixture
def angular_model():
    model = "angular"

    # setup the coordinates
    p, x = sympy.symbols("phi chi")
    coords = [p, x]

    # setup the potential
    mp, mx, a = sympy.symbols("m_phi m_chi alpha")
    potential = a / 2 * ((mp * p) ** 2 + (mx * x) ** 2).nsimplify()

    # setup the metric
    metric_diagonal = 6 * a / (1 - p**2 - x**2) ** 2
    metric = [[0 for _ in range(2)] for _ in range(2)]
    metric[0][0] = metric_diagonal
    metric[1][1] = metric_diagonal

    hesse = inflatox.InflationModelBuilder.new(
        coords,
        metric,
        potential,
        model_name=model,
        silent=True,
    ).build()

    out = inflatox.Compiler(hesse).compile()
    return GeneralisedAL(out)


def test_hesse(angular_model):
    extent = (-1.05, 1.05, -1.05, 1.05)
    N = [100, 100]
    print(angular_model.calc_H(np.array([0.0, 0.0]), args))
    angular_model.calc_H_array(args, *extent, N)


def test_potential(angular_model):
    extent = [(-1.05, 1.05), (-1.05, 1.05)]
    N = [100, 100]
    print(angular_model.calc_V(np.array([0.0, 0.0]), args))
    angular_model.calc_V_array(args, *extent, N)
