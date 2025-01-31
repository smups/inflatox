#  Copyright© 2024 Raúl Wolters(1)
#
#  This file is part of Inflatox.
#
#  Inflatox is free software: you can redistribute it and/or modify it under
#  the terms of the European Union Public License version 1.2 or later, as
#  published by the European Commission.
#
#  Inflatox is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
#  A PARTICULAR PURPOSE. See the European Union Public License for more details.
#
#  You should have received a copy of the EUPL in an/all official language(s) of
#  the European Union along with Inflatox.  If not, see
#  <https://ec.europa.eu/info/european-union-public-licence_en/>.
#
#  (1) Resident of the Kingdom of the Netherlands; agreement between licensor and
#  licensee subject to Dutch law as per article 15 of the EUPL.

import pytest
import sympy
from inflatox import SymbolicCalculation


@pytest.fixture
def angular_model():
    f1, f2 = sympy.symbols("phi_1 phi_2")
    m1, m2, alpha = sympy.symbols("m_1 m_2 alpha")
    v = (alpha / 2) * ((m1 * f1) ** 2 + (m2 * f2) ** 2)
    diag = 6 * alpha / ((1 - f1**2 - f2**2) ** 2)
    metric = [[diag, 0], [0, diag]]
    return SymbolicCalculation.new(
        [f1, f2], metric, v, "[test] angular inflation model"
    )


@pytest.fixture
def trivial_model():
    f1, f2 = sympy.symbols("phi_1 phi_2")
    m1, m2 = sympy.symbols("m_1 m_2")
    v = (m1 * f1) ** 2 + (m2 * f2) ** 2
    metric = [[1, 0], [0, 1]]
    return SymbolicCalculation.new(
        [f1, f2], metric, v, "[test] trivial inflation model"
    )


def test_inner_prod(trivial_model):
    v1 = [1, 0]
    v2 = [0, 1]
    assert sympy.Eq(trivial_model.inner_prod(v1, v2), 0)


def test_normalize(trivial_model):
    a = sympy.symbols("a")
    v = [1, a**2]
    vnorm = trivial_model.normalize(v)
    assert sympy.Eq(trivial_model.inner_prod(vnorm, vnorm), 1).simplify()


def test_trivial_christoffels(trivial_model):
    Gamma = trivial_model.christoffels()
    dim = len(trivial_model.coords)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                assert sympy.Eq(Gamma[a][b][c], 0).simplify()


def test_angular_christoffels(angular_model):
    Gamma = angular_model.christoffels()
    dim = len(angular_model.coords)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                assert sympy.Eq(Gamma[a][b][c], Gamma[a][c][b]).simplify()


def test_gramm_schmidt(trivial_model):
    a, b = sympy.symbols("a b")
    v1 = trivial_model.normalize([1, a**2])
    v2 = [sympy.sqrt(b), sympy.sin(a)]
    v2_ortho = trivial_model.gramm_schmidt([v1], v2)
    assert sympy.Eq(trivial_model.inner_prod(v2_ortho, v2_ortho), 1).simplify()
    assert sympy.Eq(trivial_model.inner_prod(v1, v2_ortho).simplify(), 0).simplify()
