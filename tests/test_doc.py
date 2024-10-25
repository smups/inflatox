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

import inflatox
import numpy as np
import sympy as sp


def test_doc_example():
    sp.init_printing()

    # define model
    φ, θ, L, m, φ0 = sp.symbols("φ θ L m φ0")
    fields = [φ, θ]

    V = (1 / 2 * m**2 * (φ - φ0) ** 2).nsimplify()
    g = [[1, 0], [0, L**2 * sp.sinh(φ / L) ** 2]]

    # symbolic calculation
    calc = inflatox.SymbolicCalculation.new(fields, g, V)
    hesse = calc.execute()

    # run the compiler
    out = inflatox.Compiler(hesse).compile()

    # evaluate the compiled potential and Hesse matrix
    from inflatox.consistency_conditions import GeneralisedAL

    anguelova = GeneralisedAL(out)

    params = np.array([1.0, 1.0, 1.0])
    x = np.array([2.0, -2.0])
    assert anguelova.calc_V(x, params) == 0.5
    assert np.allclose(
        anguelova.calc_H(x, params), np.array([[1.0, 0.0], [0.0, 13.6449586]])
    )

    extent = [-1.0, 1.0, -1.0, 1.0]
    consistency_condition, epsilon_V, epsilon_H, eta_H, delta, omega = (
        anguelova.complete_analysis(params, *extent)
    )
