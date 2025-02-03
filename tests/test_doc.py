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
    # define model
    r, θ, m = sp.symbols("r θ m")
    fields = [r, θ]

    V = (1 / 2 * m**2 * (θ**2 - 2 / (3 * r**2))).nsimplify()
    g = [[0.5, 0], [0, 0.5 * r**2]]

    # symbolic calculation
    model = inflatox.InflationModelBuilder.new(fields, g, V).build()

    # run the compiler
    out = inflatox.Compiler(model).compile()
    out.print_sym_lookup_table()

    # evaluate the compiled potential and Hesse matrix
    from inflatox.consistency_conditions import GeneralisedAL

    anguelova = GeneralisedAL(out)

    params = np.array([1.0])
    x = np.array([2.0, -2.0])
    v_val = anguelova.calc_V(x, params)
    h_val = anguelova.calc_H(x, params)
    print(v_val, h_val)
    assert v_val == 1.9166666666666667
    assert np.allclose(h_val, np.array([[0.41206897, -1.05517241], [-1.05517241, -0.07873563]]))

    extent = [0.0, 2.5, 0.0, np.pi]
    consistency_condition, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis(
        params, *extent
    )

    assert np.nanmax(consistency_condition) <= 1
