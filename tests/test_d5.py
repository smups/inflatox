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

# ************************************************************************************************ #
# Angular Inflation used in Wolters, Iarygina and Achúcarro JCAP07(2024)079
# Model from Kenton and Thomas JCAP02(2015)127 and Chakraborty, Chiovoloni et al. JCAP01(2020)020
# ************************************************************************************************ #

import os

import inflatox
import numpy as np
import sympy
from inflatox.consistency_conditions import GeneralisedAL
from sympy.simplify.radsimp import collect_sqrt

trajectory_dir = f"{os.path.dirname(os.path.abspath(__file__))}/trajectories/"


def test_egno():
    model = "d5"

    # setup model
    r, θ = sympy.symbols("r θ2")
    fields = [r, θ]

    gs, ls, N = sympy.symbols("g_s l_s N")
    mu5, T5, Lt = sympy.symbols("mu5 T5 L_T")

    mu5 = 1 / ((2 * sympy.pi) ** 5 * ls**6)
    T5 = mu5 / gs

    rho, u = sympy.symbols("rho u")
    rho = r / (3 * u)

    H = (
        ((sympy.pi * N * gs * ls**4) / (12 * u**4) * (2 / rho**2 - 2 * sympy.ln(1 / rho**2 + 1)))
        .nsimplify()
        .collect([u, r])
        .expand()
        .powsimp(force=True)
    )

    p, q = sympy.symbols("p q")

    F = (
        (H / 9 * (r**2 + 3 * u**2) ** 2 + (sympy.pi * q * ls**2) ** 2)
        .nsimplify()
        .collect([r, u])
        .expand()
        .powsimp()
    )

    gamma = 4 * sympy.pi**2 * ls**2 * p * q * T5 * gs

    sqrtF = sympy.sqrt(F)
    g00 = (
        collect_sqrt(
            4 * sympy.pi * p * T5 * sqrtF * ((r**2 + 6 * u**2) / (r**2 + p * u**2)),
            evaluate=True,
        )
        .expand()
        .powsimp()
    )
    g11 = (
        collect_sqrt((4 / 6) * sympy.pi * p * T5 * sqrtF * (r**2 + 6 * u**2), evaluate=True)
        .nsimplify()
        .collect([r, u])
        .expand()
        .powsimp()
    )

    metric = [[g00, 0], [0, g11]]

    Phi_min = (
        (
            (5 / 72)
            * (
                81 * (9 * rho**2 - 2) * rho**2
                + 162 * sympy.ln(9 * (rho**2 + 1))
                + -9
                + -160 * sympy.ln(10)
            )
        )
        .nsimplify()
        .collect([u])
        .expand()
        .powsimp()
    )

    a0, a1, b1 = sympy.symbols("a0 a1 b1")
    Phi_h = (
        (
            a0 * (2 / rho**2 - 2 * sympy.ln(1 / rho**2 + 1))
            + 2
            * a1
            * (6 + 1 / rho**2 - 2 * (2 + 3 * rho**2) * sympy.ln(1 + 1 / rho**2))
            * sympy.cos(θ)
            + (b1 / 2) * (2 + 3 * rho**2) * sympy.cos(θ)
        )
        .nsimplify()
        .collect([u, r])
        .expand()
        .powsimp()
    )

    V0 = sympy.symbols("V0")
    potential = (
        V0
        + (4 * sympy.pi * p * T5 / H) * (sympy.sqrt(F) - (ls**2) * sympy.pi * q * gs)
        + gamma * (Phi_min + Phi_h)
    )
    potential = potential.nsimplify().collect([ls, gs]).expand().powsimp()

    hesse = inflatox.SymbolicCalculation.new(
        fields,
        metric,
        potential,
        model_name=model,
        assertions=False,
        silent=True,
        simplify=False,
    ).execute()

    out = inflatox.Compiler(hesse, cleanup=False).compile()
    anguelova = GeneralisedAL(out)

    V0 = -1.17e-8
    N = 1000.0
    gs = 0.01
    ls = 501.961
    u = 50 * ls
    q = 1.0
    p = 5.0
    a0 = 0.001
    a1 = 0.0005
    b1 = 0.001
    parameters = np.array([V0, a0, p, q, u, ls, a1, b1, gs, N])

    r_start, r_stop = 0.0, 36.0
    θ_start, θ_stop = 0.0, 4 * np.pi
    extent = (r_start, r_stop, θ_start, θ_stop)
    N = 120

    # Calculate potential
    anguelova.calc_V_array(parameters, [r_start, r_stop], [θ_start, θ_stop], [N, N])

    # run analysis
    anguelova.complete_analysis(parameters, *extent, *[N, N])

    # run analysis on trajectory
    trajectory = np.loadtxt(f"{trajectory_dir}/d5_trajectory.dat")

    anguelova.complete_analysis_ot(parameters, trajectory)

    # run Anguelova's original condition
    anguelova.consistency_rapidturn(parameters, *extent, *[N, N])
