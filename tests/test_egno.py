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
# Model from Ellis, Garcia, Nanopoulos and Olive JCAP08(2014)044
# ************************************************************************************************ #

import os

import inflatox
import numpy as np
import sympy
from inflatox.consistency_conditions import GeneralisedAL

trajectory_dir = f"{os.path.dirname(os.path.abspath(__file__))}/trajectories/"


def test_egno():
    model = "egno"

    # setup model
    alpha, m, p, c, a = sympy.symbols("alpha m p c a")
    r, θ = sympy.symbols("r θ")
    fields = [r, θ]

    Phi, Phi_Bar, S, S_Bar = sympy.symbols("Phi Phi_B S S_B")
    K = (
        -3 * alpha * sympy.ln(Phi + Phi_Bar - c * (Phi + Phi_Bar - 1) ** 4)
        + (S * S_Bar) / (Phi + Phi_Bar) ** 3
    ).nsimplify()

    superfields = [Phi, S]
    superfields_conjugate = [Phi_Bar, S_Bar]
    metric = [
        [sympy.diff(sympy.diff(K, superfields[b]), superfields_conjugate[a]) for a in range(0, 2)]
        for b in range(0, 2)
    ]
    metric = [
        [g.subs({Phi: r + 1j * θ, Phi_Bar: r - 1j * θ}).nsimplify().simplify() for g in gb]
        for gb in metric
    ]
    metric = [[g.subs({S: 0, S_Bar: 0}).simplify() for g in gb] for gb in metric]
    real_metric = [[metric[0][0], 0], [0, metric[0][0]]]

    potential = (
        (6 * m**2 * r**3 * ((a - r) ** 2 + θ**2))
        / (a**2 * (2 * r - c * (1 - 2 * r) ** 4) ** (3 * alpha))
    ).nsimplify()

    hesse = inflatox.SymbolicCalculation.new(
        fields, real_metric, potential, model_name=model, silent=True, simplify=False
    ).execute()

    out = inflatox.Compiler(hesse, silent=False).compile()
    anguelova = GeneralisedAL(out)

    alpha = 1.0
    a = 0.5
    c = 1000.0
    p = 3.055
    m = 1e-3
    args = np.array([m, a, c, alpha])

    r_start, r_stop = 0.45, 0.55
    θ_start, θ_stop = 0.0, np.pi
    N_r, N_θ = 500, 100
    extent = (0.46, 0.50, θ_start, θ_stop)

    # Calculate potential
    anguelova.calc_V_array(args, [r_start, r_stop], [θ_start, θ_stop], [N_r, N_θ])

    # run analysis
    anguelova.complete_analysis(args, *extent, *[N_r, N_θ])

    # run analysis on trajectory
    tr = np.load(f"{trajectory_dir}/egno_r.npy")
    ttheta = np.load(f"{trajectory_dir}/egno_theta.npy")
    trajectory = np.column_stack((tr, ttheta))
    anguelova.complete_analysis_ot(args, trajectory)

    # run Anguelova's original condition
    anguelova.consistency_rapidturn(args, *extent, *[N_r, N_θ])
