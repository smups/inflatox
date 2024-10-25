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
# Model from Christodoudilis, Roest and Sfakianakis JCAP11(2019)002
# ************************************************************************************************ #

import os

import inflatox
import numpy as np
import sympy
from inflatox.consistency_conditions import GeneralisedAL

trajectory_dir = f"{os.path.dirname(os.path.abspath(__file__))}/trajectories/"


def test_angular():
    model = "angular"

    # setup the coordinates
    p, x = sympy.symbols("phi chi")
    coords = [p, x]

    # setup the potential
    mp, mx, a = sympy.symbols("m_phi m_chi alpha")
    V = a / 2 * ((mp * p) ** 2 + (mx * x) ** 2).nsimplify()

    # setup the metric
    metric_diagonal = 6 * a / (1 - p**2 - x**2) ** 2
    metric = [[0 for _ in range(2)] for _ in range(2)]
    metric[0][0] = metric_diagonal
    metric[1][1] = metric_diagonal

    hesse = inflatox.SymbolicCalculation.new_from_list(
        coords,
        metric,
        V,
        model_name=model,
        assertions=False,
        simplification_depth=1,
        silent=True,
    ).execute([[0, 1]])

    out = inflatox.Compiler(hesse, cleanup=False).compile()
    anguelova = GeneralisedAL(out)

    a = 1 / 600
    m_phi = 2e-5
    m_chi = m_phi * np.sqrt(9)
    args = np.array([a, m_chi, m_phi])

    extent = (-1.05, 1.05, -1.05, 1.05)
    phi_start, phi_stop = -15.0, 15.0
    chi_start, chi_stop = -5.0, 5.0
    N = 100

    # Calculate potential
    anguelova.calc_V_array(args, [phi_start, chi_start], [phi_stop, chi_stop], [N, N])

    # run analysis
    anguelova.complete_analysis(args, *extent, *[N, N])

    # run analysis on trajectory
    tx = np.load(f"{trajectory_dir}/angular_phix.npy")
    ty = np.load(f"{trajectory_dir}/angular_phiy.npy")
    trajectory = np.column_stack((tx, ty))
    anguelova.complete_analysis_ot(args, trajectory)

    # run Anguelova's original condition
    anguelova.consistency_rapidturn(args, *extent, *[N, N])
