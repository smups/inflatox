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

# External imports
import numpy as np

# Internal imports
from .compiler import CompilationArtifact
from .libinflx_rs import open_inflx_dylib, solve_eom_rk4, solve_eom_rkf

__all__ = ["solve_eom"]

class solve_eom:
    def __init__(self, artifact: CompilationArtifact):
        self.artifact = artifact
        self.n_fields = artifact.n_fields
        self.shared_object_path = artifact.shared_object_path

    def calc_eom_array(
        self,
        pars: np.ndarray[float],
        steps: int,
        fields_init: list[float],
        derivatives_init: list[float],
        delta_t: float = 1e-5,
        max_err: float = 1e-6,
        solver: str = "rk4",
    ) -> np.ndarray:
        
        n = self.n_fields
        out = np.zeros((steps, n * 2 + 1))
        out[0, 0:n] = np.array(fields_init)
        out[0, n : 2 * n] = np.array(derivatives_init)

        dylib = open_inflx_dylib(self.shared_object_path, True)
        if solver == "rk4":
            solve_eom_rk4(dylib, pars, out, delta_t)
        else:
            solve_eom_rkf(dylib, pars, out, max_err, delta_t)
        
        return out