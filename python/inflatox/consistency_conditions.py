#  Copyright© 2023 Raúl Wolters(1)
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

#External imports
from inflatox.compiler import CompilationArtifact
import numpy as np

#Internal imports
from .compiler import CompilationArtifact
from .libinflx_rs import (open_inflx_dylib, anguelova)

#Limit exports to these items
__all__ = ['InflationCondition', 'AnguelovaLazaroiuCondition']

class InflationCondition():
  """Base-class for all inflation conditions. Provides native methods to evaluate
  the potential and projected Hesse matrix. This base-class may be extended either
  by using these native methods, or by including your own native code that hooks
  into the Rust API or C ABI.
  """

  def __init__(self, compiled_artefact: CompilationArtifact):
    self.artefact = compiled_artefact
    self.dylib = open_inflx_dylib(compiled_artefact.shared_object_path)
    
  def calc_V(self, x: np.array, args: np.array) -> float:
    """calculates the scalar potential at field-space coordinates `x` with
    model-specific parameters `args`.

    ### Args
    - `x` (`np.array`): field-space coordinates at which to calculate
    - `args` (`np.array`): values of the model-dependent parameters. See
    `CompilationArtefact.print_sym_lookup_table()` for an overview of which
    sympy symbols were mapped to which args index.

    ### Returns
      `float`: Value of scalar potential with parameters `args` at coordinates `x`
    """
    return self.dylib.potential(x, args)
  
  def calc_H(self, x: np.array, args: np.array) -> np.array:
    """calculates the projected covariant hesse matrix at field-space
    coordinates `x` with model-specific parameters `args`.

    ### Args
    - `x` (`np.array`): field-space coordinates at which to calculate
    - `args` (`np.array`): values of the model-dependent parameters. See
    `CompilationArtefact.print_sym_lookup_table()` for an overview of which
    sympy symbols were mapped to which args index.

    ### Returns
    `np.ndarray`: Components of the projected covariant hesse matrix with
      parameters `args` at coordinates `x`
    """
    return self.dylib.hesse(x, args)

class AnguelovaLazaroiuCondition(InflationCondition):
  """This class extends the generic `InflationCondition` with the potential
  consistency condition from Anguelova and Lazaroiu 2022 paper
  (`arXiv:2210.00031v2`) for rapid-turn, slow-roll (RTSL) inflationary models.

  ### Usage
  To construct an instance of this class, a `CompilationArtefact` is required.
  Such an artifact can be obtained by running an instance of `inflatox.Compiler`
  with a specific model (fieldspace metric + scalar potential). For more info on
  how to use the `Compiler`, see its documentation.
  
  After obtaining the compiled artefact by calling the `.compile()` method on the
  `Compiler` instance, the artefact can be used to construct an instance of this
  class. The artefact contains all the necessary information to evaluate the
  consistency condition.
  
  To run evaluate the consistency condition for various model parameters and
  regions of field-space, use the `.evaluate()` method on an instance of this class
  with the appropriate methods. For more info, see the `.evaluate()` method.
  """
  
  def __init__(self, compiled_artefact: CompilationArtifact):
    super().__init__(compiled_artefact)
    
  def evaluate(self,
    args: np.array,
    x0_start: float,
    x0_stop: float,
    x1_start: float,
    x1_stop: float,
    N_x0: int = 10_000,
    N_x1: int = 10_000
  ) -> np.array:
    """Evaluate the potential consistency condition from Anguelova and Lazaroiu
    2022 paper (`arXiv:2210.00031v2`) for rapid-turn, slow-roll (RTSL)
    inflationary models.
    
    In their paper, the authors claim that RTSL models must satisfy a consistency
    condition:
      3V (V_vv)^2 = (V_vw)^2 V_ww
    Where V_ab are the components of the covariant Hesse matrix projected along
    the vectors v and w, where v is parallel to the gradient of the scalar
    potential V, and w is orthonormal to v.
    
    This function returns the difference between the left-hand-side (lhs) and
    right-hand-side (rhs) of this equation over the specified area in field space:
      out = 3(V_vv / V_vw)^2 - V_ww / V
    The field-space region to be investigated is specified with the arguments of
    this function.

    ### Args
    General: See `CompilationArtefact.print_sym_lookup_table()` for an overview
    of which sympy symbols were mapped to which arguments (args) and fields (x).
    - `args` (`np.array`): values of the model-dependent parameters. 
    - `x0_start` (`float`): minimum value of first field `x[0]`.
    - `x0_stop` (`float`): maximum value of first field `x[0]`.
    - `x1_start` (`float`): minimum value of second field `x[1]`.
    - `y_stop` (`float`): maximum value of second field `x[1]`.
    - `N_x` (`int`, optional): number of steps along `x[0]` axis. Defaults to 10_000.
    - `x1_stop` (`int`, optional): number of steps along `x[1]` axis. Defaults to 10_000.

    ### Returns
      `np.array`: _description_
    """
    #set-up args for anguelova's condition
    x = np.zeros((N_x0, N_x1))
    start_stop = np.array([
      [x0_start, x0_stop],
      [x1_start, x1_stop]
    ])
    
    #evaluate and return
    anguelova(self.dylib, args, x, start_stop)
    return x
  