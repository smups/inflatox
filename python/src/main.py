import os
import tempfile
import distutils.ccompiler as ccomp
from datetime import datetime
from sys import version as sys_version
from typing import Literal

import sympy
from sympy import powdenest
from sympy.printing.c import C99CodePrinter
from IPython.display import display, Math
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols
from joblib import Parallel, delayed, cpu_count

class HesseMatrix():  
  def __init__(self,
    components: list[list[sympy.Expr]],
    coordinates: list[sympy.Symbol],
    model_name: str|None
  ):
    self.cmp = components
    self.dim = len(components[0])
    self.coordinates = coordinates
    self.model_name = model_name if model_name is not None else "generic model"
    if len(components[0]) != len(components):
      raise Exception('The Hesse Matrix is square; the provided list was not (number of columns != number of rows)')
    
  @classmethod
  def from_list(cls, components: list[list[sympy.Expr]], model_name: str|None = None):
    cls(components)
    
  @classmethod
  def from_matrix(cls, matrix: sympy.MutableMatrix):
    raise Exception("Not implemented!")

class SymbolicCalculation():
  """This class represents the symbolic calculation of the Hesse matrix of the
  scalar field potential, projected on an orthonormal vielbein basis constructed
  from the supplied potential. Both of these constructs are covariant with respect
  to the supplied metric on the scalar manifold.
  
  ### Usage
  An instance of this `SymbolicCalculation` class may be constructed from a
  user-defined field_metric, a potential and a list of sympy Symbols indicating
  which symbols (used in the definition of the potential and metric) should be
  interpreted as fields. Two different constructors can be used:
  
  1. `SymbolicCalculation.new()` takes a `field_metric` argument as an instance
    of EinsteinPy's `MetricTensor` class.
  2. `SymbolicCalculation.new_from_list()` takes a `field_metric` argument as a
    normal python nested list of sympy expressions. This constructor is especially
    useful when other functionality supplied by the EinsteinPy package is not required.
    
  Calling the `.execute()` method on an instance of this (`SymbolicCalculation`)
  with an appropriate starting point for constructing the vielbein basis will 
  perform the calculation of the components of the covariant Hesse matrix, projected
  onto the vielbein basis.
  """
  
  def __init__(self,
      fields: list[sympy.Symbol],
      field_metric: MetricTensor,
      potential: sympy.Expr,
      model_name: str|None = None
    ):
    self.coords = fields
    self.g = field_metric
    self.V = potential
    self.model_name = model_name
    
  @classmethod
  def new(cls, fields: list[sympy.Symbol], field_metric: MetricTensor, potential: sympy.Expr):
    """Constructs an instance of `SymbolicCalculation`

    ### Args
    - `fields` (`list[sympy.Symbol]`): sympy symbols that will be interpreted as
    fields during the symbolic calculation.
    - field_metric (`einsteinpy.symbolic.MetricTensor`): metric tensor on the scalar
    manifold
    - potential (`sympy.Expr`): potential for the scalar fields

    ### Returns
    `SymbolicCalculation`
    """
    return cls(fields, field_metric, potential)
  
  @classmethod
  def new_from_list(cls, fields: list[sympy.Symbol], field_metric: list[list[sympy.Symbol]], potential: sympy.Expr):
    """Constructs an instance of `SymbolicCalculation`

    ### Args
    - `fields` (`list[sympy.Symbol]`): sympy symbols that will be interpreted as
    fields during the symbolic calculation.
    - field_metric (`einsteinpy.symbolic.MetricTensor`): metric tensor on the scalar
    manifold
    - potential (`sympy.Expr`): potential for the scalar fields

    ### Returns
    `SymbolicCalculation`
    """
    metric = MetricTensor(field_metric, fields, "scalar manifold metric")
    return cls(fields, metric, potential)
    
  def execute(self, basis: list[list[sympy.Expr]]) -> HesseMatrix:
    """_summary_

    ### Args:
    `basis` (`list[list[sympy.Expr]]`): _description_

    ### Returns:
    `HesseMatrix`: _description_
    """
    dim = len(self.coords)
    assert(len(basis) == dim - 1)
    
    #(1) Calculate an orthonormal basis
    print("Calculating orthonormal basis...")
    w = [self.calc_v()]
    display(Math(f"v_0={sympy.latex(w[0])}"))
    for (i, guess) in enumerate(basis):
      w.append(self.calc_next_w(w, guess))
      display(Math(f"v_{i+1}={sympy.latex(w[i])}"))
    
    #(1b) make sure the basis is orthonormal
    for a in range(dim):
      for b in range(dim):
        if a == b:
          assert(sympy.Eq(1, self.inner_prod(w[a], w[b])).simplify())
        else:
          assert(sympy.Eq(0, self.inner_prod(w[a], w[b])).simplify())
        
    #(2) Calculate the components of the covariant Hesse Matrix
    print("Calculating covariant Hesse matrix...")
    H = self.calc_hesse()
    display(Math(f"H={sympy.latex(sympy.Matrix(H))}"))
    
    #(3) Project Hesse matrix
    print("Projecting the Hesse matrix on the vielbein basis...")
    def process(a:int, b:int):
      return ([a, b], self.project_hesse(H, w[a], w[b]))
    H_proj = [[0 for _ in range(dim)] for _ in range(dim)]
    results = Parallel(n_jobs=cpu_count())(delayed(process)(a, b) for a in range(dim) for b in range(dim))
    
    #(3b) print projected components of the Hesse matrix
    for (idx, component) in results:
        a, b = idx
        H_proj[a][b] = component
        display(Math(f"H_{{{a}{b}}}={sympy.latex(component)}"))
    return HesseMatrix(H_proj, self.coords, self.model_name)
   
  def inner_prod(self, v1: list[sympy.Expr], v2: list[sympy.Expr]) -> sympy.Expr:
    """returns the inner product of vec1 and vec2 with respect to configured metric

    ### Args
    - `v1` (`list[sympy.Expr]`): first vector, once contravariant
    - `v1` (`list[sympy.Expr]`): second vector, once contravariant

    ### Returns
    `sympy.Expr`: inner product of vec1 with vec2 with respect to the configured
      metric tensor of the current instance
    """
    ans = 0
    dim = len(v1)
    assert(dim == len(v2))
    for a in range(dim):
      for b in range(dim):
        ans = ans + (v1[a] * v2[b] * self.g.arr[a][b])
    return powdenest(ans, force=True).simplify()

  def normalize(self, vec: list[sympy.Expr]) -> list[sympy.Expr]:
    """normalizes the input vector with respect to the configured metric tensor

    ### Args
    vec (`list[sympy.Expr]`): components of the vector to be normalised

    ### Returns
    `list[sympy.Expr]`: normalized components of the supplied vector vec with
      respect to the metric tensor of the current instance 
    """
    norm = sympy.sqrt(self.inner_prod(vec, vec))
    return [(cmp / norm).simplify() for cmp in vec]
    
  def calc_hesse(self) -> list[list[sympy.Expr]]:
    """returns the components of the covariant Hesse matrix in a twice-covariant
    form. Components for all pairs of the supplied coordinates are calculated for
    the scalar potential V using the supplied metric tensor.
    
    ### Precise formulation of calculated quantities
    The components of the covariant Hesse matrix are defined as:
      V_ab(ϕ) = ∇_a ∇_b V(ϕ)
    This is expanded as (using Einstein notation):
      V_ab(ϕ) = ∇_a (∂_b V(ϕ)) = ∂_a ∂_b V(ϕ) - Γ_ab^c(ϕ) ∂_c V(ϕ)
    Where Γ_ab^c(ϕ) is the standard Christoffel connection defined as:
      Γ_ab^c = 1/2 g^cd (∂_a g_bd + ∂_b g_ad - ∂_d g_ab)

    ### Returns
    `list[list[sympy.Expr]]`: nested list of components of the Hesse matrix
    """
    dim = len(self.coords)
    #The connection has indices up-down-down (opposite order that we usually use)
    conn = ChristoffelSymbols.from_metric(self.g).tensor()
    #output components of the Hesse matrix
    Vab = [[0 for _ in range(dim)] for _ in range(dim)]
    
    for a in range(dim):
      for b in range(dim):
        #Calculate ∂_a ∂_b V(ϕ)
        da_dbV = sympy.diff(self.V, self.coords[b], self.coords[a]).simplify()
        #Calculate the contraction Γ_ab^c(ϕ) ∂_c V(ϕ)
        gamma_ab = 0
        for c in range(dim):
          gamma_ab = (gamma_ab + conn[c][b][a]*sympy.diff(self.V, self.coords[c])).simplify()
        #set the output components
        Vab[a][b] = powdenest((da_dbV - gamma_ab).simplify(), force=True)
    return Vab

  def calc_v(self) -> list[sympy.Expr]:
    """calculates a normalized vector pointing in the direction of the gradient of
    the configured scalar potential of the current instance
    
    ### Precise formulation of calculated quantities
    The contravariant components of the gradient of V are given by:
      (grad V)^a(ϕ) = g^ab(ϕ) ∂_b V(ϕ)

    ### Returns
    `list[sympy.Expr]`: contravariant components of normalized gradient vector v.
    """
    dim = len(self.coords)
    #non-normalised components of grad V
    v = [sympy.diff(self.V, φ) for φ in self.coords]  
    
    #contract v with the inverse of the metric tensor
    for a in range(dim):
      for b in range(dim):
        v[a] = (v[a] + self.g.inv().arr[a][b] * v[a]).simplify()
    
    #normalize v
    v = self.normalize(v)
    return [powdenest(va, force=True).simplify() for va in v]

  def calc_next_w(self, current_basis: list[list[sympy.Expr]], guess: list[sympy.Expr]) -> list[sympy.Expr]:
    """Use the Gramm-Schmidt procedure to find a new orthogonal basis vector given
    an incomplete set of orthogonal basis vectors and a third vector that is linearly
    independent from the other vectors.

    ### Args
    - `current_basis` (`list[list[sympy.Expr]]`): list of current *orthogonal*
      basisvectors. The components of the vectors should be given in
      *contravariant* form.
    - `guess` (`list[sympy.Expr]`): vector that is linearly independent from the
      (incomplete) set of current basis vectors. The components of this vector
      should be given in *contravariant* form. This vector needn't be
      normalized nor orthogonal to the set of current basis vectors.
    
    ### Precise formulation of calculated quantities
    The Gramm-Schmidt procedure starts with a(n incomplete) set of orthonormal
    basis vectors x_i and new vector y that is linearly independent of all x_i. We
    then simply subtract the overlap between y and each basisvector to obtain a
    vector x_i+1 that is orthogonal to all x_i:
      x_i+1 = y - Σ_a g_ij x^i_a y^j
    The final step is to normalise x_i+1

    ### Returns
    `list[sympy.Expr]`: list of the contravariant components of an additional
      basis vector orthogonal to the supplied set of basis vectors, with respect
      to the supplied metric.
    """
    dim = len(current_basis[0])
    #make sure the supplied basis is not already complete
    assert(len(current_basis) < dim)
    
    #start with vector y
    y = guess
    #subtract the overlap of each current basis vector from y
    for x in current_basis:
      #first assert that vec is actually normalised
      #assert(sympy.Eq(inner_prod(x, x, g), 1))
      xy = self.inner_prod(x, guess) #we have to use guess here, not y!
      for a in range(dim):
        y[a] = (y[a] - xy*x[a]).simplify()    
    #normalize y
    y = self.normalize(y)
    return [powdenest(ya, force=True).simplify() for ya in y]

  def project_hesse(self, hesse_matrix: list[list[sympy.Expr]], vec1: list[sympy.Expr], vec2: list[sympy.Expr]) -> sympy.Expr:
    """_summary_

    ### Args
    - `hesse_matrix` (`list[list[sympy.Expr]]`): twice-covariant components of
      the Hesse matrix
    - `vec1` (`list[sympy.Expr]`): first vector along which to project the Hesse
      matrix
    - `vec2` (`list[sympy.Expr]`): second vector along which to project the Hesse
      matrix

    ### Returns
    `sympy.Expr`: _description_
    """
    dim = len(vec1)
    assert(len(vec1) == len(vec2))
    V_proj = 0
    for a in range(dim):
      for b in range(dim):
        V_proj = V_proj + hesse_matrix[a][b]*vec1[a]*vec2[b]
    return powdenest(V_proj.simplify(), force=True)

class CInflatoxPrinter(C99CodePrinter):
  """C99CodePrinter with modified `_print_Symbol` method. Converting Sympy
  expressions with this printer will map all sympy symbols to either:
    - `x[i]` for symbols that must be interpreted as coordinates on the scalar
    manifold
    - `args[i]` for other symbols
  Which symbols should be interpreted as coordinates and which ones should not
  can be specified with the class constructor by passing it the appropriate
  value for the `coordinate_symbols` argument.
  """
  
  def __init__(self, coordinate_symbols: list[sympy.Symbol], settings=None):
    super().__init__(settings)
    coord_dict = {}
    for (i, symbol) in enumerate(coordinate_symbols):
      coord_dict[super()._print_Symbol(symbol)] = f'x[{i}]'
    self.coord_dict = coord_dict
    self.param_dict = {}
    
  def _print_Symbol(self, expr):
    """Modified _print_Symbol function that maps sympy symbols to argument indices"""
    sym_name = self.get_symbol(expr)
    if sym_name is not None:
      return sym_name
    else:
      return self.register_parameter(expr)
    
  def register_parameter(self, symbol: sympy.Symbol) -> str:
    """Adds symbol to the parameter dictionary"""
    sym_name = f"args[{len(self.param_dict)}]"
    self.param_dict[super()._print_Symbol(symbol)] = sym_name
    return sym_name
    
  def get_symbol(self, symbol: sympy.Symbol) -> str | None:
    """Returns string mapping of symbol, if there is one"""
    sym_name = super()._print_Symbol(symbol)
    if self.coord_dict.get(sym_name) is not None:
      return self.coord_dict[sym_name]
    elif self.param_dict.get(sym_name) is not None:
      return self.param_dict[sym_name]
    else:
      return None

class CompilationArtifact:
  def __init__(self, symbol_dictionary: dict, shared_object_path: str):
    pass

class Compiler:
  
  @classmethod
  def _set_compiler(cls, compiler_path: str|None = None, linker_path: str|None = None):
    """Configures the compiler to be the default compiler for the current platform,
    unless an explicit compiler and linker path have been specified.

    ### Args
      compiler_path (str, optional): path to the compiler executable. Defaults to None.
      linker_path (str, optional): path to the linker executable. Defaults to None.
    """
    default_compiler = ccomp.new_compiler()
    if compiler_path is not None:
      default_compiler.set_executables(compiler=compiler_path)
    if linker_path is not None:
      default_compiler.set_executables(linker_exe=linker_path)
    default_compiler
    cls.compiler = default_compiler
    
  @classmethod
  def _compiler_options(cls):
    compiler_type = cls.compiler.compiler_type
    if compiler_type == 'unix':
      return ['-O3','-Wall','-Werror','-fpic'], []
        
  compiler = None

  def _set_preamble(self, model_name: str):
    self.c_code_preamble = f"""//This source file was automatically generated by Inflatox v0.1
// Model: {model_name}, timestamp: {datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}
// Inflatox version: v0.1
// System info: {sys_version}

#include<math.h>
"""
  
  def __init__(self,
    hesse_matrix: HesseMatrix,
    precision: Literal['single', 'double', 'quad'] = 'double',
    output_path: str|None = None,
    cleanup: bool = True
  ):
    self.output_file = open(output_path) if output_path is not None else tempfile.NamedTemporaryFile(
      mode='wt',
      delete=False,
      suffix='.c',
      prefix='inflx_autoc_'
    )
    if Compiler.compiler is None: Compiler._set_compiler()
    self.hesse = hesse_matrix
    self.precision = precision
    self._set_preamble(hesse_matrix.model_name)
    self.cleanup = cleanup
    
  def _generate_c_file(self):
    with self.output_file as out:
      out.write(self.c_code_preamble)
      ty = 'double'
      if self.precision == 'single':
        ty = 'float'
      elif self.precision == 'quad':
        ty = 'long double'

      ccode_writer = CInflatoxPrinter(self.hesse.coordinates)
        
      for a in range(self.hesse.dim):
        for b in range(self.hesse.dim):
          function_body = ccode_writer.doprint(self.hesse.cmp[a][b]).replace(')*', ') *\n    ')
          out.write(f"""
{ty} v{a}{b}(const {ty} x[], const {ty} args[]) {{
  return {function_body};
}}
"""
          )
      #update symbol dictionary
      self.symbol_dict = ccode_writer.coord_dict
      self.symbol_dict.update(ccode_writer.param_dict)
    
  def _c_compile_and_link_inner(self):
    source_path = f'{self.output_file.name}'
    libname = os.path.basename(source_path)[:-2]
    shared_obj_path = self.compiler.library_filename(
      libname, 'shared', output_dir=tempfile.tempdir
    )
    
    #(2) Compile the generated source file
    pre_opts, post_opts = self._compiler_options()
    obj_path = self.compiler.compile(
      [source_path],
      output_dir='/',
      extra_preargs=pre_opts,
      extra_postargs=post_opts
    )
    
    #(3) Link the generated source file
    self.compiler.link_shared_lib(
      obj_path,
      libname,
      output_dir=tempfile.tempdir
    )
    
    #(R) Path to all generated artifacts
    return (source_path, obj_path[0], shared_obj_path)
    
  def compile(self):
    #(1) generate the actual C-source
    self._generate_c_file()
    
    #(2) run compiler and linker
    source_path, obj_path, dylib_path = self._c_compile_and_link_inner()
    
    #(3) cleanup unused artifacts
    if self.cleanup:
      os.remove(source_path)
      os.remove(obj_path)
    
    #(R) return compilation artifact
    return CompilationArtifact(self.symbol_dict, dylib_path)