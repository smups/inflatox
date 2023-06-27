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

#System imports
import os
import tempfile
import textwrap
import distutils.ccompiler as ccomp
from datetime import datetime
from sys import version as sys_version

#Sympy imports
import sympy
from sympy.printing.c import C99CodePrinter

#Internal imports
from .symbolic import HesseMatrix
from .version import __version__

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
    """Returns string representing sympy symbol"""
    sym_name = super()._print_Symbol(symbol)
    if self.coord_dict.get(sym_name) is not None:
      return self.coord_dict[sym_name]
    elif self.param_dict.get(sym_name) is not None:
      return self.param_dict[sym_name]
    else:
      return None

class CompilationArtifact:
  """Class representing the output of the `Compiler`. It contains all information
  necessary to access the compiled artifact.
  
  ### Compiler symbols
  The `Compiler` class maps all sympy symbols found in the expressions for the
  potential and projected Hesse matrix to arguments of two numpy arrays:
    - `x` for the scalar fields themselves.
    - `args` for all other symbols (model parameters).
  All functions and classes that make use of this `CompilationArtifact` class will
  most likely require the user to supply `x` and `args` as numpy arrays. Therefore,
  one must know which sympy symbols were mapped to which position in the `x` and
  `args` arrays. The `CompilationArtifact` class provides two methods for this:
  `symbol_lookup` and `print_sym_table`. See their documentation for more details.
  """
  
  symbol_printer = C99CodePrinter()
  
  def __init__(
    self,
    symbol_dictionary: dict,
    shared_object_path: str,
    n_fields: int,
    n_parameters: int,
    auto_cleanup: bool = True
  ):
    self.symbol_dictionary = symbol_dictionary
    self.shared_object_path = shared_object_path
    self.n_fields = n_fields
    self.n_parameters = n_parameters
    self.auto_cleanup = auto_cleanup
    
  def __del__(self):
    #Delete compilation artifact
    if self.auto_cleanup: os.remove(self.shared_object_path)

  def lookup_symbol(self, symbol: sympy.Symbol) -> str|None:
    """returns the compiled symbol (string) for the supplied sympy symbol,
    if the sympy symbol is known, `None` otherwise. See class docs for more
    info on compiled symbols.

    ### Args
    `symbol` (`sympy.Symbol`): sympy symbol to be converted.

    ### Returns
    `str|None`: name of the supplied symbol (either `args[n]` or `x[n]`), or
    `None` if the symbol is unknown.
    """
    sym_name = self.symbol_printer._print_Symbol(symbol)
    if not isinstance(sym_name, str):
      return None
    else:
      return self.symbol_dictionary[sym_name]
    
  def print_sym_lookup_table(self):
    """prints full mapping of all known sympy symbols and their corresponding
    compiled symbols. See class docs for more info on compiled symbols.
    """
    print('[Symbol Dictionary]')
    for (old, new) in self.symbol_dictionary.items():
      print(f'{old} -> {new}')

class Compiler:
  """This class wraps the native platform C compiler. It can be used to generate,
  compile and link C code from a `HesseMatrix` instance to produce a `CompilationArtifact`
  which can be used to calculate consistency conditions. This process involves
  creating a symbol dictionary that maps all symbols used in the `HesseMatrix` to
  C-friendly symbols.
  """
  
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
    """Sets the appropriate compiler flags based on the current platform"""
    compiler_type = cls.compiler.compiler_type
    if compiler_type == 'unix':
      return ['-O3','-Wall','-Werror','-fpic', '-lm', '-march=native'], ['-O3', '-fpic', '-lm', '-march=native']
    else:
      raise NotImplementedError(f'Inflatox currently does not support {compiler_type}.')
    #TODO: add Windows support
        
  compiler = None
  
  def __init__(self,
    hesse_matrix: HesseMatrix,
    output_path: str|None = None,
    cleanup: bool = True
  ):
    """Constructor for a C Compiler (provided by the platform), which can be used
    to convert the provided `HesseMatrix` object into a platform- and arch-specific
    shared library object.
    
    ### Examples
    To compile a previously calculated Hesse matrix, we simply construct a
    `Compiler` instance and call `.compile()` on it:
    ```python
    artifact = inflatox.Compiler(hesse_matrix).compile()
    ```
    See the docs for `HesseMatrix` and `SymbolicCalculation` for info on how to
    obtain a `HesseMatrix` instance.

    ### Args
    - `hesse_matrix` (HesseMatrix): HesseMatrix object that will be turned into
      C code and compiled.
    - `output_path` (str | None, optional): output path of compilation artifacts.
      Will auto-select the platform-defined temporary folder if option is set to
      `None`. Defaults to `None`.
    - `cleanup` (bool, optional): if `True`, generated artifacts will be deleted
      when they are no longer necessary. Defaults to True.
    """
    self.output_file = open(output_path) if output_path is not None else tempfile.NamedTemporaryFile(
      mode='wt',
      delete=False,
      suffix='.c',
      prefix='inflx_autoc_'
    )
    if Compiler.compiler is None: Compiler._set_compiler()
    self.hesse = hesse_matrix
    self._set_preamble(hesse_matrix.model_name)
    self.cleanup = cleanup
  
  def _set_preamble(self, model_name: str):
    """prints preamble in generated c code"""
    self.c_code_preamble = f"""//This source file was automatically generated by Inflatox v0.1
// Model: {model_name}, timestamp: {datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}
// Inflatox version: v{__version__}
// System info: {sys_version}

#include<math.h>
#include<stdint.h>
"""
    
  def _generate_c_file(self):
    """Generates C source file from Hesse matrix specified by the constructor"""
    #(1) Initialise C-code printer
    ccode_writer = CInflatoxPrinter(self.hesse.coordinates)
    
    with self.output_file as out:
      #(2) Write preamble
      out.write(self.c_code_preamble)
      
      #(3) Write potential
      potential_body = textwrap.fill(
        ccode_writer.doprint(self.hesse.potential).replace(', ', ','),
        width=80,
        tabsize=4,
        break_long_words=False,
        break_on_hyphens=False
      ).replace('\n', '\n    ')
      out.write(f"""
double V(const double x[], const double args[]) {{
  return {potential_body};
}}
"""
      )
      
      #(4) Write all the components of the Hesse matrix
      for a in range(self.hesse.dim):
        for b in range(self.hesse.dim):
          function_body = ccode_writer.doprint(self.hesse.cmp[a][b]).replace(')*', ') *\n    ')
          out.write(f"""
double v{a}{b}(const double x[], const double args[]) {{
  return {function_body};
}}
"""
          )
          
      #(5) Write global constants
      v = __version__.split('.')
      out.write(f"""
//Inflatox version used to generate this file
const uint16_t VERSION[3] = {{{v[0]},{v[1]},{v[2]}}};
//Number of fields (dimensionality of the scalar manifold)
const uint32_t DIM = {self.hesse.dim};
//Number of parameters
const uint32_t N_PARAMTERS = {len(ccode_writer.param_dict)};
""")
      
    #(6) Update symbol dictionary
    self.symbol_dict = ccode_writer.coord_dict
    self.symbol_dict.update(ccode_writer.param_dict)
    
  def _c_compile_and_link_inner(self):
    """Compiles and links the generated C source as a platform-specific shared lib"""
    source_path = f'{self.output_file.name}'
    libname = os.path.basename(source_path)[:-2]
    shared_obj_path = self.compiler.library_filename(
      libname, 'shared', output_dir=tempfile.tempdir
    )
    
    #(2) Compile the generated source file
    compiler_args, linker_args = self._compiler_options()
    obj_path = self.compiler.compile(
      [source_path],
      output_dir='/',
      extra_preargs=compiler_args
    )
    
    #(3) Link the generated source filewill
    self.compiler.link_shared_lib(
      obj_path,
      libname,
      output_dir=tempfile.tempdir,
      extra_preargs=linker_args
    )
    
    #(R) Path to all generated artifacts
    return (source_path, obj_path[0], shared_obj_path)
    
  def compile(self) -> CompilationArtifact:
    """Compiles the Hesse matrix specified in the constructor of this class into
    a shared library that can be used in further calculations. This process involves
    creating a symbol dictionary that maps all symbols used in the `HesseMatrix`
    to C-friendly symbols. The output of this function contains this dictionary,
    see the docs of `CompilationArtifact` for more info.
    
    ### Examples
    To compile a previously calculated Hesse matrix, we simply construct a
    `Compiler` instance and call `.compile()` on it:
    ```python
    artifact = inflatox.Compiler(hesse_matrix).compile()
    ```
    See the docs for `HesseMatrix` and `SymbolicCaluclation` for info on how to
    obtain a `HesseMatrix` instance.

    ### Returns
      `CompilationArtifact`: artifact that can be used in further calculations.
      It contains info about the model and inflatox version used to create the
      artifact.
    """
    #(1) generate the actual C-source
    self._generate_c_file()
    
    #(2) run compiler and linker
    source_path, obj_path, dylib_path = self._c_compile_and_link_inner()
    
    #(3) cleanup unused artifacts
    if self.cleanup:
      os.remove(source_path)
      os.remove(obj_path)
    
    #(R) return compilation artifact
    return CompilationArtifact(
      self.symbol_dict,
      dylib_path,
      self.hesse.dim,
      len(self.symbol_dict) - self.hesse.dim,
      auto_cleanup=self.cleanup
    )
