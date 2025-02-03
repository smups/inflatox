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

# System imports
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from sys import version as sys_version

# Sympy imports
import sympy
from sympy.printing.c import C99CodePrinter

# Internal imports
from .symbolic import InflationModel
from .version import __abi_version__, __version__


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

    def __init__(
        self,
        coordinate_symbols: list[sympy.Symbol],
        coordinate_derivative_symbols: list[sympy.Symbol],
        settings=None,
    ):
        super().__init__(settings)
        coord_dict = {}
        for i, symbol in enumerate(coordinate_symbols):
            coord_dict[super()._print_Symbol(symbol)] = f"x[{i}]"
        dotcoord_dict = {}
        for i, symbol in enumerate(coordinate_derivative_symbols):
            dotcoord_dict[super()._print_Symbol(symbol)] = f"xdot[{i}]"
        self.coord_dict = coord_dict
        self.dotcoord_dict = dotcoord_dict
        self.param_dict = {}

    def print_preamble(self, model_name: str):
        """prints preamble in generated c code"""
        return f"""// This source file was automatically generated by Inflatox
// Model: {model_name}, timestamp: {datetime.now().strftime("%Y-%m-%d, %H:%M:%S")}
// Inflatox version: v{__version__}
// System info: {sys_version}

#include <math.h>
#include <stdint.h>
#ifndef M_PI
#define M_E 2.71828182846 
#define M_LOG2E 1.44269504089
#define M_LOG10E 0.4342944819
#define M_LN2 0.69314718056
#define M_LN10 2.30258509299
#define M_PI 3.14159265359
#define M_PI_2 1.57079632679
#define M_PI_4 0.78539816339
#define M_1_PI 0.31830988618
#define M_2_PI 0.63661977236
#define M_2_SQRTPI 1.1283791671
#define M_SQRT2 1.41421356237
#define M_SQRT_1_2 0.70710678118
#endif
"""

    def _print_Symbol(self, expr):
        """Modified _print_Symbol function that maps sympy symbols to argument indices"""
        if expr.is_number:
            return expr.evalf(self._settings["precision"])

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
        if sym_name.startswith("cse"):
            return sym_name
        if self.coord_dict.get(sym_name) is not None:
            return self.coord_dict[sym_name]
        elif self.dotcoord_dict.get(sym_name) is not None:
            return self.dotcoord_dict[sym_name]
        elif self.param_dict.get(sym_name) is not None:
            return self.param_dict[sym_name]
        else:
            return None


class GSLInflatoxPrinter(CInflatoxPrinter):
    """An extended version of the `CInflatoxPrinter` capable of converting some special functions to
    their GSL counterparts. For a list of supported functions, see `Compiler` class documentation."""

    HYPERH = "gsl_sf_hyperg"
    BESSELH = "gsl_sf_bessel"

    def __init__(
        self,
        coordinate_symbols: list[sympy.Symbol],
        coordinate_derivative_symbols: list[sympy.Symbol],
        settings=None,
    ):
        super().__init__(coordinate_symbols, coordinate_derivative_symbols, settings)
        self.required_headers = []

    def print_preamble(self, model_name: str):
        """Print necessary gsl include directives, as well as the err_setup() function, called by
        lib_inflx_rs when the library is loaded."""
        preamble = super().print_preamble(model_name)
        for required_header in self.required_headers:
            preamble += f"#include<gsl/{required_header}.h>"
        preamble += """// Declare external function for gsl error handling
#include<gsl/gsl_errno.h>
void err_setup(gsl_error_handler_t* rust_panic) {
    gsl_set_error_handler(rust_panic);
}
    """
        return preamble

    def update_preamble(self, header):
        """Add header to preamble"""
        if header not in self.required_headers:
            self.required_headers.append(header)

    def _print_hyper(self, expr):
        """printer for hypergeometric functions"""
        self.update_preamble(self.HYPERH)
        ap = expr.args[0]
        bq = expr.args[1]
        x = self.doprint(expr.args[2])

        type = (len(ap), len(bq))
        if type == (2, 0):
            return f"gsl_sf_hyperg_2F0({self.doprint(ap[0])}, {self.doprint(ap[1])}, {x})"
        elif type == (2, 1):
            return f"gsl_sf_hyperg_2F1({self.doprint(ap[0])}, {self.doprint(ap[1])}, {self.doprint(bq[0])}, {x})"
        elif type == (1, 1):
            return f"gsl_sf_hyperg_1F1({self.doprint(ap[0])}, {self.doprint(bq[0])}, {x})"
        elif type == (0, 1):
            return f"gsl_sf_hyperg_0F1({self.doprint(bq[0])}, {x})"
        else:
            raise Exception(
                "Cannot compute hypergeometric functions other than 2F0, 2F1, 1F1 and 0F1"
            )

    def generic_print_bessel(self, expr, namedict):
        """printer for Bessel family of functions"""
        self.update_preamble(self.BESSELH)
        nu = expr.args[0]
        x = self._print_Symbol(expr.args[1])
        if nu.is_integer:
            nu = int(float(self._print_Symbol(nu)))
            try:
                name = namedict[f"{nu}"]
                return f"gsl_sf_bessel_{name}({x})"
            except KeyError:
                return f"gsl_sf_bessel_{namedict['int']}({nu}, {x})"
        try:
            return f"gsl_sf_bessel_{namedict['float']}({self._print_Symbol(nu)}, {x})"
        except KeyError:
            raise KeyError("No non-integer impl found.")

    def _print_besselj(self, expr):
        return self.generic_print_bessel(expr, {"0": "J0", "1": "J1", "int": "Jn", "float": "Jnu"})

    def _print_bessely(self, expr):
        return self.generic_print_bessel(expr, {"0": "Y0", "1": "Y1", "int": "Yn", "float": "Ynu"})

    def _print_besseli(self, expr):
        return self.generic_print_bessel(expr, {"0": "I0", "1": "I1", "int": "In", "float": "Inu"})

    def _print_besselk(self, expr):
        return self.generic_print_bessel(expr, {"0": "K0", "1": "K1", "int": "Kn", "float": "Knu"})

    def _print_jn(self, expr):
        return self.generic_print_bessel(expr, {"0": "j0", "1": "j1", "2": "j2", "int": "jl"})

    def _print_yn(self, expr):
        return self.generic_print_bessel(expr, {"0": "y0", "1": "y1", "2": "y2", "int": "yl"})


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
        auto_cleanup: bool = True,
    ):
        self.symbol_dictionary = symbol_dictionary
        self.shared_object_path = shared_object_path
        self.n_fields = n_fields
        self.n_parameters = n_parameters
        self.auto_cleanup = auto_cleanup

    def __del__(self):
        # Delete compilation artifact
        if self.auto_cleanup:
            os.remove(self.shared_object_path)

    def lookup_symbol(self, symbol: sympy.Symbol) -> str | None:
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
        print("[Symbol Dictionary]")
        for old, new in self.symbol_dictionary.items():
            print(f"{old} -> {new}")


class Compiler:
    """This class wraps the native platform C compiler. It can be used to generate,
    compile and link C code from a `HesseMatrix` instance to produce a `CompilationArtifact`
    which can be used to calculate consistency conditions. This process involves
    creating a symbol dictionary that maps all symbols used in the `HesseMatrix` to
    C-friendly symbols.

    ## Special function support
    By passing `link_gsl=True` to the constructor of this class, the GSL will be linked by the final
    binary. This allows inflatox to map some special functions from `scipy` to their GSL counterparts.
    Currently supported functions are:
    - Bessel functions (besselj, besseli, besselk, bessely, jn and yn)
    - Hypergeometric functions (2F1, 2F0, 1F1 and 2F0)
    If you have the need for more special functions or are experiencing other issues with the gsl
    feature, contact the authors or open an issue on [github](https://github.com/smups/inflatox/issues)
    """

    c_prefix = "inflx_auto_"
    lib_prefix = "libinflx_auto_"

    default_zigcc_flags = [
        "-O3",
        "-Wall",
        "-Werror",
        "-fpic",
        "-lm",
        "-march=native",
        "-shared",
        "-std=c17",
        "-fno-math-errno",
        "-fno-signed-zeros",
    ]
    # The linker option -Wl --no-as-needed is required on Fedora/Redhat based systems. It could
    # very well be the case that this does not work on Debian/Ubuntu based systems.
    gsl_zigcc_flags = ["-lgsl", "-Wl,--no-as-needed", "-lgslcblas"]

    def __init__(
        self,
        model: InflationModel,
        output_path: str | None = None,
        cleanup: bool = True,
        silent: bool = False,
        link_gsl: bool = False,
        cse: bool = False,
        max_cses: int = 1000,
        compiler_flags: list[str] | None = None,
    ):
        """Constructor for a C Compiler (provided by zig-cc), which can be used
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
        - `silent` (bool, optional): if `True`, no console output will be
          generated. Defaults to `False`
        - `link_gsl` (bool, optional): enables experimental gsl (GNU Scientific Library) support for
          output binary. This enables some special functions to be compiled by inflatox. Enabling this
          binary requires the gls library to be installed and available to the linker. Defaults to False.
        - `cse` (bool, optional): if `True`, common subexpressions will be split into variables.
          This enables additional optimizations by the compiler, BUT MAY BREAK CODE! Defaults to False.
        - `max_cses` (int, optional): maximum number of common subexpressions within a single function.
           This limit is set primarily to avoid hitting an infinite loop when translating sympy
           expressions to C functions using common subexpression elimination. Defaults to 1000.
        - `compiler_flags` (list(str)|None, optional): replace default compiler flags with user-supplied
          ones. Make sure to link libmath and libgsl/libgslcblas (if link_gls==True). The default compiler
          flags can be found under `Compiler.default_zigcc_flags` and `Compiler.gsl_zigcc_flags`.
          Defaults to None.
        """
        self.output_file = (
            open(output_path, "w")
            if output_path is not None
            else tempfile.NamedTemporaryFile(
                mode="wt", delete=False, suffix=".c", prefix=Compiler.c_prefix
            )
        )
        self.symbolic_out = model
        self.cleanup = cleanup
        self.silent = silent
        self.gsl = link_gsl
        self.cse = cse
        self.max_cses = max_cses
        self.zigcc_opts = [flag for flag in self.default_zigcc_flags]
        # Add gsl linker flags
        if link_gsl:
            for flag in self.gsl_zigcc_flags:
                self.zigcc_opts.append(flag)
        # Override default flags if user supplied some
        if compiler_flags is not None:
            self.zigcc_opts = compiler_flags

    def _new_cse_generator(self):
        """Returns a generator that yields new symbols for common subexpressions until the
        user-specified maximum is reached.
        """

        def symbol_generator():
            num = 0
            while num <= self.max_cses:
                yield sympy.symbols(f"cse{num}")
                num += 1
            raise Exception("Maximum number of common subexpressions reached!")

        return symbol_generator()

    def _generate_c_function(
        self, fn_signature: str, body: sympy.Expr, printer: CInflatoxPrinter
    ) -> str:
        out = fn_signature + "{\n"
        """Generates a function body from the provided expression"""
        if self.cse:
            cse_list = sympy.cse(body, symbols=self._new_cse_generator(), order="none", list=False)
            if not self.silent:
                print(f"Found {len(cse_list[0])} common subexpressions")
            for cse_symbol, cse_definition in cse_list[0]:
                out += f"    const double {printer.doprint(cse_symbol)} = {printer.doprint(cse_definition)};"
                out += "\n"
            out += f"    return {printer.doprint(cse_list[1])};"
            out += "\n}\n"
        else:
            out += f"    return {printer.doprint(body)};"
            out += "\n}\n"
        return out + "\n"

    def _generate_c_function_for_vector(
        self, fn_signature: str, vector: list[sympy.Expr], printer: CInflatoxPrinter
    ) -> str:
        """Generates a function body for a vector expression"""
        out = fn_signature + "{\n"
        ordered_output_expr = []

        # First print subexpressions (possibly common to *all* components of the vector)
        if self.cse:
            cse_list = sympy.cse(vector, symbols=self._new_cse_generator(), list=True)
            if not self.silent:
                print(f"Found {len(cse_list[0])} common subexpressions")
            for cse_symbol, cse_definition in cse_list[0]:
                out += f"    const double {printer.doprint(cse_symbol)} = {printer.doprint(cse_definition)};"
                out += "\n"
            for output_expr in cse_list[1]:
                ordered_output_expr.append(output_expr)
        else:
            ordered_output_expr = vector

        # Assign each element of the output vector to a component of the vector
        for i, output_cmp in enumerate(ordered_output_expr):
            out += f"    v_out[{i}] = {printer.doprint(output_cmp)};"
            out += "\n"
        out += "    return;\n}\n\n"

        return out

    def _generate_c_function_for_inner_prod(self, printer: CInflatoxPrinter) -> str:
        out = "double inner_prod(const double x[], const double args[], const double v1[], const double v2[]){\n"
        flattened_metric = []
        for i in range(self.symbolic_out.dim):
            for j in range(self.symbolic_out.dim):
                flattened_metric.append(self.symbolic_out.metric[i][j])
        if self.cse:
            cse_list = sympy.cse(flattened_metric, symbols=self._new_cse_generator(), list=True)
            for cse_symbol, cse_definition in cse_list[0]:
                out += f"    const double {printer.doprint(cse_symbol)} = {printer.doprint(cse_definition)};"
                out += "\n"
            flattened_metric = cse_list[1]

        # Write function for outer product
        return_expr = "0.0"
        for i in range(self.symbolic_out.dim):
            n = self.symbolic_out.dim * i
            for j in range(self.symbolic_out.dim):
                n += j
                symbol_str = printer.doprint(flattened_metric[n])
                if symbol_str == "0" or symbol_str == "0.0":
                    continue
                out += f"    const double g{i}{j} = {symbol_str};"
                out += "\n"
                return_expr += f" + (g{i}{j} * v1[{i}] * v2[{j}])"
        out += f"    return {return_expr};"
        out += "\n}\n\n"
        return out

    def _generate_c_file(self):
        """Generates C source file from Hesse matrix specified by the constructor"""
        # Initialise C-code printer
        fields = self.symbolic_out.coordinates
        dotfields = self.symbolic_out.coordinate_tangents
        printer = (
            GSLInflatoxPrinter(fields, dotfields)
            if self.gsl
            else CInflatoxPrinter(fields, dotfields)
        )

        if not self.silent and self.cse:
            print("Converting sympy to C using common subexpression elimination...")
        contents = ""

        # Write potential
        contents += self._generate_c_function(
            "double V(const double x[], const double args[])", self.symbolic_out.potential, printer
        )

        # Use metric to create function for evaluating inner prods.
        contents += self._generate_c_function_for_inner_prod(printer)

        # Write all the components of the Hesse matrix
        for a in range(self.symbolic_out.dim):
            for b in range(self.symbolic_out.dim):
                contents += self._generate_c_function(
                    f"double v{a}{b}(const double x[], const double args[])",
                    self.symbolic_out.hesse_cmp[a][b],
                    printer,
                )

        # Output functions for each of the basis vectors
        for idx in range(self.symbolic_out.dim):
            vector = self.symbolic_out.basis[idx]
            print(vector)
            name = "v" if idx == 0 else f"w{idx}"
            signature = f"void {name}(const double x[], const double args[], double v_out[])"
            contents += self._generate_c_function_for_vector(signature, vector, printer)

        # Write the size of the gradient
        contents += self._generate_c_function(
            "double grad_norm_squared(const double x[], const double args[])",
            self.symbolic_out.gradient_square,
            printer,
        )

        # Write the equations of motion
        for a in range(self.symbolic_out.dim):
            contents += self._generate_c_function(
                f"double eom{a}(const double x[], const double xdot[], const double args[])",
                self.symbolic_out.eom_fields[a],
                printer,
            )

        # Write the equations of motion for the hubble parameter
        contents += self._generate_c_function(
            "double eomh(const double x[], const double xdot[], const double args[])",
            self.symbolic_out.eom_h,
            printer,
        )
        contents += self._generate_c_function(
            "double eomhdot(const double x[], const double xdot[], const double args[])",
            self.symbolic_out.eom_hdot,
            printer,
        )

        # Output to actual file
        with self.output_file as out:
            # Write preamble
            out.write(printer.print_preamble(self.symbolic_out.model_name))

            # Write global constants
            v = __abi_version__.split(".")
            out.write(f"""
//Inflatox version used to generate this file
const uint16_t VERSION[3] = {{{v[0]},{v[1]},{v[2]}}};
//Number of fields (dimensionality of the scalar manifold)
const uint32_t DIM = {self.symbolic_out.dim};
//Number of parameters
const uint32_t N_PARAMETERS = {len(printer.param_dict)};
//Model name
char *const MODEL_NAME = \"{self.symbolic_out.model_name}\";
// Gsl flag
const char USE_GSL = {1 if self.gsl else 0};

""")
            # Write actual file contents
            out.write(contents)

        # Update symbol dictionary
        self.symbol_dict = printer.coord_dict
        self.symbol_dict.update(printer.param_dict)

    def _zigcc_compile_and_link(self):
        source_path = f"{self.output_file.name}"
        source_name = os.path.basename(source_path)[:-2].removeprefix(Compiler.c_prefix)
        lib_name = f"{Compiler.lib_prefix}{source_name}.bin"
        lib_path = f"{tempfile.tempdir}/{lib_name}"

        # Compile source with zig
        zigargs = [
            sys.executable,
            "-m",
            "ziglang",
            "cc",
            "-o",
            lib_path,  # out
            source_path,  # in
            *self.zigcc_opts,  # compiler options
        ]

        process = subprocess.Popen(zigargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = b""

        while process.stdout.readable():
            line = process.stderr.readline()
            if not line:
                break
            out += line
            if not self.silent:
                print(line.decode("utf-8"), end=None)

        exitcode = process.wait()
        return (source_path, lib_path, (out, exitcode))

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
        # (0) Say hello
        if not self.silent:
            print("Compiling model...")

        # (1) generate the actual C-source
        self._generate_c_file()

        # (2) run compiler and linker
        source_path, dylib_path, (output, exitcode) = self._zigcc_compile_and_link()

        # (3) cleanup unused artifacts
        if self.cleanup:
            os.remove(source_path)

        # (4) print output
        if exitcode != 0:
            if self.silent:
                print(output.decode("utf-8"))

            print(f'Problematic source file located at: "{source_path}"')
            raise Exception("Zig compiler error (see previous output)")

        # (R) return compilation artifact
        return CompilationArtifact(
            self.symbol_dict,
            dylib_path,
            self.symbolic_out.dim,
            len(self.symbol_dict) - self.symbolic_out.dim,
            auto_cleanup=self.cleanup,
        )
