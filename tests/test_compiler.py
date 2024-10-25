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

import pytest
import sympy
from inflatox.compiler import *


def test_cinflatox_printer():
    x, y, a, b = sympy.symbols("x y a b")
    printer = CInflatoxPrinter([x, y])
    # test basic symbols
    assert "x[0]" == printer._print_Symbol(x)
    assert "x[1]" == printer._print_Symbol(y)
    assert "args[0]" == printer._print_Symbol(a)
    assert "args[1]" == printer._print_Symbol(b)

    # test basic arithmetic
    assert "pow(x[0], 2) + x[1]" == printer.doprint(x**2 + y)
    assert "x[0]*x[1]" == printer.doprint(x * y)

    # test some math.h funcs
    assert "sqrt(args[0])*x[1]" == printer.doprint(sympy.sqrt(a) * y)
    assert "sin(x[0])" == printer.doprint(sympy.sin(x))


def test_gslinflatox_printer_headers():
    x, y = sympy.symbols("x y")
    printer = GSLInflatoxPrinter([x, y])
    printer.doprint(sympy.besselj(1, x))
    assert printer.BESSELH in printer.required_headers
    printer.doprint(sympy.hyper([], [1], x))
    assert printer.HYPERH in printer.required_headers


def test_gslinflatox_bessel():
    x, y, n = sympy.symbols("x y n")
    printer = GSLInflatoxPrinter([x, y])
    assert "gsl_sf_bessel_J0(x[0])" == printer.doprint(sympy.besselj(0, x))
    assert "gsl_sf_bessel_J1(x[0])" == printer.doprint(sympy.besselj(1, x))
    assert "gsl_sf_bessel_Jn(10, x[0])" == printer.doprint(sympy.besselj(10, x))
    assert "gsl_sf_bessel_Jnu(0.50000000000000000, x[0])" == printer.doprint(
        sympy.besselj(0.5, x)
    )


def test_gslinflatox_hyper():
    x, y = sympy.symbols("x y")
    printer = GSLInflatoxPrinter([x, y])
    assert "gsl_sf_hyperg_2F0(0, 1, x[0])" == printer.doprint(
        sympy.hyper([0, 1], [], x)
    )
    assert "gsl_sf_hyperg_2F1(0, 1, 2, x[0])" == printer.doprint(
        sympy.hyper([0, 1], [2], x)
    )
    assert "gsl_sf_hyperg_1F1(0, 1, x[0])" == printer.doprint(sympy.hyper([0], [1], x))
    assert "gsl_sf_hyperg_0F1(0, x[0])" == printer.doprint(sympy.hyper([], [0], x))
    with pytest.raises(Exception) as exinfo:
        printer.doprint(sympy.hyper([0, 3, 4], [1, 2], x))
    assert "Cannot compute" in str(exinfo.value)
