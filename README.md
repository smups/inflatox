![inflatox_banner](https://raw.githubusercontent.com/smups/inflatox/dev/logos/banner.png)
# Inflatox - multifield inflation consistency conditions in python
[![License: EUPL v1.2](https://img.shields.io/badge/License-EUPLv1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
[![arXiv](https://img.shields.io/badge/arXiv-2405.11628-b31b1b.svg)](https://arxiv.org/abs/2405.11628)
[![PyPi](https://img.shields.io/pypi/v/inflatox)](https://pypi.org/project/inflatox)
[![CI](https://github.com/smups/inflatox/actions/workflows/CI.yml/badge.svg)](https://github.com/smups/inflatox/actions/workflows/CI.yml)

Inflatox provides utilities to compute slow-roll parameters and turn-rates for
two-field inflation models, based on the consistency condition from Generalised consistency condition
first presented in Anguelova & Lazaroiu (2023)[^1] and later generalised for the purposes of this package
in Wolters, Iarygina & Achúcarro (2024)[^3] [arXiv:2405.11628](https://arxiv.org/abs/2405.11628).
The consistency conditions can be used in a parameter sweep of a two-field model to find
possible inflation trajectories.

> [!NOTE]
> If this software has proven useful to your research, please consider citing
JCAP07(2024)079[^3]

## Features
- symbolic solver for components of the Hesse matrix of an inflationary model
  with non-canonical kinetic terms, powered by [`sympy`](https://www.sympy.org).
- transpiler to transform `sympy` expressions into executable compiled (`C`) code.
- (experimental) support for special function transpilation using the GSL
- built-in multithreaded `rust` module for high-performance calculations of
  consistency conditions that interfaces directly with `numpy` and python.
- utilities for performing parameter sweeps.
- extendability: inflatox' exposes a python interface to calculate any intermediate
  quantity, which can be used to extend it with additional consistency conditions.
- no need to read, write or compile any `rust` or `C` code manually
  (this is all done automatically behind the scenes).
- no system dependencies, everything needed to run the package can be automatically
  installed by `pip`.

## Installation and Dependencies
> [!IMPORTANT]
> Inflatox requires at least python version `3.8`.

The latest version of inflatox can be installed using pip:
```console
pip install inflatox
```
Inflatox can be updated using:
```console
pip install --upgrade inflatox
```

## Example programme
The following code example shows how `inflatox` can be used to calculate the
potential and components of the Hesse matrix for a two-field hyperinflation model.
```python
#import inflatox
import inflatox
import sympy as sp
import numpy as np
sp.init_printing()

#define model
φ, θ, L, m, φ0 = sp.symbols('φ θ L m φ0')
fields = [φ, θ]

V = (1/2*m**2*(φ-φ0)**2).nsimplify()
g = [
  [1, 0],
  [0, L**2 * sp.sinh(φ/L)**2]
]

#print metric and potential
display(g, V)

#symbolic calculation
calc = inflatox.SymbolicCalculation.new(fields, g, V)
hesse = calc.execute()

#run the compiler
out = inflatox.Compiler(hesse).compile()

#evaluate the compiled potential and Hesse matrix
from inflatox.consistency_conditions import GeneralisedAL
anguelova = GeneralisedAL(out)

p = np.array([1.0, 1.0, 1.0])
x = np.array([2.0, 2.0])
print(anguelova.calc_V(x, p))
print(anguelova.calc_H(x, p))

extent = (-1, 1, -1, 1)
consistency_condition, epsilon_V, epsilon_H, eta_H, delta, omega =
    anguelova.full_analysis(p, *extent)
```

## Special function support
Inflatox features (experimental) support for transpiling special functions from `scipy` to C using
the GSL library. The GSL (GNU Scientific Library) is not packaged together with inflatox due to its
conflicting license. Inflatox merely generates code that calls GSL functions, you must still provide
the headers and compiled shared libraries (`libgsl` and `libgslcblas`) yourself.

If you intend on using this feature, make sure that:
- The GSL is installed and can be found
- GSLCBLAS is installed and can be found
- The GSL headers are installed and can be found

Check out the documentation of the `Compiler` class for a list of currently supported special
functions, and the `docs.md` file for more technical details.
If you are experiencing any issues with the gsl feature (or if a special function you need is missing),
please [open an issue on github](https://github.com/smups/inflatox/issues) or contact the authors.

## Supported Architectures
The combinations of OS and CPU architecture listed down below
have pre-compiled binary distributions of `inflatox` available
via `PiPy`. If your arch is not listed here, you will have to
[compile `inflatox` manually](/BUILD.md).
- Intel/AMD x86/i686 (32 bit)
  - linux/gnu (glibc >= 2.17, kernel >= 3.2)
  - windows 7+ [^2]
- ARM armv7 (32 bit)
  - linux/gnu (glibc >= 2.17, kernel >= 3.2, hard float)
- Intel/AMD x86_64/amd64 (64 bit)
  - linux/gnu (glibc >= 2.17, kernel >= 3.2)
  - windows 7+ [^2]
  - macOS 10.12+ / Sierra+
- ARM aarch64 (64 bit)
  - linux/gnu (glibc >= 2.17, kernel >= 4.1)
  - macOS 11.0+ / Big Sur+
> [!NOTE]
> Apple silicon M-series chips are supported (aarch64)*

## License
[![License: EUPL v1.2](https://img.shields.io/badge/License-EUPLv1.2-blue.svg)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
>[!NOTE]
> Inflatox is explicitly not licensed under the dual
> Apache/MIT license common to the Rust ecosystem. Instead it is licensed under
> the terms of the [European Union Public License v1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12).

Inflatox is a science project and embraces the values of open science and free
and open software. Closed and paid scientific software suites hinder the
development of new technologies and research methods, as well as diverting much-
needed public funds away from researchers to large publishing and software
companies.

See the [LICENSE.md](../LICENSE.md) file for the EUPL text in all 22 official
languages of the EU, and [LICENSE-EN.txt](../LICENSE-EN.txt) for a plain text
English version of the license.

## References and footnotes
[^1]: Anguelova, L., & Lazaroiu, C. (2023). Dynamical consistency conditions for
  rapid-turn inflation. *Journal of Cosmology and Astroparticle Physics*,
  May 2023(20). https://doi.org/10.1088/1475-7516/2023/ 05/020
[^2]: Windows 7 is no longer considered a tier-1 target by the rust project. Usage
  of Windows 10+ is recommended.
[^3]: Wolters, R, Iarygina & O. Achúcarro, A (2024). Generalised conditions for
  rapid-turn inflation. *Journal of Cosmology and Astroparticle Physics*, July 2024(79).
  https://doi.org/10.1088/1475-7516/2024/07/079
