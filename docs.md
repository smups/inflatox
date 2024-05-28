![inflatox_banner](https://raw.githubusercontent.com/smups/inflatox/dev/logos/banner.png)

# Table of contents
1. [Introduction](#introduction)
2. [Installation](#installation)
    - [Dependencies](#dependencies)
    - [Versioning](#versioning)
3. [Architecture Overview](#architecture-overview)
4. [Using `inflatox`](#using-inflatox)
    - [Using `sympy` to set-up a model](#using-sympy-to-set-up-a-model)
    - [Compiling models](#compiling-models)
    - [Evaluating the consistency condition](#evaluating-the-consistency-condition)
5. [Building `inflatox`](#building-inflatox)

# Introduction
The purpose of this document is to get you started using `inflatox`. It is not a comprehensive API
reference. `inflatox`'s API is well-documented using docstrings. You can access the API documentation
by reading the source code or by using a tool like [`pydoc`](https://docs.python.org/3/library/pydoc.html)
to generate HTML from the docstrings.

In this document, we will go over the following points:
1. Installing `inflatox`
2. Architecture overview
3. Using inflatox
    1. `sympy` basics (getting the model set-up)
    2. compiling models
    3. overview of functionality provided by `inflatox`
4. Building inflatox (see `build.md` as well)

The science behind inflatox, some details about its design and example results are discussed in the
accompanying [arXiv:2405.11628](https://arxiv.org/abs/2405.11628) paper.

# Installation
The `inflatox` package consists of two parts:

- The `inflatox` python code
- The `libinflx_rs` rust code

`libinflx_rs` is packaged as a pre-compiled binary together with the python code when you download
`inflatox` via github or [PyPi](https://pypi.org/project/inflatox/). `libinflx_rs` binaries are
avaliable for all major platforms (Linux/MacOS/Windows, x86 and ARM). None of the rust code is
considered user-facing. It may change drastically in between inflatox updates.

`inflatox` may also be installed or upgraded using the `pip` package manager:
```console
pip install inflatox
pip install --upgrade inflatox
```

## Dependencies
Inflatox has various dependencies, both from rust packages and python packages. All rust dependencies
(except `libc`) are statically compiled into the provided binary. Python dependencies should be
installed automatically by `pip` when you install `inflatox`.

It is worth noting that `inflatox` uses typing hints from Python's standard library. **This means that
you need a relatively recent version of python!** In particular, `inflatox` requires python 3.8 or higher,
so you may have to update your python installation to use `inflatox`.

## Versioning
`inflatox` uses the MAJOR.MINOR.PATCH [semantic versioning](https://semver.org/) scheme.

# Architecture overview
`inflatox` provides a numerical implementation of the consistency conditions from 
[arXiv:2405.11628](https://arxiv.org/abs/2405.11628). Besides the consistency condition itself,
inflatox provides on-trajectory estimates for $\varepsilon_H$, $\eta_{\parallel}$, $|\delta|$ and
$\omega$. These quantities are numerically evaluated by `libinflx_rs` (the rust binary) based on
equations for the first and second derivatives of the potential with respect to the fields.

The expressions for the potential and field-space metric are off-course not known by `inflatox`
beforehand. This is where the remainder of the python code comes in. The python part of `inflatox`
is evaluates the necessary derivatives, christoffels and the likes from a user-specified model
consisting of a potential $V(\phi_1,\phi_2)$ and field-space metric $G_{ij}(\phi_1,\phi_2)$ (in the
coordinate basis $\{\partial_{\phi1},\partial_{\phi2}\}$).

After deriving analytical expressions for all the components needed to evaluate the consistency
conditions, `inflatox` generates compilable `C` code from the analytical expressions. The C code is
then compiled by passing it to the (lightweight and portable) `zig-cc` compiler, which is packaged
as a dependency of `inflatox`.

Finally, inflatox will hand the compiled binary off to `libinflx_rs` for numerical evaluation.

In short, a computation using inflatox passes through the following stages:

1. Model specification
2. Analytical (symbolic) calculation
3. Compilation (symbolic → C code → executable binary)
4. Evaluation of consistency conditions

In the following chapter we will go over these steps in practice.

# Using `inflatox`
Each of the steps necessary to evaluate the consistency condition is implemented in a separate
python class. Each step outputs all data required for the next step, usually packaged in a container
class. Here's an overview of which steps correspond to which classes:

1. **Model specification** → `sympy.Expr`
2. **Analytical calculation** → `inflatox.SymbolicCalculation` outputs `inflatox.SymbolicOutput`
3. **Copmilation** → `inflatox.Compiler` outputs `inflatox.CompilationArtifact`
4. **Consistency Conditions** → `inflatox.consistency_conditions.InflationCondition` and
    `inflatox.consistency_conditions.GeneralisedAL`

## Using `sympy` to set-up a model
`sympy` is a python package for symbolic mathematics. Full documentation for `sympy` is available 
on [their website](https://docs.sympy.org/latest/index.html). This section serves as a quick
introduction to the usage of `sympy` in inflatox.

In `sympy`, mathematical expressions are represented by combinations of symbols. It provides
various operations that can be applied to these expressions, such as simplifications, taking
derivatives, finding zeroes, simple linalg operations etc.

To create a `sympy` expression, we have to declare the symbols beforehand using `sympy.symbols`:
```python
import sympy as sp
x, y = sp.symbols("x y")
```
`sympy` symbols represented by python variables can then be combined using python's built-in math
operations to form expressions. Expressions can also be combined:
```python
expr = x + y
expr_2 = x**2 / y
expr_3 = (8 * expr + 3) / expr_2
```
`sympy` provides its own functions for functions not included in standard python:
```python
expr = x * sp.sin(y)
```
> **Note**: only use the functions provided by `sympy` itself. Using `numpy` or `scipy` methods 
**will not work**.

`inflatox` requires three sets of `sympy` expressions for its calculations:
1. A list of fields (literally a python list of which symbols are to be interpreted as fields. 
    All other symbols are automatically interpreted as model parameters)
2. An expression for the potential $V(\phi_1,\phi_2)$
3. A 2D list (list of lists) containing expressions for the components of the field-space metric
    $G_{ij}(\phi_1,\phi_2)$

An example model input would be:
```python
import sympy as sp

#define the model symbols
φ, θ, L, m, φ0 = sp.symbols('φ θ L m φ0')

#define a list of symbols representing fields
fields = [φ, θ]

#define the scalar potential
V = (1/2*m**2*(φ-φ0)**2).nsimplify()

#define the field-space metric as a 2×2 nested list
g = [
    [1, 0],
    [0, L**2 * sp.sinh(φ/L)**2]
]
```
Once these expressions are in place, the model can be passed to the next step: the symbolic calculation.

## Compiling models
Compiling models using inflatox is fairly straightforward. First, we have to set-up the analytical
calculation using the `inflatox.SymbolicCalculation` class, which we provide with our model (
fields, list of expressions for the metric and an expression for the potential):
```python
import inflatox
sym = inflatox.SymbolicCalculation.new_from_list(fields, metric, potential)
```
To execute the calculation, simply call `.execute()` on the `SymbolicCalculation` instance:
```python
sym_out = sym.execute()
```
This provides us with an instance of `SymbolicOutput`. You can check that you got all the symbolic
calculation bits correct by printing the symbolic output instance (`print(sym_out)`). Both the
constructor and execute method have various configuration options, see the API reference for details.

Next, we must run the actual compiler. This proceeds in much the same manner:
```python
comp_out = inflatox.Compiler(sym_out).compile()
```
During compilation, each field and parameter will assigned an index. For example, the compiler may
decide that our $\phi$ field corresponds to the "first" field-space coordinate $x_0$ and that the $L$
parameter corresponds to the "second" parameter $p_1$. This is relevant for the numerical calculation,
since the parameters and field-space points will be supplied as ordered arrays, where the n-th
entry corresponds to the value of the n-th parameter or field. To see which field/parameter was
assigned which index, you can call the `print_sym_lookup_table()` method on the compiler output:
```python
comp_out.print_sym_lookup_table()
```

## Evaluating the consistency condition
After compiling the consistency condition, the compiler output can be passed to the consistency
conditions implemented in `inflatox`. There are two classes of particular interest here:

1. `inflatox.consistency_conditions.InflationCondition` provides access to basic functionality
(calculating the potential and projected Hesse matrix components $V_{vv},V_{vw},V_{ww}$)
2. `inflatox.consistency_conditions.GeneralisedAL` provides a numerical implementation for the
consistency conditions from [arXiv:2405.11628](https://arxiv.org/abs/2405.11628)

All numerical methods make use of `numpy` arrays. For example, if we want to calculate the value of
the potential at a specific point, we may call:
```python
import numpy as np
from inflatox.consistency_conditions import GeneralisedAL
al = GeneralisedAL(comp_out)

p = np.array([0.2, 1e3, 16/7])
x = np.array([0,0])

# Calculate V(0,0) using the parameters [L, m, phi0] = [0.2, 1e3, 16/7]
V = al.calc_V(x, p)
```
Of particular interest is the `complete_analysis()` method on the `GeneralisedAL` class. It outputs
the values of the consistency condition, as well as $\varepsilon_H$, $\eta_{\parallel}$, $|\delta|$ and
$\omega$ at all points in a user-defined field-space range. In short: everything you want to know to
evaluate if your model has candidate trajectories. To use it, simply specify a range of field-space
that you wish to investigate:
```python
extent = (-10, 10, -20, 20)
number = (2000, 4000)

consistency, epsilon_V, epsilon_H, eta_parallel, delta, omega = al.complete_analysis(p, *extent, *number)
```
Note that satisfying either consistency << 1 *or* $\varepsilon_H\ll1$ is insufficient to identify
the slow-roll attractor. One should look for areas where both hold at the same time.

Of course, an important part in any analysis is plotting the final data. One possible way of plotting
`inflatox` results can be found at https://github.com/smups/srrt_models

# Building `inflatox`
TODO