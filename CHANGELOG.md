# Inflatox Changelog

### v0.8.2 printing bugfix
General
- fixed bug with printing sympy expressions outside IPython environments
- added assertion error messages

### v0.8.1 Synchronise
Version 0.8.1 is identical to version 0.8.0. It has come to our attention that a duplicate 0.8.0
published in March 2024 (rather than the May 2024 _real_ 0.8.0) was unintentionally published to
PyPi. Version 0.8.1 should remedy this issue.

## v0.8.0 (💣BREAKING CHANGES💣)
General
- Bumped the minimum required python version from 3.7 to 3.8. This was already the case (due to the
  usage of `std`'s `typing` lib), but not reflected in the package manifest.
- Improved and updated documentation where necessary
- Added `docs.md` documentation file
- Added `builds.md` wiht build instructions

Mathematical changes
- Moved away from the $\kappa=3$ paradigm to a more sensible calculation for 
  $\varepsilon_H$:
  $$
    \varepsilon_H=3 \times \frac{\varepsilon_V - 1/2 (V_t/V)^2}
    {\varepsilon_V+V_{tt}/V - 1/2 (V_t/V)^2}
  $$
  Where $V_{tt}$ is calculated using the assumption $\tan\delta=V_{vw}/V_{vv}$ and $1/2(V_t/V)^2=
  \varepsilon_V\sin^2\delta$. Note that this new equation gives different results: it is less
  restrictive than the previous $\kappa=3$ paradigm, although it is better motivated from a 
  theoretical standpoint.
- Replaced $\varepsilon_H$ calculation with $\eta_{\parallel}$ calculation, where $\eta_{\parallel}$
  is calculated as:
  $$
    \eta_{\parallel}=\omega\tan\delta -3
  $$

API changes
- Renamed `AnguelovaLazaroiuCondition` to `GeneralisedAL`
- `SymbolicCalculation` no longer requires the vector $w \perp \nabla V$ to be speficied by default
  (it is still possible to do so as an option)

Upgrades
- Updated Rayon 1.8 -> 1.10
- Upgraded PyO3 0.20 -> 0.21
- Upgraded numpy 0.20 -> 0.21

## v0.7.0 - major refactor (💣BREAKING CHANGES💣)
Additions
- Added functionality to calculate the potential slow-roll parameter $\varepsilon_V$
- Added functionality to calculate $\varepsilon_H$ assuming the AL condition holds.
- Added functionality to calculate $\eta_{\parallel}$ and $\eta_H$ assuming
  the AL condition holds.
- Added new `complete_analysis()` method that is able to calculate six dynamical
  quantities at once. This cuts computation time by about a factor six.
- Added `consistency_only()` method for situations when `full_analysis()` is
  overkill (no real speed-up).
- Added `consistency_only_old()` method for computing the original AL condition
  only (not included in `complete_analysis`)
- Added `epsilon_v_only()` method for computing the first potential slow-roll
  parameter.
- Reworked console output

Mathematical changes
- All consistency conditions are now returned in the following format:
  $$
    \rm{out}=\left|\frac{|\rm{rhs}|-|\rm{lhs}|}{|\rm{rhs}|+|\rm{lhs}|}\right|
  $$
  where $\rm{lhs}$ is the left-hand side of the consistency condition and $\rm{rhs}$
  the right-hand side.
- The characteristic angle $\delta$ is now returned in the range $[0,\pi/2]$
  since the sign of the angle between $w^a$ and $t^a$ is arbitrary since the 
  given that the choice of $w^a$ is arbitrary.

Deletions
- Removed `evaluate()` method of `AnguelovaLazaroiuCondition`. Functionality
  replaced by `consistency_only()`.
- Removed `calc_delta()` method of `AnguelovaLazaroiuCondition`. Functionality
  replaced by `complete_analysis()`.
- Removed `calc_epsilon()` method of `AnguelovaLazaroiuCondition`. Functionality
  replaced by `complete_analysis()` and `epsilon_v_only()`.
- Removed `calc_omega()` method of `AnguelovaLazaroiuCondition`. Functionality
  replaced by `complete_analysis()`.

Upgrades
- Upgraded rayon 1.7.0 -> 1.8.0

## v0.6.0 - More quantities: $\varepsilon_H$ and $\omega$
- Specified that package is only compatible with python 3.7 - 3.11, because
  no version of `Numba` dependency (which is a dependency of EinsteinPy) that is
  compatible with python 3.12 has been released yet. Package still interfaces
  with rust using the stable python 3.7 ABI. This will not be changed until
  the 3.7 ABI is deprecated. 
- Added `hesse_array` method that allows calculating the hesse matrix (for an
  arbitrary number of field-space dimensions) at all points in a given field-
  space array.
- added functionality to calculate the turn rate $\omega$ under the assumption
  that the slow-roll parameters are small.
- added functionality to calculate the hesse matrix for a whole range of field
  space values at once.
- Upgraded numpy 0.19 -> 0.20
- Upgraded PyO3 0.19 -> 0.20

## v0.5.0 - Quantum diffusion
- breaking ABI change (new symbols)
- added functionality to calculate if gradient of potential flips sign (goes to
  zero). This is relevant for those looking for areas where quantum diffusion
  dominates over the background
- improved error messages
- moved examples to different repository

### v0.4.1
- simplified build system
- inflatox now uses pythons ABI3, specifically version 3.7
- added (optional) progress bars

## v0.4.0 - $\delta$ calculations
- added functionality to calculate the $\delta$ slow-roll, rapid-turn parameter (see _future publication_)
- yanked the inflatox crate from `crates.io`. All rust code is now considered internal

## v0.3.0 - Higher order conditions
- added higher-order consistency conditions

### v0.2.1
- Bugfixes

## v0.2.0 - Zig
- replaced platform native C compiler with pip-bundled `zig-cc` compiler

## v0.1.0 - Initial Release