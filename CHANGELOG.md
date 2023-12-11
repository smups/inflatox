# Inflatox Changelog

### v0.6.0
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