# Inflatox build instructions
Synopsis of build system:
- `rustc` and `Cargo` are required to build `libinfltx_rs`, the rust crate backing the 
  numerical implementation of the consistency condition(s).
- `maturin` is used to package the compiled binaries and python source code into a
  python wheel.