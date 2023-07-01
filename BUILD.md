![inflatox_banner](https://raw.githubusercontent.com/smups/inflatox/dev/logos/banner.png)
# Build instructions for Inflatox
Inflatox is built using the maturin package manager for hybrid rust/python projects. To build inflatox, the following requirements must be met:
- maturin must be installed (version above 1.0)
- cargo and rustc must be installed (version above 1.70)
- the zig compiler must be installed (any version with a functioning c compiler and linker)
- the rust standard library must be installed for each desired target
- gnulibc version 2.17 or higher (for linux distributions)

With maturin and the zig compiler installed, packaging inflatox for linux is as easy as running
```bash
maturin build --release --target x86_64-unknown-linux-gnu --zig --manylinux 2014
maturin build --release --target aarch64-unknown-linux-gnu --zig --manylinux 2014
```

## Generate Github Continuous Integration (CI) actions
run
```bash
maturin generate-ci github > .github/workflows/CI.yml
```