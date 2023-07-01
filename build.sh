TARGETS="aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu i686-unknown-linux-gnu x86_64-pc-windows-msvc i686-pc-windows-msvc x86_64-apple-darwin aarch64-apple-darwin" 

for t in $TARGETS
do
  rustup target add "$t"
  maturin build --release --target "$t" --zig
done