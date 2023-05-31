use std::path::Path;


type HdylibFn = libloading::Symbol<unsafe extern fn (*const f64, *const f64) -> f64>;
type HdylibStatic = libloading::Symbol<u32>;

struct HesseDylib {
  lib: libloading::Library
}

impl HesseDylib {
  pub fn new(path: &Path) -> Result<Self, std::io::Error> {
    HesseDyLib { lib: unsafe { libloading::Library::new(path)? } }
  }
}