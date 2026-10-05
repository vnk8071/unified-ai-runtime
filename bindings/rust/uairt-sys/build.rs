// SPDX-License-Identifier: Apache-2.0
use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=UAIRT_LIB_DIR");
    let dir = env::var("UAIRT_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../../build-shared")
        });
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=uairt");
    // Lets this crate's own test binary find the library; dependents add their own rpath.
    if env::var("CARGO_CFG_TARGET_OS").map_or(true, |os| os != "windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    // Exported to dependent crates as DEP_UAIRT_LIB_DIR so their tests can set an rpath.
    println!("cargo:lib_dir={}", dir.display());
}
