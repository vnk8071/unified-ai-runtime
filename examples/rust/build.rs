// SPDX-License-Identifier: Apache-2.0
// Links libuairt and libuairt_llm from the directory in UAIRT_LIB_DIR (a build of this repository with -DBUILD_SHARED_LIBS=ON).
use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=UAIRT_LIB_DIR");
    let dir = env::var("UAIRT_LIB_DIR").expect(
        "set UAIRT_LIB_DIR to the build directory that holds libuairt and libuairt_llm (cmake -DBUILD_SHARED_LIBS=ON)",
    );
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=uairt_llm");
    println!("cargo:rustc-link-lib=dylib=uairt");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
}
