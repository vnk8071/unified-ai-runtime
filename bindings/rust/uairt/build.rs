// SPDX-License-Identifier: Apache-2.0
fn main() {
    // DEP_UAIRT_LIB_DIR comes from uairt-sys; give this crate's tests an rpath to the library.
    // Windows has no rpath: put the directory on PATH when running (see the README).
    let windows = std::env::var("CARGO_CFG_TARGET_OS").map_or(false, |os| os == "windows");
    if let (Ok(dir), false) = (std::env::var("DEP_UAIRT_LIB_DIR"), windows) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
}
