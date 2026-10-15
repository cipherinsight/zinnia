//! Build script: Python extension link arguments and the location of `libz3`.
//!
//! The `z3` crate links `libz3` without building it (see `Cargo.toml`). It
//! finds a system copy through pkg-config, or the directory named by
//! `Z3_LIBRARY_PATH_OVERRIDE`. When neither is set, this script looks for the
//! copy shipped in the `z3-solver` wheel of the Python interpreter PyO3 builds
//! against (`PYO3_PYTHON`, else the active virtual environment, else `python3`),
//! and adds its directory to the linker search path and to the rpath, so the
//! extension module and the Rust test binaries load it without
//! `LD_LIBRARY_PATH`.

use std::path::PathBuf;
use std::process::Command;

/// Prints the directory of the wheel's `libz3`, or nothing if `z3` is not
/// installed or ships no shared library of its own.
const FIND_WHEEL_LIBZ3: &str = r#"
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("z3")
if spec is not None and spec.submodule_search_locations:
    ext = {"win32": "dll", "cygwin": "dll", "darwin": "dylib"}.get(sys.platform, "so")
    lib_dir = pathlib.Path(spec.submodule_search_locations[0]) / "lib"
    if (lib_dir / f"libz3.{ext}").is_file():
        print(lib_dir.resolve())
"#;

fn main() {
    pyo3_build_config::add_extension_module_link_args();

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=Z3_LIBRARY_PATH_OVERRIDE");
    if std::env::var_os("Z3_LIBRARY_PATH_OVERRIDE").is_some() {
        return;
    }
    if let Some(lib_dir) = wheel_libz3_dir() {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        if std::env::var("CARGO_CFG_TARGET_FAMILY").as_deref() == Ok("unix") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        }
    }
}

fn wheel_libz3_dir() -> Option<PathBuf> {
    let python = pyo3_build_config::get().executable.as_ref()?;
    let output = Command::new(python)
        .args(["-c", FIND_WHEEL_LIBZ3])
        .output()
        .ok()?;
    let dir = String::from_utf8(output.stdout).ok()?;
    let dir = dir.trim();
    (output.status.success() && !dir.is_empty()).then(|| PathBuf::from(dir))
}
