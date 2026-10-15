#!/usr/bin/env bash
#
# check_tests_compile.sh: type-check the Rust test targets with both backends.
#
# The `#[cfg(test)]` modules compile as one unit, so a single test module that
# stops compiling disables every Rust test, and `cargo build` does not notice.
# This script only compiles the tests (no run, no proving).
#
# Usage: scripts/check_tests_compile.sh   (non-zero exit if the tests do not compile)
#
# To run the library tests, see "Testing" in README.md.
set -euo pipefail

cd "$(dirname "$0")/.."

exec cargo check --tests --features halo2-base-backend,noir-backend
