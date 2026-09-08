#!/usr/bin/env bash
#
# check_tests_compile.sh — guard against the in-crate Rust test suite rotting.
#
# WHY THIS EXISTS
#   The lib `#[test]` targets compile as a single unit: one non-compiling
#   `#[cfg(test)]` module takes down *every* test in the crate, silently
#   (a green `cargo build` says nothing about `cargo test`). This bit us once
#   already — the zcash→PSE halo2 v0.3 migration broke `src/prove/halo2/tests.rs`
#   (missing `type Params`, removed `ParamsIPA::k()`), which in turn masked a
#   never-compiled test module in `src/prove/noir/bound_analysis.rs`. The
#   historical review notes referenced here were removed during cleanup.
#
# WHAT IT DOES
#   Type-checks the test targets (no run, no proving) with the pinned toolchain
#   and the backend features the tests need. Fast (~seconds warm) and meant to
#   run in CI / a pre-commit / pre-push hook so `--tests` can never stop
#   compiling unnoticed again.
#
# USAGE
#   scripts/check_tests_compile.sh
#   (exits non-zero if the test targets fail to compile)
#
# As a git pre-push hook:
#   ln -s ../../scripts/check_tests_compile.sh .git/hooks/pre-push
#
# RUNNING THE SUITE (this script only *compiles* it)
#   cargo +nightly-2025-12-01 test \
#       --features halo2-base-backend,noir-backend -- --test-threads=1
#
#   `--test-threads=1` is currently required for a deterministic result: ten
#   tests share process-global state (z3's global solver context behind a
#   mutex, used by the optim::resolver / contracts_tests SMT cases, and the
#   global halo2 proving-key / IPA-params caches in two prove::halo2 cache
#   tests) and flake when run concurrently. Single-threaded, 17 tests still
#   fail; those are genuine pre-existing gaps committed non-compiling in the
#   bb67c32 checkpoint (incomplete IR-gen fact/witness-bundle publication,
#   where-lowering strategy, fancy-gather bounds, transpose const-fold, and
#   two prove()/verify() validation gaps) — tracked, not #[ignore]'d.
set -euo pipefail

TOOLCHAIN="nightly-2025-12-01"
FEATURES="halo2-base-backend,noir-backend"

# Run from the repo root regardless of where the hook invokes us.
cd "$(dirname "$0")/.."

exec cargo "+${TOOLCHAIN}" check --tests --features "${FEATURES}"
