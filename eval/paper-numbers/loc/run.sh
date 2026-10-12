#!/usr/bin/env bash
# Lines of code of the compiler core at the final commit, remote host shuaicpu6 (container zhantong-dev).
# Tools: tokei 15.0.0 (built from crates.io on the local machine, copied to bin/) and cloc 2.06
# (perl script from its GitHub release). Counts are "code" lines: not blank, not comment-only.
# Run in $RUN/paper-numbers-final/loc after `source $RUN/env.sh`.
set -euo pipefail
cd "$(dirname "$0")"
W=/home/zhantong/zinnia-eval-run/zinnia-sizefix
test "$(git -C $W rev-parse HEAD)" = 2569c9edb8a2bf0b88e9351ca5752f164d2fad24
test -z "$(git -C $W status --porcelain --untracked-files=no -- src)"
echo "host $(hostname) (shuaicpu6, docker zhantong-dev)  $(date -Is)  commit $(git -C $W rev-parse HEAD)"
bin/tokei --version; perl bin/cloc --version
rm -rf work && mkdir -p work out
git -C $W archive 2569c9edb8a2bf0b88e9351ca5752f164d2fad24 src | tar -x -C work   # committed sources only
# D1: all Rust in src/.  D2: without src/phase0 and src/phase1 (research modules, not on the compile path).
bin/tokei -t Rust work/src > out/D1-tokei.txt
bin/tokei -t Rust work/src -o json > out/D1-tokei.json
bin/tokei -t Rust work/src -e phase0 -e phase1 > out/D2-tokei.txt
bin/tokei -t Rust work/src -e phase0 -e phase1 -o json > out/D2-tokei.json
# D3: without test code (test-only modules and #[cfg(test)] items; strip_tests.py).
# D4: D3 without phase0 and phase1.
$W/.venv/bin/python strip_tests.py work/src work/nontest/src > out/strip_tests.log
bin/tokei -t Rust work/nontest/src > out/D3-tokei.txt
bin/tokei -t Rust work/nontest/src -o json > out/D3-tokei.json
bin/tokei -t Rust work/nontest/src -e phase0 -e phase1 > out/D4-tokei.txt
bin/tokei -t Rust work/nontest/src -e phase0 -e phase1 -o json > out/D4-tokei.json
cp work/nontest/src.test-files.txt out/test-only-files.txt
# Cross-check with cloc.
perl bin/cloc --include-lang=Rust work/src > out/D1-cloc.txt
perl bin/cloc --include-lang=Rust --exclude-dir=phase0,phase1 work/src > out/D2-cloc.txt
perl bin/cloc --include-lang=Rust work/nontest/src > out/D3-cloc.txt
perl bin/cloc --include-lang=Rust --exclude-dir=phase0,phase1 work/nontest/src > out/D4-cloc.txt
# Check that the stripped sources are still the same compiled crate: cargo check of the library
# with the test code removed (it must type-check; tests are not compiled by cargo check anyway).
# RUSTC_BOOTSTRAP=1 as in benchmarking/harness/build.py (poseidon-primitives uses #![feature]).
Z3_DIR="$($W/.venv/bin/python -c 'import z3,pathlib; print(pathlib.Path(z3.__file__).parent / "lib")')"
export LIBRARY_PATH="$Z3_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}"
mkdir -p work/crate && git -C $W archive 2569c9edb8a2bf0b88e9351ca5752f164d2fad24 Cargo.toml Cargo.lock build.rs benchmarking/noir_fpc | tar -x -C work/crate
cp -r work/nontest/src work/crate/src
(cd work/crate && RUSTC_BOOTSTRAP=1 CARGO_TARGET_DIR=$PWD/../../target cargo check --offline --release --features halo2-base-backend,noir-backend -j 8 > ../../out/cargo-check.log 2>&1) \
  && echo "cargo check of stripped sources: ok" || echo "cargo check of stripped sources: FAILED (out/cargo-check.log)"
$W/.venv/bin/python - <<'PY'
import json
for d in "D1 D2 D3 D4".split():
    r = json.load(open(f"out/{d}-tokei.json"))["Rust"]
    emb = sum(c["stats"]["code"] for c in r["children"].get("Markdown", [])) if r.get("children") else 0
    print(f"{d}: files {len(r['reports'])}, lines {r['code']+r['comments']+r['blanks']}, code {r['code']}, comments {r['comments']}, blanks {r['blanks']}")
PY
