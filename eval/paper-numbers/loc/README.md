# Lines of code of the compiler core at the final commit

Commit `2569c9edb8a2bf0b88e9351ca5752f164d2fad24` (committed `src/` via `git archive`), remote host
shuaicpu6 (container zhantong-dev), 2026-10-12 08:15 UTC. Tools: tokei 15.0.0 (`cargo install tokei` on the local machine)
and cloc 2.06 (release perl script), copied to `$RUN/paper-numbers-final/loc/bin/` on the remote
(not kept here). "Code" = lines that are neither blank nor comment-only.

```sh
source $RUN/env.sh; cd $RUN/paper-numbers-final/loc && ./run.sh > run.log 2>&1
```

| Definition | Files | tokei code | cloc code | tokei all lines |
|---|---:|---:|---:|---:|
| D1: all Rust in `src/` | 516 | 149,745 | 149,518 | 171,918 |
| D2: D1 without `src/phase0`, `src/phase1` | 391 | 113,052 | 112,823 | 130,565 |
| D3: D1 without test code | 458 | 111,033 | 110,994 | 128,030 |
| D4: D1 without test code, `phase0`, `phase1` | 353 | **82,944** | 82,903 | 96,498 |

Test code (`strip_tests.py`): the 58 files reached only through `#[cfg(test)] mod …;`
(`out/test-only-files.txt`) and every `#[cfg(test)]` item inside compiled files (119 files),
with its doc comment. Every file in `src/` is reached from `lib.rs`. Check: the stripped tree
passes `cargo check --release --features halo2-base-backend,noir-backend` (`out/cargo-check.log`),
so it is still the whole library. `phase0` and `phase1` hold 10,702 and 17,387 non-test code lines.

No definition gives about 88K. The compiler without tests and without the research modules (D4)
is about 83K; keeping the tests gives 113K (D2). D4 code plus comment lines is 89,197.
D4 by directory (tokei code): prove 16,588; ops 13,696; helpers 11,739; ir_gen 10,605; optim 9,994;
lower 9,809; builder 5,459; other top-level files 5,054.

Raw outputs: `out/D{1..4}-tokei.{txt,json}`, `out/D{1..4}-cloc.txt`, `out/strip_tests.log`, `run.log`.
