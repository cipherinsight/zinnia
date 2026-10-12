# Task D: walkthrough circuit

**X = 5,494; Y = 6,046 halo2-base advice + lookup cells.** The numerical comparison holds, but the draft's scan explanation does not describe this compiler's circuit.

## Definition and per-program counts

The sole program in every count is `walkthrough/kernel`, saved as [kernel.py](kernel.py). `run.py` extracts the body verbatim from the listing in `zinnia-paper/oopsla-2027-r1/fig/tikz/walkthrough.tex`, then adds `def kernel(a: Integer[Array, "5 6"], key: Integer[Array, "6"]):`. The figure contains no signature or type annotations; the signature follows the task's explicit types and the figure's array dimensions. No imports are passed to `ZKCircuit`.

Compiler worktree: `/home/zhantong/Projects/zinnia-project/wt/npb-hw`; HEAD `3ebe0558f25592a0c1f8b6cf9bf7ad09f3ddb1d1`. Source SHA-256: `9b8d8326d9b57f16d1385600473778b2d9c4800725154d191b68a986d514151b`. Default means all ablation switches unset; local means only `ZINNIA_ABLATE_JOINT=1`. Each target is compiled separately with its own backend cost model.

Cells are `halo2_base_cell_stats(compiled._compiled_ir)`'s `advice + lookup`, at its automatically chosen parameters. `advice` already includes both phases and auxiliary advice; `lookup` sums lookup advice over phases (implementation: `src/prove/halo2_base/mod.rs`, `CellStats` construction). Do not add `second_advice` again. Neither circuit uses second-phase advice or challenges. These are used cells, not allocated column capacity; fixed cells and public instances are excluded.

| Program | Selection | Advice, both phases | Lookup, both phases | Cells | k / lookup bits | Second advice (included) | Noir gates | ACIR opcodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| walkthrough/kernel | default | 5,057 | 437 | 5,494 | 8 / 7 | 0 | 3,997 | 776 |
| walkthrough/kernel | local | 5,603 | 443 | 6,046 | 8 / 7 | 0 | 3,974 | 799 |

Default saves 552 halo2-base cells (9.13% of local). The lookup widths coincide, so a common-width comparison gives exactly the same numbers. Alternative fixed-inclusive counting gives 5,511 / 6,063 cells (default/local); this is **not** the requested metric. Raw fixed, instance, polynomial and parameter data are retained in [results.json](results.json).

Noir gates mean `bb gates`'s `functions[0].circuit_size`, using the harness's `ultra_honk` scheme and `noir-recursive` verifier target; they do not mean ACIR opcode counts (the alternative counts are given in the table). The harness's `noir_gates.gate_count` stages a copy of `noir_fpc`, runs `nargo compile`, then `bb gates`. Nargo is `1.0.0-beta.20`, commit `b4236c1957d0c26cb65d82adc9e5447b6ff1d629`; bb is `5.0.0-nightly.20260324`. Exact version outputs are saved. Noir default is 23 gates larger than local: the halo2 improvement must not be claimed for Noir.

## NumPy checks

[fixture.json](fixture.json) supplies both inputs through the existing harness fixture loader. All four compiled variants were run through the **mock interpreter only**, on both inputs, and compared using the harness's public-output decoder, dynamic shape layout and exact integer tolerance. Every mock reported satisfied constraints, and every comparison reported `match`, including shapes. No cryptographic proof, key generation, setup, or timing benchmark was run.

* A uses the figure's key `[20,40,60,10,30,50]` and its pictured nonzero cells, assigned value one. NumPy and every mock produce `[[1,1,0],[0,1,1],[1,0,1]]`, shape `[3,3]`.
* B uses duplicate and negative keys `[2,-1,2,0,-1,3]` and a single pixel `a[4,2] = -7`, all others zero. NumPy and every mock produce `[[-7]]`, shape `[1,1]`. This also exercises stable tie ordering, a boundary row, negative pixel values, and a different dynamic output shape.

The complete inputs, returned values, public vectors and comparison records are saved. Output padding is decoded using the harness layout; it is not mistaken for part of NumPy's result. These two honest-input tests are not a proof of soundness or an adversarial witness test.

## Claims checked against the compiled circuit

The following findings concern **default halo2-base**, the walkthrough's stated target. [SELECTION.md](SELECTION.md) lists every selected group, for every configuration and lowering round; raw traces are in each configuration's `selection.jsonl`. [default_halo2/ir.txt](default_halo2/ir.txt) is a compact statement-ID listing of the final IR, generated directly from `ir.json`.

| Draft claim | Verdict | Trace and IR evidence |
|---|---|---|
| Sort checks a prover-supplied stable order and establishes permutation | Holds | First round, group 0 selects `sort: permutation check`. IR statements 183, 185, 187, 189, 191, 193 are `stable_argsort` hints. The hinted indices are range-checked and used to read keys; statements 242, 244, 246, 248, 250 are `KeyLessIR` comparisons, each asserted immediately afterward. Strict lexicographic order on `(key,index)` forces distinct indices; with all indices in range, the full order is a permutation, including tie-breaking. |
| Scatter-add shares `[p_k = j]` as one-hot vectors for reads | Holds | First round group 1 selects `gather: permutation scatter by hits`; groups 2–6 select `gather: element by a recorded one-hot`. IR 252–287 are the shared equality matrix. The following multiply/add blocks read image columns directly with these tests (e.g. 288–298); there is no intervening computation of rank values `c`. |
| Endpoint reads become scans using running products | Does not hold | Neither lowering round contains a scan group or selected scan construction. The final IR builds compacted row/column index values with selects and sums (673–715 and 740–801); first endpoints are 695 and 771, while last endpoints are reads at runtime `length - 1`, lowered by index bits (first-round groups 7 and 8). |
| Scan sum-to-one checks make `len >= 1` assertions unselected | Does not hold | Row length is statement 715, the sum of row flags; column length is 801, the sum of column flags. IR 802/803 compare `0 < 715` and assert it; 814/815 do the same for 801. Thus both nonempty assertions remain. Scalar assertions need not appear as their own groups in the selection report, so this conclusion uses final IR as well as the absence of scan nodes. |
| Window reuses the scans' one-hot vectors | Does not hold | First-round group 9 selects `gather: window shift`, but no scan encoding exists. IR 826–830 freshly compare row start 695 with each position; 1005–1010 do the same for column start 771. The window uses its own equality selectors and stop-bound comparisons. A window-shift node alone is not evidence of scan sharing. |
| Local selection would choose select-chain scans, unlike default | Does not hold | Local also has no scan groups. It keeps the same sort, scatter-by-hits and window families, but the five image-read groups use `memory: element by index bits` instead of recorded one-hot reads. The measured improvement is consistent with scatter/read sharing, not the claimed change of scan construction. |

Source interpretation is corroborated by `src/lower/sort/emit.rs` (`ElementMemory::key_at`, lexicographic comparison), `src/lower/gather/constructions.rs` (`ScatterByHits`, `ScatterEmitter::shared`), and `src/lower/gather/window.rs` / `encoding.rs` (fresh equalities when an encoding is absent). These files were read, not modified. The figure itself labels its construction graph a draft awaiting compiler support. Its claim that `rows` is never built is also inconsistent with this emitted compacted-index computation.

The final output still has the bounded dynamic layout expected by the figure: capacity for the original array and runtime dimensions, verified on both input shapes. This does not rescue the unsupported scan-selection explanation.

## Reproduction and artifacts

From this task directory:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/zhantong/Projects/zinnia-project/wt/npb-hw/.venv/bin/python run.py > run.log 2>&1
PYTHONDONTWRITEBYTECODE=1 /home/zhantong/Projects/zinnia-project/wt/npb-hw/.venv/bin/python summarize.py
```

`run.py` launches one measurement process at a time under `systemd-run --user --scope -q -p MemoryMax=16G`. It uses the requested venv and source-based `ZKCircuit(...).compile()`, reuses the harness fixture/reference/output code and Noir gate counter, and puts temporary Noir packages under `D/tmp`. Python bytecode writes are disabled. All scripts and outputs are inside D; no worktree or paper files are edited.

Each of `default_halo2`, `local_halo2`, `default_noir`, `local_noir` contains `result.json`, `selection.jsonl`, `ir.json`, `ir.txt`, and `layout.json`; the Noir folders also retain `main.nr`. [results.json](results.json) combines the measurements. [SELECTION.md](SELECTION.md) is the complete per-program/per-group list behind the selection findings. `summarize.py` rebuilds this report and the readable evidence without compiling again.

The historical acceptance account is not used as a source of these numbers: this task measures the exact walkthrough body on the requested final compiler. The body, types, backend, local-selection definition and gate-tool versions above are part of the reproducibility contract.

## Final compiler (2569c9ed), measured on the remote machine (2026-10-12)

Everything above was measured locally on 3ebe0558. This section replaces those numbers for the
paper: the walkthrough (Sec. 6) now cites **5,042 (joint) vs 5,594 (per-operator) halo2-base
cells**, measured on the remote container (host shuaicpu6, container `zhantong-dev`, hostname
`df2cfd4e011a`) with the checkout the final evaluation used,
`/home/zhantong/zinnia-eval-run/zinnia-sizefix` at `2569c9edb8a2bf0b88e9351ca5752f164d2fad24`
(branch `sizefix-cairo`, clean tree, release build recorded in `target/zinnia-harness-build.json`
for the same commit). Files: [final/](final/).

Same program ([final/kernel.py](final/kernel.py), identical to [kernel.py](kernel.py), SHA-256
`9b8d8326…151b`), same fixture, same metric (`halo2_base_cell_stats` advice + lookup, as in
`benchmarking/ablations/common.py`), halo2-base only. `default` = no ablation switch; `local` =
`ZINNIA_ABLATE_JOINT=1`. The script also dumps the IR that enters array lowering
(`ZINNIA_LAYER_INPUT`) and the selection report (`ZINNIA_LAYER_REPORT`), and runs the mock
interpreter on inputs A and B against NumPy (both `match`, constraints satisfied, both modes).

Commands (from this machine):

```sh
cd zinnia-src/eval/paper-numbers/D
tar -C . -cf - final | ssh shuaicpu6-tunnel 'docker exec -i -u zhantong zhantong-dev bash -c \
  "mkdir -p /home/zhantong/zinnia-eval-run/paper-D && tar -C /home/zhantong/zinnia-eval-run/paper-D -xf -"'
ssh shuaicpu6-tunnel 'docker exec -u zhantong zhantong-dev bash -lc "source /home/zhantong/zinnia-eval-run/env.sh; \
  export TMPDIR=/home/zhantong/zinnia-eval-run/paper-D/tmp; mkdir -p \$TMPDIR; \
  cd /home/zhantong/zinnia-eval-run/paper-D/final && ulimit -v 32000000 && \
  /home/zhantong/zinnia-eval-run/zinnia-sizefix/.venv/bin/python run_final.py 2>&1 | tee run.log"'
ssh shuaicpu6-tunnel 'docker exec -u zhantong zhantong-dev bash -c "cd /home/zhantong/zinnia-eval-run/paper-D && tar -cf - final"' | tar -xf -
```

Output ([final/run.log](final/run.log)):

```
default {"commit": "2569c9edb8a2bf0b88e9351ca5752f164d2fad24", "dirty": "", "host": "df2cfd4e011a", "cells": 5042, "ir_stmts": 1263} advice 4682 lookup 360 k 9
local {"commit": "2569c9edb8a2bf0b88e9351ca5752f164d2fad24", "dirty": "", "host": "df2cfd4e011a", "cells": 5594, "ir_stmts": 1161} advice 5228 lookup 366 k 9
```

| Selection | Advice | Lookup | Cells | k / lookup bits | Estimated cost (layer) |
|---|---:|---:|---:|---:|---:|
| joint (default) | 4,682 | 360 | **5,042** | 9 / 8 | 3,034 |
| per-operator (local) | 5,228 | 366 | **5,594** | 9 / 8 | 3,544 |

Joint selection saves 552 cells (9.9% of per-operator).

### What the IR and the selection contain

Array lowering input ([final/default/layer_input.txt](final/default/layer_input.txt)):
`sort` (6 lanes, argsort of key) → `permutation_scatter` (c[p] = arange(6)) → 30 `array_read`s
of a's rows at c_j (no index check) → row/column flags by `bool_cast`/`logical_or` →
**`rows` and `cols` built explicitly** by a network of selects and products (positions of the set
flags, then padding), lengths ℓ_r = Σh, ℓ_c = Σg, with array facts `ElementsWithin [0,4]` /
`[0,5]` on their entries → `assert(0 < ℓ_r)`, `assert_non_neg(ℓ_r-1)`, `assert_u128_lt(ℓ_r-1, 5)`,
`array_read(rows, ℓ_r-1)` (same for cols) → one `window` operator with bounds rows[0],
rows[-1]+1, cols[0], cols[-1]+1 → height/width as `max(0, stop-start)` (lt + select) → output
(30 cells + 2 dims + length: capacity 30).

There is no scan operator: unlike `rows.min()` in DS-1000 case 507 (Fig. 3), whose selection on
the same commit has four `first_set`/`last_set` groups by running products, `rows[0]`/`rows[-1]`
are not rewritten.

Selection ([final/default/selection.jsonl](final/default/selection.jsonl), round 0; round 1 lowers
the sort's key reads x[p_r] by index bits in both modes):

| Group | Operator | Joint | Per-operator |
|---|---|---|---|
| 0 | sort | sort: permutation check (Sort-Hint) | same |
| 1 | permutation scatter | scatter by hits (Scatter-Add) | same |
| 2–6 | reads a[i, c_j] (one group per row) | element by a recorded one-hot (the scatter's tests) | memory: element by index bits (+ index bits of each c_j) |
| 7, 8 | rows[ℓ_r-1], cols[ℓ_c-1] | element by index bits | same |
| 9 | window | window shift (own equality tests and stop comparisons) | same |

Final IR ([final/default/ir.txt](final/default/ir.txt)): stable_argsort hints 183–193, their
index bits and key reads 194–241, KeyLess asserts 242–251; scatter tests [p_k = j] 252–287;
reads as Σ_k [p_k=j]·a_ik 288–617 (c is never computed); flags 618–672; rows built 673–715,
cols 716–801; `assert 0 < ℓ_r` 802/803 and `0 < ℓ_c` 814/815 remain; ℓ_r-1 decomposed into bits
with bound 5 (806–808) that both check the index and drive the multiplexer (809–812), same for
cols with bound 6 (817–824); the separate `assert_non_neg`/`assert_u128_lt` of these indices are
gone; window: equality of r0 with positions 0–4 (826–830), bounded comparisons with the stop
(831–839), likewise for columns (1005–1021); `max(0, ·)` of height and width 1223–1228.
Per-operator IR ([final/local/ir.txt](final/local/ir.txt)) additionally computes c_j = Σ_k
[p_k=j]·k (from statement 288) and reads by the bits of each c_j.

### Claims of the old Sec. 6 against 2569c9ed

| Claim | Verdict |
|---|---|
| p is a permutation (fact 1); line 3 becomes one scatter operator | Holds (`permutation_scatter`; perm(p) is the argsort's definitional fact) |
| c_j ∈ [0,5] (fact 2), reads on line 4 need no index check | Holds |
| h, g flag nonzero rows/columns; ℓ_r = Σ h_i | Holds |
| rows[0], rows[-1] become scan operators; `rows` is never built | Does not hold: rows/cols built by selects; rows[0] is cell 0, rows[-1] a read at ℓ_r-1 |
| ℓ_r ≥ 1 and ℓ_c ≥ 1 remain assertions (symbolic execution) | Holds |
| both "scans" lie in [0,4] | Holds as an array fact on the entries of rows |
| r0 ≤ r1 (fact 3) | Does not hold: no such fact |
| slice needs neither wrap nor clamp | Holds |
| height needs no max(0, ·) | Does not hold: max(0, ·) is computed |
| window output has type ⟨Int; 5,6 \| 30⟩, padding (fact 4) | Holds |
| sort takes the order from the prover and checks it, establishing perm(p) | Holds |
| scatter-add relies on perm(p), its tests are the one-hot vectors of c_j | Holds |
| reads are Σ_k [p_k=j] a_ik, c never computed | Holds (joint only) |
| scans use running products checked to sum to one, so the ℓ ≥ 1 assertions are not selected | Does not hold: no scans, both assertions selected |
| window reuses the scans' one-hot vectors | Does not hold: own equality tests and comparisons |
| per-operator: scans would use a select chain, window would compare, assertions remain | Does not hold: per-operator differs only in the reads on line 4 (index bits instead of the scatter's tests) |
| 5,494 vs 6,046 cells | Superseded: 5,042 vs 5,594 on 2569c9ed (remote) |

Sec. 6 and its figure were rewritten to these findings (2026-10-12). Note for Sec. 7
(implementation.tex, "The minimum of the result of flatnonzero ... never builds the array, as in
Sec. 6"): true of Fig. 3's crop, not of the Sec. 6 variant.
