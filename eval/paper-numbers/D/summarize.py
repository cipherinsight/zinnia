#!/usr/bin/env python3
"""Rebuild human-readable evidence and report from run.py's saved measurements."""
import json
from pathlib import Path
from collections import Counter
D=Path(__file__).resolve().parent
r=json.loads((D/'results.json').read_text())
for tag in r:
    statements=json.loads((D/tag/'ir.json').read_text())
    (D/tag/'ir.txt').write_text(''.join(f"{s['stmt_id']}: {s['ir_instance']['__class__']} {s['arguments']} {s['ir_instance']['ir_data']}\n" for s in statements))
lines=['# Selection evidence\n','Every group in every lowering round is listed below. Group IDs restart in each round. `selection.jsonl` also lists every selected helper node and its estimated cost; these estimates are not measured cell counts.\n']
for tag in r:
    lines += [f'## {tag}\n','| Round | Group | Family | Lanes | Chosen |','|---|---|---|---|---|']
    for i,line in enumerate((D/tag/'selection.jsonl').read_text().splitlines(),1):
        rd=json.loads(line)
        for g in rd['groups']:
            lines.append(f"| {i} | {g['group']} | {g['family']} | {g['lanes']} | {', '.join(g['chosen'])} |")
    lines.append('')
(D/'SELECTION.md').write_text('\n'.join(lines))
x,y=r['default_halo2'],r['local_halo2']
rows=[]
for mode in ['default','local']:
    h=r[mode+'_halo2']; s=h['stats']; n=r[mode+'_noir']
    rows.append(f"| walkthrough/kernel | {mode} | {s['advice']:,} | {s['lookup']:,} | {h['cells']:,} | {s['k']} / {s['lookup_bits']} | {s['second_advice']} | {n['gates']:,} | {n['acir_opcodes']:,} |")
report=f'''# Task D: walkthrough circuit

**X = {x['cells']:,}; Y = {y['cells']:,} halo2-base advice + lookup cells.** The numerical comparison holds, but the draft's scan explanation does not describe this compiler's circuit.

## Definition and per-program counts

The sole program in every count is `walkthrough/kernel`, saved as [kernel.py](kernel.py). `run.py` extracts the body verbatim from the listing in `zinnia-paper/oopsla-2027-r1/fig/tikz/walkthrough.tex`, then adds `def kernel(a: Integer[Array, "5 6"], key: Integer[Array, "6"]):`. The figure contains no signature or type annotations; the signature follows the task's explicit types and the figure's array dimensions. No imports are passed to `ZKCircuit`.

Compiler worktree: `/home/zhantong/Projects/zinnia-project/wt/npb-hw`; HEAD `{x['commit']}`. Source SHA-256: `{x['source_sha256']}`. Default means all ablation switches unset; local means only `ZINNIA_ABLATE_JOINT=1`. Each target is compiled separately with its own backend cost model.

Cells are `halo2_base_cell_stats(compiled._compiled_ir)`'s `advice + lookup`, at its automatically chosen parameters. `advice` already includes both phases and auxiliary advice; `lookup` sums lookup advice over phases (implementation: `src/prove/halo2_base/mod.rs`, `CellStats` construction). Do not add `second_advice` again. Neither circuit uses second-phase advice or challenges. These are used cells, not allocated column capacity; fixed cells and public instances are excluded.

| Program | Selection | Advice, both phases | Lookup, both phases | Cells | k / lookup bits | Second advice (included) | Noir gates | ACIR opcodes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Default saves {y['cells']-x['cells']:,} halo2-base cells ({100*(y['cells']-x['cells'])/y['cells']:.2f}% of local). The lookup widths coincide, so a common-width comparison gives exactly the same numbers. Alternative fixed-inclusive counting gives {x['cells']+x['stats']['fixed']:,} / {y['cells']+y['stats']['fixed']:,} cells (default/local); this is **not** the requested metric. Raw fixed, instance, polynomial and parameter data are retained in [results.json](results.json).

Noir gates mean `bb gates`'s `functions[0].circuit_size`, using the harness's `ultra_honk` scheme and `noir-recursive` verifier target; they do not mean ACIR opcode counts (the alternative counts are given in the table). The harness's `noir_gates.gate_count` stages a copy of `noir_fpc`, runs `nargo compile`, then `bb gates`. Nargo is `1.0.0-beta.20`, commit `b4236c1957d0c26cb65d82adc9e5447b6ff1d629`; bb is `5.0.0-nightly.20260324`. Exact version outputs are saved. Noir default is {r['default_noir']['gates']-r['local_noir']['gates']} gates larger than local: the halo2 improvement must not be claimed for Noir.

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
'''
(D/'REPORT.md').write_text(report)
