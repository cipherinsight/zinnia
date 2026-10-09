# Batch npb-b

Original fixture shapes; optimized and literal handwritten Halo2/Noir circuits.
No Zinnia compilation, IR, emitted code, or circuit sizes were consulted. No
key generation, proving, proof verification, timing experiment, or scaling run.

Validation uses `hw_check.check --no-prove`, the worktree virtual environment,
`MEMRUN_MAX=8G`, three Cargo jobs, `--profile dev-fast`, and the assigned
`all-work/lock-a`. Evidence is in
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/npb-b/`.
The checker includes A/B output comparisons, Halo2 MockProver, Noir witness
execution, counts, and rejection of a selected out-of-domain input. This does
not establish proof verification or equivalence over the entire input domain.

## npbench_polybench_floyd_warshall_floyd_warshall__kernel

Directory: `benchmarking/datasets/npbench/npbench_polybench_floyd_warshall_floyd_warshall/`.

| File | Check | Advice | Lookup | Advice + lookup | Noir gates |
|---|---|---:|---:|---:|---:|
| `target.kernel.rs` | pass | 359168 | 71680 | 430848 | - |
| `main.kernel.nr` | pass | - | - | - | 76546 |
| `target.kernel.literal.rs` | pass | 432896 | 104448 | 537344 | - |
| `main.kernel.literal.nr` | pass | - | - | - | 83630 |

Algorithm audit: 16 iterations in source order. Each outer sum uses a snapshot
of the previous matrix; no sequential in-place Floyd-Warshall substitution,
symmetry shortcut, or diagonal/sign assumption. Literal circuits materialize
all 256 outer sums, then all 256 elementwise minima, then replace the matrix.
Optimized circuits reuse signed-window proofs across minima and iterations.
All 256 mutated `path` cells are public integers, row-major.

Arithmetic audit/interpretation: plain exact integer additions, signed64
comparisons, no Q32 multiplication or rounding. Each compared sum must fit
signed64 in both variants, following the shared comparison/window contract;
NumPy's wrapping int64 overflow is not emulated. Ties retain the old entry
(the values coincide). No private hints outside shared gadgets.

Reuse: no twin or earlier implementation exists for this entry; plumbing and
shared gadget APIs follow existing baselines. Literal computation was derived
from `kernel.py`, independently of optimized computation.

Lint: both variants pass. Optimized Noir's two `0 as u64` warnings are safe
casts of compile-time zero, not truncating checks of private inputs.

Literal is larger on both backends. Halo2 uses lookup width 8 in both variants
(automatic k=10 optimized, k=11 literal); Noir uses the pinned Nargo/BB
versions and unpadded UltraHonk gate counts, `noir-recursive` target. Additional
negative-weight/snapshot regression fixtures pass both backends and both variants (`floyd-{opt,lit}-edges-both.json`).

Source SHA-256:

```text
kernel.py 1756d464be2511e3ad1b3b06a7c1da870790879bab45b6f1b3b51a3beee54888
main.kernel.literal.nr 7479ed401cb840ad653044ae7e69876b88d0490a1d944a43bcbb54dcc9b459be
main.kernel.nr 22d7e1d467045ed1f1d036b0471497ea145ed83f9bf743af1d622ce6382675d5
target.kernel.literal.rs 8797110ec8591479da5851c391d3d6e1372caccecd51a0d9bd23b0a30f4a241a
target.kernel.rs da6b7678eedb4514b479d7d623adaf04fe78e886fd4dae319a2e628f8f661a18
```

## pythran_ccopy__CCOPY

Directory: `benchmarking/datasets/pythran/pythran_ccopy/`.

| File | Check | Advice | Lookup | Advice + lookup | Noir gates |
|---|---|---:|---:|---:|---:|
| `target.CCOPY.rs` | pass | 424734 | 18620 | 443354 | - |
| `main.CCOPY.nr` | pass | - | - | - | 82318 |
| `target.CCOPY.literal.rs` | pass | 1035292 | 73965 | 1109257 | - |
| `main.CCOPY.literal.nr` | pass | - | - | - | 120714 |

Both Rust variants compile; both variants pass lint. Halo2 lookup width is 8
for both; automatic k is 10 optimized, 11 literal. Evidence: `ccopy-opt.json`
(Halo2), `ccopy-opt-noir.json` (final Noir), `ccopy-lit.json` (both literal).
Literal is larger on both backends. Optimized Noir initially used native
private writes; replacing those with conditional writes over the short
array reduced 179150 gates to 82318 without changing the source algorithm.

Algorithm audit: retain the early return, unit-stride branch, general-stride
branch, source negative-stride start formulas, sequential assignments and
index increments. Each private `range(N)` has the source precondition bound
1024, not the fixture's N=2 or the array length 4. Zero strides remain valid,
including N=1024, and repeated destination writes preserve last-write wins.
Inactive accesses use zero before indexing. Outputs are all four final CY
complex values, flattened as real/imaginary pairs (eight Q32 cells).

Arithmetic audit: N and strides are private signed64 integers. The source
precondition 0<=N<=1024 is constrained. Copy arithmetic is exact on Q32 cells,
with no multiplication/rescaling of the complex values. Index arithmetic is
exact field arithmetic and cannot wrap the field under these bounds; only
active array accesses impose bounds. In particular, an unobserved final index
increment or inactive start expression may leave signed64 without invalidating
an otherwise valid copy. Both variants have the same acceptance domain.

Interpretations: complex indexing is one source read of two scalar components;
normalizing a destination and selecting each cell is the literal indexed-write
operation, not an additional source gather. Literal Halo2 uses q.index for
both component reads and a fresh bound/wrap for the write index. Literal Noir
uses shared index_pos for the read/write and scans every destination. Initial
branch values that only reach guarded operations can be computed in the field
without extra inactive-branch range guards. Static components and constant
indices are wiring. Optimized unit-stride accesses constant-fold the loop
index, asserting inactivity for out-of-range steps; they do not replace or
shorten the private loop. General-stride copies remain sequential. Optimized
Halo2 shares complex-read one-hots and input bounds; optimized Noir uses pick
for the short input and checked destination normalization followed by
conditional writes over the short destination.

Reuse: no twin or existing CCOPY baseline; only house-style plumbing and shared
gadget APIs. No custom hints. L1-L7 reviewed against every kernel statement.

Additional regression checks: 13 scenarios on each backend and variant,
52 passing checks total. Accepted scenarios: N=0 with extreme strides; full
unit-stride copy; reversing both arrays; negative source stride; N=1024 with
both strides zero; zero destination stride with last-write wins; zero source
stride; N=1 with signed64-extreme strides. Rejected scenarios: unit-stride
out-of-bounds; source out-of-bounds; negative destination stride out-of-bounds;
N=1025; N=-1. Accepted outputs match the NumPy reference. Invalid Halo2
witnesses fail MockProver; invalid Noir executions fail constraints. Evidence:
`ccopy-lit-edges-both.json`, `ccopy-opt-edges-halo2.json`, and
`ccopy-opt-edges-noir.json`. The runner and temporary fixtures are outside the
repository in the evidence directory; original fixtures were not modified.

Source SHA-256:

```text
kernel.py 166632b57be582ff922e910c29df68fd5f243239117d58c45035bac4ecaa64b0
main.CCOPY.literal.nr a0ddca42d75a3c75d5a5acb4c22816f06930ea655be1fb649fcd7402d7daf54b
main.CCOPY.nr 7db6a10dd92f8fdd5f5bacb8e9daad4ad36afb39ea4a5ea0fa2dbaf156f9b883
target.CCOPY.literal.rs 30586a438b2496c1a714c24b69a51090f6b646b8780131e8e406502c5e20635e
target.CCOPY.rs d6489992b61bae7648817325f17bb63fdb0f622f40e57cc58da61a2f7031f6c6
```

## Toolchain and scope

Nargo/noirc 1.0.0-beta.20, commit b4236c1957d0c26cb65d82adc9e5447b6ff1d629;
BB 5.0.0-nightly.20260324; UltraHonk `noir-recursive`, unpadded gate counts.
The inherited case set labels both cases `zinnia_rejects`; the authorized checker accepts explicitly named non-perf entries.
No case-set files were changed by this batch. Compiler-side algorithm and
arithmetic equivalence were not checked; this report audits these handwritten
sources and the shared gadget contract only, not a new fair aggregate.

No blocking tool defects. The virtual environment emits benign sys.prefix /
sys.exec_prefix warnings when invoked through the prescribed relative path.
The shared validation lock caused waits; all heavy jobs retained the 8 GiB cap.

Completed: both cases, all eight circuits. Unfinished: none. No literal
circuit is equal to or smaller than its optimized counterpart.
