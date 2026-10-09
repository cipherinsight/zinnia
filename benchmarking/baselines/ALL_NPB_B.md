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
(automatic k=10); Noir uses the pinned Nargo/BB versions and unpadded
UltraHonk gate counts, `noir-recursive` target. Optimized additional negative
weight/snapshot regression fixtures pass both backends.

Source SHA-256:

```text
kernel.py 1756d464be2511e3ad1b3b06a7c1da870790879bab45b6f1b3b51a3beee54888
main.kernel.literal.nr 7479ed401cb840ad653044ae7e69876b88d0490a1d944a43bcbb54dcc9b459be
main.kernel.nr 22d7e1d467045ed1f1d036b0471497ea145ed83f9bf743af1d622ce6382675d5
target.kernel.literal.rs 8797110ec8591479da5851c391d3d6e1372caccecd51a0d9bd23b0a30f4a241a
target.kernel.rs da6b7678eedb4514b479d7d623adaf04fe78e886fd4dae319a2e628f8f661a18
```

## pythran_ccopy__CCOPY

Not yet implemented/checked.
