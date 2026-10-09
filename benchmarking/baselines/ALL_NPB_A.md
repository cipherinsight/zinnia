# ALL NPB-A: ResNet basic block

Case: `npbench_deep_learning_resnet_resnet__resnet_basicblock`. Original shapes: input `(4,7,7,16)`,
conv1 `(1,1,16,4)`, conv2 `(3,3,4,4)`, conv3 `(1,1,4,16)`.
All 3,136 output elements are public in row-major order.

| Variant | Halo2 check | Advice cells | Lookup cells | Advice + lookup | Noir check | Noir gates |
|---|---|---:|---:|---:|---|---:|
| optimized | pass | 4,145,575 | 1,028,528 | 5,174,103 | pass | 586,040 |
| literal | pass | 4,773,415 | 1,152,096 | 5,925,511 | pass | 606,620 |

Files (all newly authored; no duplicate/twin baseline reused):

- [target.resnet_basicblock.rs](../datasets/npbench/npbench_deep_learning_resnet_resnet/target.resnet_basicblock.rs)
- [main.resnet_basicblock.nr](../datasets/npbench/npbench_deep_learning_resnet_resnet/main.resnet_basicblock.nr)
- [target.resnet_basicblock.literal.rs](../datasets/npbench/npbench_deep_learning_resnet_resnet/target.resnet_basicblock.literal.rs)
- [main.resnet_basicblock.literal.nr](../datasets/npbench/npbench_deep_learning_resnet_resnet/main.resnet_basicblock.literal.nr)

## Validation and settings

Both variants were built with `HALO2_HW_CASES` restricted to this case,
`CARGO_BUILD_JOBS=3`, `--profile dev-fast`, and separate `target` / `target-literal`
directories. `hw_check.check --no-prove` passed all four circuits: Halo2
MockProver A/B, Noir witness execution A/B, output comparison against the unchanged
NumPy reference, and rejection of a read input moved outside the Q32 domain.
An additional comparison found all four circuits' complete decoded Q32 outputs
exactly equal on each fixture, not merely within NumPy tolerance.
Both static lint runs have zero errors. The sole warning treats the exponent
in the source constant `1e-5` as size-like; epsilon is not shape-derived.

All checks used `memrun.sh`, `MEMRUN_MAX=8G`, and the assigned `lock-c`.
No 8 GiB check failed, so the authorized 16 GiB exception was not needed.
Halo2 used automatic k selection with the same 8-bit lookup width in both
variants; k and full step records are in the JSON evidence below. Counts exclude
fixed cells and instances from the advice+lookup total. Noir counts are unpadded
UltraHonk circuit sizes under `noir-recursive`, nargo 1.0.0-beta.20
(b4236c1957d0c26cb65d82adc9e5447b6ff1d629), bb 5.0.0-nightly.20260324.
No keys, proofs, verification runs, timing benchmarks, scaling, or Zinnia runs
were performed. No compiler IR, emitted circuits, or compiler sizes were read.

Final evidence under `/home/zhantong/Projects/zinnia-project/eval-redo/all-work/npb-a/`:
`opt-final.json`, `literal.json`, `check-opt-final.log`, `check-literal.log`.
Earlier `opt-halo2.json` and `opt-noir.json` are preliminary runs and are superseded;
only final evidence supplies the table. `summarize.py` checks exact fixture output
equality across the four circuits. Temporary packages and logs stay outside git.

## Algorithm and arithmetic audit

- Source `conv2d` loops over output height then width. Each broadcast product
  array is formed in batch/kernel-row/kernel-column/input-channel/output-channel
  order; each reduction scans kernel-row/kernel-column/input-channel in order.
  Every elementwise product is Q32-rounded before summation. This follows the
  conservative unfused interpretation of NPB_FAIRNESS item 3: an explicit
  multiplication followed by `np.sum` is not assumed to be a fused primitive.
  No historical convolution baseline's fused arithmetic was copied.
- `batchnorm2d` reduces axis 0 only, retaining every spatial/channel position.
  `std` uses its own mean, centered elementwise rounded squares, population mean,
  and square root. The normalization denominator is exactly `sqrt(std + eps)`,
  including the second square root. Epsilon is `Q32(1e-5) = 42950`.
- Both variants use floor-rounded products, positive mean division, square roots,
  and normalization division through shared gadgets. Rust's constant Q32 mean
  divisor and Noir's division of the encoded sum by integer N give the same
  floor quotient and domain. Float32 intermediate storage adds no extra rounding,
  as required by the canonical Q32 contract.
- Literal code retains both mean evaluations and recomputes `x - mean` for the
  returned expression. It materializes all intermediate arrays and the complete
  zero-padded tensor, calls each source helper at its source position, and uses
  ordinary comparison/select ReLU and shared arithmetic only. No custom hints,
  forbidden gadgets, or cross-expression range-proof reuse occur.
- Optimized code applies exact CSE to the repeated mean and centered differences.
  Halo2 retains quotient range proofs into ReLU, avoids constructing unused
  selected offsets, and reuses square-root bounds for positive division. Noir
  uses positive division directly because its square root constrains the
  denominator nonnegative and epsilon makes it positive. These changes preserve
  the arithmetic, algorithm, and acceptance domain.
- Every private input leaf is domain-checked; indexing uses compile-time loop
  positions. Three convolutions, three batch normalizations, padding, both
  intermediate ReLUs, residual addition, and final ReLU are retained.

The audit establishes the handwritten variants' shared arithmetic and source
algorithm. It does not certify equivalence to an uninspected compiler lowering;
no full-domain compiler-equivalence or fair aggregate claim is made.

## Tool notes and completion

The existing checker case-set label is `zinnia_rejects`, although this batch
identifies the case as accepted. The explicit case still checks successfully;
no case sets or tools were changed. The worktree virtual environment emits
harmless `sys.prefix`/`sys.exec_prefix` path warnings. There were no blocking
tool failures. All four circuits are complete and passing; no case is unfinished.

Literal halo2 is larger by 751,408; there is no size inversion.

Literal noir is larger by 20,580; there is no size inversion.

## Source SHA-256

| File | SHA-256 |
|---|---|
| `kernel.py` | `8e9e6bd0ab77569b5bc5e205f29f9573157183598b8c208a9c2bee41ac6fd8bf` |
| `target.resnet_basicblock.rs` | `ff995ce0b1af51b856dcfaef18ba6e005b9d858b8fde06d1ff8a70ef23af9dc7` |
| `main.resnet_basicblock.nr` | `0d15239d6862f8c9e5eb07ac57b4a5cce585ed180507ccc26160ffab6030f965` |
| `target.resnet_basicblock.literal.rs` | `8017d450b271f350ce7ee77869eb5cd4f7b17dd5fd38c20cfa8f15a46a8374ef` |
| `main.resnet_basicblock.literal.nr` | `eade38d9a4a2288db0c9b5a6cd5f673bee64f36a66c3ac3910476e2180861c9c` |
