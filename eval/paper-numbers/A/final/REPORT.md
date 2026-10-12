# Task A — accepted programs and performance set

Generated from compiler evidence by `report.py`; the unit is a **(corpus, program, entry function)**, not a distinct source file. The supplied acceptance account selects only `outcome == "accepted_correct"`. Its 214 entries comprise 132 DS-1000, 32 NPBench, and 50 Pythran entries. All 214 compile successfully with final compiler `2569c9edb8a2bf0b88e9351ca5752f164d2fad24` (remote host shuaicpu6, container zhantong-dev).

## Results

| Definition | DS-1000 | NPBench | Pythran | Total |
|---|---:|---:|---:|---:|
| Runtime-shaped array (Q1, primary) | 33 | 0 | 3 | 36 |
| Any DynamicNDArray type (Q1 alternative) | 33 | 0 | 3 | 36 |
| Runtime shape OR runtime-position insert (Q1 example-inclusive alternative) | 36 | 0 | 3 | 39 |
| Input-dependent array selection/movement (Q2, primary) | 54 | 3 | 10 | 67 |
| Q2 limited to lowering input, omitting pre-expanded compaction | 46 | 3 | 9 | 58 |
| Q2 explicit runtime addresses only (strict alternative) | 23 | 1 | 5 | 29 |
| Q2 excluding programs counted only through Sort placeholders | 43 | 1 | 8 | 52 |

## Definitions and compiler evidence

**Q1.** A program qualifies if compilation encounters `Value::DynamicNDArray` whose `runtime_length`, `runtime_rank`, or at least one `runtime_shape` scalar has `static_val == None`. The bound is `envelope.total_bound`. This includes intermediate, discarded, and returned arrays, and views such as input-dependent reshape. It does not require the final output to have runtime shape. A constant length alone does not imply constant shape (e.g. reshape `(nrow, -1)`). This is the compiler's static-versus-runtime classification, not a claim that we proved two distinct feasible shapes with an SMT solver.

`instrument.py` copies the compiler under `A/compiler/` and adds read-only logging at IR-generator expression results (recursively through composites) and `DynamicNDArrayData::max_length`, which also observes internal temporary buffers. It records metadata without minting compiler IDs or emitting instructions. See `instrumentation.diff`, `compiler/src/types/composite.rs`, and `records/*/dynamic.jsonl`. All programs with this type also have nonconstant shape metadata, so the broader type-only count is identical.

**Fixed-size insert is not runtime-shaped.** Inserting one element at a runtime position into a fixed-size array changes element placement, but always increases length by one. The user's example therefore spans two definitions. The example-inclusive alternative additionally counts `case0363::kernel`, `case0364::kernel`, and `case0365::kernel`, all DS-1000; their lowering has an input-dependent `Insert` position. They are included in Q2, not Q1's primary count.

**Q2.** The primary definition counts input-dependent selection or placement of array elements as recognized by the compiler: nonconstant input-dependent addresses in `ArrayReadIR`/`ArrayWriteIR` (or ROM/RAM counterparts), a runtime read of a sorted output, input-dependent `Window` bounds, `Insert` positions, `PermutationScatter` destinations, `MaskWrite` predicates, sort/argsort permutations over input-dependent values, and boolean-mask compaction. These are program-level unions, not operator counts.

For the structured IR entering `ArrayLowering`, `summarize.py` propagates input dependence along statement operands, starting at `ReadInteger`, `ReadFloat`, `ReadHash`, external-result and witness reads. Static indices occupy the `ArrayAccess.offset` field; only input-dependent runtime axis operands count. Static unrolled loop indices and static shape arithmetic do not qualify. Sort selection is detected from input-dependent shared values; `Window`, `Insert`, and scatter are tested only on their position/mask operands. Reductions returning an index (such as argmax) alone are not counted unless that index is subsequently used for selection.

Boolean-mask gathers can be expanded into a scalar one-hot shift network **before** the lowering input. `compaction.py` logs actual calls to `compact_by_mask` with a nonconstant mask, and checks their final IR against the stock compiler. This adds nine entries to the lowering-input-only count: `case0412::kernel`, `case0413::kernel`, `case0442::kernel`, `case0443::kernel`, `case0444::kernel`, `case0462::kernel`, `case0463::kernel`, `case0481::kernel`, `pythran_tax::tax`. All such compactions produce dynamic arrays, so the revisit covers every dynamic-type program.

**Q2 ambiguity.** The strict alternative counts only explicit runtime addresses and runtime Sort-output reads; it excludes mask predicates and compiler operators whose placement is implicit. The primary definition is an operational compiler count: the frontend also implements some `min`/`max` reductions with `Sort` placeholders, so programs such as NPBench MLP and softmax qualify even though the source need not contain a sorting call or explicit indexed load. The final alternative removes programs whose only evidence is such a Sort placeholder. These alternatives, with their exact membership, prevent conflating compiler lowering behavior with source-level indexing syntax. A direct array access simplified away before lowering is outside the lowering-input definitions; scalar conditional selections alone are not treated as indexed loads.

**Validation.** For all 214 entries, stock and instrumented compilations have identical final exported-IR SHA-256 hashes, output layouts, and byte-for-byte identical pre-lowering text dumps. The instrumented extension disables optional proving backends when built; compilation still targets the default `halo2` lowering, and equality checks establish that this does not change the measured IR. No proof, key, timing benchmark, or mock execution was requested or run. Internal compile-time bookkeeping in the existing compiler is not reported.

## Q1 per-program lists

### DS-1000 (33)

`case0298::kernel`, `case0301::kernel`, `case0302::kernel`, `case0303::kernel`, `case0304::kernel`, `case0322::kernel`, `case0353::kernel`, `case0383::kernel`, `case0389::kernel`, `case0391::kernel`, `case0392::kernel`, `case0393::kernel`, `case0412::kernel`, `case0413::kernel`, `case0414::kernel`, `case0415::kernel`, `case0416::kernel`, `case0417::kernel`, `case0418::kernel`, `case0419::kernel`, `case0442::kernel`, `case0443::kernel`, `case0444::kernel`, `case0462::kernel`, `case0463::kernel`, `case0467::kernel`, `case0468::kernel`, `case0481::kernel`, `case0490::kernel`, `case0507::kernel`, `case0508::kernel`, `case0509::kernel`, `case0510::kernel`

### NPBench (0)

None.

### Pythran (3)

`pythran_periodic_dist::dist`, `pythran_projection_simplex::projection_simplex`, `pythran_tax::tax`

## Q2 per-program lists (primary definition)

### DS-1000 (54)

`case0298::kernel`, `case0300::kernel`, `case0314::kernel`, `case0318::kernel`, `case0319::kernel`, `case0322::kernel`, `case0335::kernel`, `case0345::kernel`, `case0346::kernel`, `case0347::kernel`, `case0353::kernel`, `case0355::kernel`, `case0356::kernel`, `case0357::kernel`, `case0358::kernel`, `case0363::kernel`, `case0364::kernel`, `case0365::kernel`, `case0374::kernel`, `case0375::kernel`, `case0381::kernel`, `case0382::kernel`, `case0383::kernel`, `case0389::kernel`, `case0391::kernel`, `case0392::kernel`, `case0393::kernel`, `case0412::kernel`, `case0413::kernel`, `case0419::kernel`, `case0433::kernel`, `case0434::kernel`, `case0436::kernel`, `case0437::kernel`, `case0442::kernel`, `case0443::kernel`, `case0444::kernel`, `case0450::kernel`, `case0454::kernel`, `case0462::kernel`, `case0463::kernel`, `case0467::kernel`, `case0468::kernel`, `case0481::kernel`, `case0490::kernel`, `case0501::kernel`, `case0502::kernel`, `case0503::kernel`, `case0504::kernel`, `case0505::kernel`, `case0507::kernel`, `case0508::kernel`, `case0509::kernel`, `case0510::kernel`

### NPBench (3)

`npbench_deep_learning_mlp_mlp::mlp`, `npbench_deep_learning_softmax_softmax::softmax`, `npbench_scattering_self_energies_scattering_self_energies::scattering_self_energies`

### Pythran (10)

`pythran_ccopy::CCOPY`, `pythran_insertion_sort::insertion_sort`, `pythran_lapl3d::laplacien`, `pythran_lu::factorMatrix0`, `pythran_lu::factorMatrix1`, `pythran_make_decision::md`, `pythran_periodic_dist::dist`, `pythran_projection_simplex::projection_simplex`, `pythran_tax::tax`, `pythran_value_index::tri`

## Performance set

82 `status == "perf"` entries; accepted NPBench/Pythran entries not in it: 0; perf entries not accepted: 0.

## Reproduce and audit

```sh
# remote (shuaicpu6, container zhantong-dev)
cd /home/zhantong/zinnia-eval-run/paper-numbers-final/A
./reproduce.sh
```

The script uses the requested worktree `.venv/bin/python`, reuses `benchmarking.harness.kernels.compile_kernel` (including decorated-source extraction and lifted helpers), and compiles the acceptance-account kernel path and entry function. Inputs are not fixture-specialized; the same annotations/source used by the acceptance harness determine the compilation. No fixture values are needed for this compiler-only characterization. All new files, copied compiler source, build outputs, and logs stay in `A/`; Python bytecode writes are disabled. The container has no systemd; each compilation worker runs under `ulimit -v` (32 GiB virtual memory), with 16 workers per stage.

For a fast report-only rebuild from saved evidence:

```sh
/home/zhantong/zinnia-eval-run/zinnia-sizefix/.venv/bin/python summarize.py
/home/zhantong/zinnia-eval-run/zinnia-sizefix/.venv/bin/python report.py
```

- `classification.json`: all 214 rows, flags, dynamic metadata, representative IR statements and per-family counts.
- `records/<dataset>__<program>__<function>/`: original account row, kernel hash, stock/instrumented final IR hash and output layout, raw lowering text, structured lowering IR (`layer.json.gz`), dynamic metadata, and compaction evidence/checks.
- `performance_difference.json`: full case-set records for all three exclusions.
- `provenance.json`: input hashes and compiler revision.
- `phase0.json`: auxiliary shape analysis only. This separate analysis rejects 12 otherwise compilable programs, so it is **not** used in the reported classifications. The actual IR-generation instrumentation covers all 214.

## Complete membership table

`S`: Q1 primary; `S+I`: shape-or-insert alternative; `I`: Q2 primary; `L`: lowering-input-only alternative; `A`: strict explicit-address alternative; `NS`: exclude Sort-only evidence. “—” means false. This table lists all accepted entries, including negatives; the dynamic-type alternative has exactly the `S` membership.

| Corpus | Program::function | S | S+I | I | L | A | NS | Q2 compiler evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DS-1000 | `case0291::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0298::kernel` | yes | yes | yes | yes | yes | yes | ArrayWriteIR |
| DS-1000 | `case0300::kernel` | — | — | yes | yes | yes | yes | Sort, SortRuntimeRead |
| DS-1000 | `case0301::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0302::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0303::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0304::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0309::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0310::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0311::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0312::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0313::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0314::kernel` | — | — | yes | yes | yes | yes | Sort, ArrayWriteIR |
| DS-1000 | `case0317::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0318::kernel` | — | — | yes | yes | yes | yes | ArrayWriteIR, ArrayReadIR |
| DS-1000 | `case0319::kernel` | — | — | yes | yes | yes | yes | ArrayWriteIR, ArrayReadIR |
| DS-1000 | `case0320::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0321::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0322::kernel` | yes | yes | yes | yes | — | yes | Sort, MaskCompaction |
| DS-1000 | `case0323::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0324::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0325::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0326::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0329::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0330::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0334::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0335::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0336::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0337::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0338::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0339::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0340::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0341::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0342::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0343::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0345::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0346::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0347::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0349::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0353::kernel` | yes | yes | yes | yes | — | yes | Sort, MaskCompaction |
| DS-1000 | `case0355::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0356::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0357::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0358::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0359::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0360::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0361::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0363::kernel` | — | yes | yes | yes | — | yes | Insert |
| DS-1000 | `case0364::kernel` | — | yes | yes | yes | — | yes | Insert |
| DS-1000 | `case0365::kernel` | — | yes | yes | yes | — | yes | Insert |
| DS-1000 | `case0368::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0369::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0370::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0373::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0374::kernel` | — | — | yes | yes | yes | yes | Sort, SortRuntimeRead |
| DS-1000 | `case0375::kernel` | — | — | yes | yes | yes | yes | Sort, SortRuntimeRead |
| DS-1000 | `case0381::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0382::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0383::kernel` | yes | yes | yes | yes | — | — | Sort |
| DS-1000 | `case0385::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0386::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0387::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0389::kernel` | yes | yes | yes | yes | yes | yes | ArrayWriteIR |
| DS-1000 | `case0391::kernel` | yes | yes | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0392::kernel` | yes | yes | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0393::kernel` | yes | yes | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0398::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0399::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0400::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0401::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0402::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0403::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0409::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0412::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0413::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0414::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0415::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0416::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0417::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0418::kernel` | yes | yes | — | — | — | — | — |
| DS-1000 | `case0419::kernel` | yes | yes | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0420::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0422::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0428::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0429::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0430::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0431::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0433::kernel` | — | — | yes | yes | yes | yes | ArrayWriteIR |
| DS-1000 | `case0434::kernel` | — | — | yes | yes | yes | yes | ArrayWriteIR |
| DS-1000 | `case0435::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0436::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0437::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0438::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0439::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0440::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0441::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0442::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0443::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0444::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0448::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0449::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0450::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0451::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0452::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0453::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0454::kernel` | — | — | yes | yes | — | — | Sort |
| DS-1000 | `case0456::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0457::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0458::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0462::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0463::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0467::kernel` | yes | yes | yes | yes | — | yes | Window |
| DS-1000 | `case0468::kernel` | yes | yes | yes | yes | — | yes | Window |
| DS-1000 | `case0471::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0472::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0477::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0478::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0480::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0481::kernel` | yes | yes | yes | — | — | yes | MaskCompaction |
| DS-1000 | `case0485::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0490::kernel` | yes | yes | yes | yes | — | yes | MaskWrite, MaskCompaction |
| DS-1000 | `case0491::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0500::kernel` | — | — | — | — | — | — | — |
| DS-1000 | `case0501::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0502::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0503::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0504::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0505::kernel` | — | — | yes | yes | yes | yes | ArrayReadIR |
| DS-1000 | `case0507::kernel` | yes | yes | yes | yes | — | yes | Window, MaskCompaction |
| DS-1000 | `case0508::kernel` | yes | yes | yes | yes | — | yes | Window, MaskCompaction |
| DS-1000 | `case0509::kernel` | yes | yes | yes | yes | — | yes | Window, MaskCompaction |
| DS-1000 | `case0510::kernel` | yes | yes | yes | yes | — | yes | Window, MaskCompaction |
| NPBench | `npbench_compute_compute::compute` | — | — | — | — | — | — | — |
| NPBench | `npbench_crc16_crc16::crc16` | — | — | — | — | — | — | — |
| NPBench | `npbench_deep_learning_conv2d_bias_conv2d::conv2d_bias` | — | — | — | — | — | — | — |
| NPBench | `npbench_deep_learning_mlp_mlp::mlp` | — | — | yes | yes | — | — | Sort |
| NPBench | `npbench_deep_learning_resnet_resnet::resnet_basicblock` | — | — | — | — | — | — | — |
| NPBench | `npbench_deep_learning_softmax_softmax::softmax` | — | — | yes | yes | — | — | Sort |
| NPBench | `npbench_go_fast_go_fast::go_fast` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_atax_atax::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_bicg_bicg::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_cholesky_cholesky::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_covariance2_covariance2::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_deriche_deriche::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_durbin_durbin::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_floyd_warshall_floyd_warshall::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_gemm_gemm::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_gemver_gemver::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_gesummv_gesummv::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_gramschmidt_gramschmidt::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_k2mm_k2mm::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_k3mm_k3mm::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_lu_lu::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_ludcmp_ludcmp::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_mvt_mvt::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_symm_symm::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_syr2k_syr2k::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_syrk_syrk::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_trisolv_trisolv::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_polybench_trmm_trmm::kernel` | — | — | — | — | — | — | — |
| NPBench | `npbench_pythran_arc_distance_arc_distance::arc_distance` | — | — | — | — | — | — | — |
| NPBench | `npbench_scattering_self_energies_scattering_self_energies::scattering_self_energies` | — | — | yes | yes | yes | yes | ArrayReadIR |
| NPBench | `npbench_weather_stencils_hdiff_hdiff::hdiff` | — | — | — | — | — | — | — |
| NPBench | `npbench_weather_stencils_vadv_vadv::vadv` | — | — | — | — | — | — | — |
| Pythran | `pythran_allpairs::sqr_dists` | — | — | — | — | — | — | — |
| Pythran | `pythran_allpairs::sqr_dists_loops` | — | — | — | — | — | — | — |
| Pythran | `pythran_arc_distance::arc_distance` | — | — | — | — | — | — | — |
| Pythran | `pythran_bubble_sort::bubble_sort` | — | — | — | — | — | — | — |
| Pythran | `pythran_ccopy::CCOPY` | — | — | yes | yes | yes | yes | ArrayReadIR, ArrayWriteIR |
| Pythran | `pythran_clip::clip` | — | — | — | — | — | — | — |
| Pythran | `pythran_create_grid::create_grid` | — | — | — | — | — | — | — |
| Pythran | `pythran_cronbach::cronbach` | — | — | — | — | — | — | — |
| Pythran | `pythran_crotg::CROTG` | — | — | — | — | — | — | — |
| Pythran | `pythran_d2q9_nxnyns::one_time_step` | — | — | — | — | — | — | — |
| Pythran | `pythran_deuxd_convolution::conv` | — | — | — | — | — | — | — |
| Pythran | `pythran_euclidean_distance_square::euclidean_distance_square` | — | — | — | — | — | — | — |
| Pythran | `pythran_fbcorr::fbcorr` | — | — | — | — | — | — | — |
| Pythran | `pythran_fibo_seq::fibo` | — | — | — | — | — | — | — |
| Pythran | `pythran_goodexpomeasure::goodExpoMeasure` | — | — | — | — | — | — | — |
| Pythran | `pythran_harris::harris` | — | — | — | — | — | — | — |
| Pythran | `pythran_hasting::fweb` | — | — | — | — | — | — | — |
| Pythran | `pythran_insertion_sort::insertion_sort` | — | — | yes | yes | yes | yes | ArrayWriteIR |
| Pythran | `pythran_ising::ising_step` | — | — | — | — | — | — | — |
| Pythran | `pythran_l2norm::l2_norm` | — | — | — | — | — | — | — |
| Pythran | `pythran_lap::laplacian` | — | — | — | — | — | — | — |
| Pythran | `pythran_lapl3d::laplacien` | — | — | yes | yes | — | — | Sort |
| Pythran | `pythran_ln::ln` | — | — | — | — | — | — | — |
| Pythran | `pythran_log_likelihood::log_likelihood` | — | — | — | — | — | — | — |
| Pythran | `pythran_lu::factorMatrix0` | — | — | yes | yes | yes | yes | ArrayReadIR, ArrayWriteIR |
| Pythran | `pythran_lu::factorMatrix1` | — | — | yes | yes | yes | yes | ArrayReadIR, ArrayWriteIR |
| Pythran | `pythran_make_decision::md` | — | — | yes | yes | yes | yes | ArrayReadIR |
| Pythran | `pythran_matmul::matrix_multiply` | — | — | — | — | — | — | — |
| Pythran | `pythran_multi_export::a` | — | — | — | — | — | — | — |
| Pythran | `pythran_multitype::check` | — | — | — | — | — | — | — |
| Pythran | `pythran_multitype::check2` | — | — | — | — | — | — | — |
| Pythran | `pythran_multitype::times` | — | — | — | — | — | — | — |
| Pythran | `pythran_pairwise::pairwise` | — | — | — | — | — | — | — |
| Pythran | `pythran_periodic_bc::periodic_bc` | — | — | — | — | — | — | — |
| Pythran | `pythran_periodic_dist::dist` | yes | yes | yes | yes | — | yes | MaskWrite, MaskCompaction |
| Pythran | `pythran_perm::permutations` | — | — | — | — | — | — | — |
| Pythran | `pythran_projection_simplex::projection_simplex` | yes | yes | yes | yes | — | yes | Sort, MaskCompaction |
| Pythran | `pythran_pythagorean_triples::next_pythagorean_triples` | — | — | — | — | — | — | — |
| Pythran | `pythran_reverse_cumsum::reverse_cumsum` | — | — | — | — | — | — | — |
| Pythran | `pythran_rosen::rosen` | — | — | — | — | — | — | — |
| Pythran | `pythran_smoothing::smoothing` | — | — | — | — | — | — | — |
| Pythran | `pythran_specialconvolve::specialconvolve` | — | — | — | — | — | — | — |
| Pythran | `pythran_spectraldns::add_pressure_diffusion` | — | — | — | — | — | — | — |
| Pythran | `pythran_spectraldns::cross2` | — | — | — | — | — | — | — |
| Pythran | `pythran_sumarray3d::summation` | — | — | — | — | — | — | — |
| Pythran | `pythran_tax::tax` | yes | yes | yes | — | — | yes | MaskCompaction |
| Pythran | `pythran_value_index::tri` | — | — | yes | yes | — | — | Sort |
| Pythran | `pythran_vibr_energy::calculate_vibr_energy` | — | — | — | — | — | — | — |
| Pythran | `pythran_wdist::slow_wdist` | — | — | — | — | — | — | — |
| Pythran | `pythran_weights::weights` | — | — | — | — | — | — | — |
