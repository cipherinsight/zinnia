> **Current study protocol:** [12 participants × 6 tasks crossover](site/CROSSOVER.md). Each participant completes each task once, three in each environment, after both practice sessions.

> **Current site assignments:** [Revision 4: distinct stride/dilation and attempt trails](revision4/README.md). The six tasks below describe the original pilot and remain available under their original IDs.

# Revised Zinnia–RISC Zero task suite

This suite combines task redesign and compiler repairs. It is for the reviewer-requested usability study, independent of the research pivot. It compares implementation effort in prepared Zinnia and RISC Zero projects. Two assisted self-pilots have been completed; they are excluded from research exports.

The design goal is that ordinary solutions work without being rewritten into a prescribed compiler-friendly algorithm. All six tasks have fixed output shapes derived from their input shapes. Their conventional algorithms use shape-bounded loops, arithmetic, indexing, reductions, and array transformations. Runtime filtering is still permitted as an intermediate computation.

## What the compatibility gate establishes

The executable inventory contains 55 solution variants across six tasks. **Every inventoried solution must compile and compute the specified result.** There are no accepted failures, exclusions, or expected-failure markers. Both the original function and an instrumented correctness-checking circuit are compiled. Each circuit is compiled once per shape and then evaluated on changing private inputs, including boundary and empty-selection cases. Incorrect expected outputs must be rejected.

This is an empirical compatibility gate, not a proof that every program a person might call “natural” will compile. No finite inventory can certify that unrestricted claim, and Zinnia remains a Python/NumPy subset. Before the participant study, an independent pilot must try additional ordinary solutions; every new valid solution becomes a gate case, and its failure must be fixed or the task revised before freezing the study. Do not delete failing alternatives to pass the gate or hide compiler failures during the study.

The proposed programming setting is numerical arrays in the provided Python/NumPy-style interface or the corresponding Rust project, with loops, ordinary list construction, and compatible libraries available. The task instructions do not restrict algorithms or teach the particular inventory solutions. Full Pandas/PyTorch execution is not part of this prepared-project setting. RISC Zero participants retain ordinary Rust semantics and compatible library choices.

## Tasks, from intended easy to hard

| # | Task | Input → output | Larger validation shape | Inventoried alternatives |
|---|---|---|---|---:|
| 1 | Calibrate and clip readings | `x[N] → y[N]` | `N=32` | 12 |
| 2 | Subtract channel background | `x[T,C] → y[T,C]` | `T=16,C=4` | 8 |
| 3 | Energy above a noise floor | `x[N,C], floor[C] → energy[N]` | `N=16,C=4` | 10 |
| 4 | Pairwise squared distances | `x[N,D], y[M,D] → distance[N,M]` | `N=8,M=6,D=3` | 10 |
| 5 | Sum image blocks | `x[B,H,W,C] → y[B,H/2,W/2,C]` | `B=2,H=W=8,C=3` | 8 |
| 6 | Repeated spatial smoothing | `x[H,W] → y[H,W]`, public fixed `R` | `H=W=8,R=3` | 7 |

Detailed participant-facing statements are in [tasks.md](tasks.md). The order describes expected difficulty; counterbalance presentation order rather than always using this order. The difficulty and session limits still require piloting.

Shapes and the smoothing round count are public project parameters, instantiated before compilation. Input contents are private witnesses. Use bounded integers with ample 64-bit headroom; for the validated profiles, values in `[-512,4096]` fit all intermediate arithmetic. Smoothing accepts nonnegative inputs only. Rust and Zinnia must implement the same integer semantics. Output validation checks shape as well as values.

Straightforward implementations do work proportional to `N`, `T*C`, `N*C`, `N*M*D`, `B*H*W*C`, and `R*H*W`, respectively. An alternative may do more work, such as repeatedly computing a column minimum, but the same public dimensions still bound its ordinary loops. Any filtered row in task 3 has at most `C` elements. None of the specifications requires convergence, an input-dependent number of rounds, or discovery of an unbounded structure.

## Why the tasks changed

Calibration and background subtraction are retained. Filtering now asks for one energy score per observation, allowing masks, scalar conditions, comprehensions, and filtered temporary arrays without requiring compacted rows as the output. Sparse category grouping is replaced by pairwise distances, avoiding an invitation to use unsupported dictionaries or incompatible `unique` behavior. General convolution is replaced by image block pooling, whose windows follow directly from the image dimensions. Farthest-point sampling is replaced by fixed-round spatial smoothing, which naturally invites bounded iteration and ordinary array copying.

These choices narrow the study to bounded numerical tensor programming. They do not support conclusions about arbitrary Python analytics or ecosystem coverage. That limitation should be explicit in the paper.

## Coverage and compiler repairs

The inventory tests scalar and nested loops, `while`, comprehensions, generators, `zip`, append-based construction, conditional appends, filtered arrays, Boolean assignment, broadcasting, method/function aliases, positional/keyword arguments, in-place clipping, matrix products, `tensordot`, `einsum`, reshaping, strided slices, copies, and repeated buffer updates. Four loop variants also exercise chained indexing such as `a[i][j]` for both reads and writes, alongside the original `a[i,j]` spelling.

Repairs made while validating the inventory:

- Recognize NumPy clipping keywords and ndarray `.clip`, including `out` and broadcast bounds.
- Implement `np.copy` and copying expression results without aliasing the source.
- Preserve ndarray semantics for matrix products, so multiplying their result scales values instead of repeating Python lists.
- Support static `zip` in comprehensions and generators.
- Preserve runtime comprehension filters and conditional scalar-list appends using bounded dynamic arrays; make `sum` and `len` respect their logical lengths.
- Extend static `einsum` beyond diagonal extraction to integer/real contractions, reductions, repeated axes, implicit output labels, and ellipsis broadcasting.
- Preserve the minimum signed 64-bit integer during field conversion; the full regression run exposed an overflow when the optional Halo2-base backend constructed integer bounds.

The previously observed dictionary and `np.unique` limitations are not repaired by this work. The revised tasks avoid relying on them. This is not a promise of complete NumPy compatibility.

## Reproduce validation

Run from the `zinnia-src` directory with its development environment:

```sh
PYTHON=.venv/bin/python scripts/develop.sh --offline
.venv/bin/python -m studies.user_study.validate --output studies/user_study/results/validation.json
.venv/bin/python -m studies.user_study.validate --backend halo2-base --profile 0 --fixture 6 --output studies/user_study/results/proofs.json
.venv/bin/python -m pytest -q testing/lang/test_user_study_array_compat.py
```

The default run has 202 implementation/shape combinations and seven input fixtures each: 1,414 correct-output checks plus 202 incorrect-output controls. Shape profiles include singleton dimensions, rectangular tensors, multiple batches/channels, a grid without interior cells, and the larger shapes above. The real-proof run (halo2-base; the legacy `halo2` backend is retired because its comparisons cannot express the 64-bit input checks) uses one nonzero randomized fixture at profile 0 for every alternative; it does not replace the broader mock-backend correctness checks. A mock result is not a cryptographic proof.

Fixtures cover the allowed numeric extremes, equality with thresholds, empty and full selections, and thresholds that differ between channels. Current JSON reports record the working source, native extension, inventory, and validator hashes, as well as the Python/NumPy versions and exact selections. A dirty checkout is recorded explicitly; the Git revision alone does not identify this development build.

`results/before.json` preserves the original 41-variant inventory before fixes (66/87 implementation/shape checks passed). `validation.json` and `proofs.json` use the final 55-variant inventory; the other JSON reports are intermediate snapshots. These different totals must not be interpreted as participant outcomes or a controlled performance comparison.

See [results/REPORT.md](results/REPORT.md) for final results, the exact validation build, and regression coverage.

## Interactive study environment

The [study site](site/README.md) provides participant links, one fixed language per participant, tutorial/practice, all six timed tasks, sandboxed Zinnia and RISC Zero execution, and researcher monitoring/exports. It is a local study tool; time allowances and participant materials still need a human pilot.

## Remaining study preparation

An additional [independent implementation pilot](pilots/README.md) now samples ten GPT-5.6 agents, with ten submitted programs per task. Its frozen submissions and separate results extend the compatibility evidence beyond the curated inventory. This is an LLM compatibility pilot, not a human usability study.

Validate independent pilot solutions against this gate, then freeze the compiler build and task contracts. Prepare and verify equivalent RISC Zero templates and implementations, choose participant/task assignment and time limits from the pilot, and preregister outcomes. This repository work does not demonstrate Rust/Zinnia usability differences, establish task difficulty empirically, or guarantee acceptance of unforeseen implementations.

Apply the same task contracts, correctness tests, preparation, and access to documentation in both conditions. Measure correctness, time, assistance, and errors without assuming that Zinnia wins; Rust may be equally usable or easier on individual tasks. Compiler failures encountered in the eventual study remain observed failures. The prepared workload supports conclusions about these bounded numerical tasks, not general-purpose language usability.
