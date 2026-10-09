# NPBench/Pythran fair circuit-size comparison

Effective 2026-10-09, following the user's requirement for the same algorithm
and fair arithmetic. This contract overrides conflicting permissions in
CONVENTIONS.md, NPB_BASELINES.md and historical implementation headers.

## Scope and historical results

Original 1× shapes only; circuit sizes only. No proving, key generation,
timing or scaling runs. The per-job ceiling is 64 GiB; local runs must also
respect the machine's smaller physical memory.

The earlier 79-entry optimized-baseline measurements and passed fixture proofs
are historical. Their aggregate ratios (1.501× Halo2, 1.300× Noir) are withdrawn
as a fair compiler-overhead comparison. Fixture agreement within tolerance does
not establish arithmetic or algorithm equivalence. Changed sources do not inherit
old proof-validation labels. The optimized baseline sources remain in git at
commit d47bb13f.

## Required comparison contract

1. Preserve the source algorithm: iteration order, recurrence, reductions,
   data-dependent branches, tie policy, and observable outputs. Do not replace
   scalar Fibonacci with matrix powers, insertion sort with sorting networks,
   exp/log expressions with algebraically simplified identities, or complex
   magnitude comparisons with exact raw squared-distance comparisons.
2. Use the same arithmetic contract on both sides: Q32 representation, exact
   multiplication/rescale boundaries, division and square-root rounding,
   signed comparison widths, casts, transcendental approximations and domains.
   Binary64 NumPy agreement is a separate numerical check, not a substitute
   for agreement with this fixed-point contract. A source dtype such as float32
   must not introduce a baseline-only rounding step.
3. A scalar loop product is rounded before accumulation. Fused product sums
   are allowed only where the agreed lowering actually uses that primitive.
   Squaring followed by summation does not automatically authorize fusion.
   Inspect frozen IR to audit primitive semantics, not to copy circuit layout.
4. Backend gadget implementation may be efficient, and exact CSE, constant
   folding, dead-output elimination, and range-proof reuse are allowed when
   they preserve the operations' arithmetic and acceptance domain. Reordering,
   symmetry reductions and algorithm replacements require a separate study.
5. Match input/output domains and preconditions. Do not shrink a loop bound or
   input domain using assumptions made only by one implementation. Document
   compiler limitations separately; do not silently call them NumPy semantics.
6. For Halo2 report advice and lookup cells separately and together; use a
   common lookup width for a controlled comparison. Report automatic sizing
   separately if desired. For Noir report unpadded UltraHonk gate counts with
   the same toolchain/settings. Neither metric is a proving-time ratio.

## Evidence and reporting

Every entry needs an explicit algorithm audit, arithmetic audit, source hashes,
original-shape counts, common backend settings and validation status. Use
compile/count-only and non-proving arithmetic/witness checks. Cases with
unresolved semantic or arithmetic differences are excluded from the fair
aggregate, and listed with the exclusion reason and coverage. No corpus-wide
fairness claim may be inferred from a few repaired outliers.

Shared transcendental arithmetic and all 79 entries are under audit. Reports
in eval-redo/npb-hw/diagnosis describe why the historical numbers differed.
