# All accepted programs: degen batch

All nine cases have all four circuit files, committed case by case on all-degen.
0413, 0467 and 0468 pass the complete --no-prove checker for all four files.
The six degenerate cases pass A/B output checks, Halo2 mocks, Noir witness
execution and counts, but cannot receive a full checker PASS without a tool fix:
the unmodified checker unconditionally fails when no output-sensitive input
exists. No failing circuit draft is left; the remaining obstacle is checker
classification, not an observed output or constraint failure.

Checks used lock-b, MEMRUN_MAX=8G, dev-fast builds with CARGO_BUILD_JOBS=3,
CARGO_TARGET_DIR=target (optimized) and target-literal with HW_VARIANT=literal.
Work, TMPDIR, logs and evidence are under
`/home/zhantong/Projects/zinnia-project/eval-redo/all-work/degen`.
Nargo/noirc: 1.0.0-beta.20 (b4236c1957d0c26cb65d82adc9e5447b6ff1d629).
Barretenberg: 5.0.0-nightly.20260324; driver's noir-recursive target.
No setup, key generation, proof, verification, timing benchmark, scaling, or
Zinnia compiler run. No IR, emitted circuit code, or compiler sizes were read.

## Fixtures and validation scope

The nine case folders initially lacked fixture.json. External authoring fixtures
are in `fixtures/ds1000_caseNNNN/fixture.json`, passed via --fixture-root.
Input A comes from record.json generate_test_case (0291: test 4; 0373 and 0413:
test 2; others: test 1). For 0467/0468 the original size parameter is the kernel's
compile-time constant (3,3). Input B uses NumPy RandomState(91000 + numeric id):
normal floats or integers in [-50,50]; complex B has imaginary components
[1,-2,3,-4] at indices [0,3,6,9], zero elsewhere. Nullary kernels have empty A/B.
These fixtures are not committed or claimed to be canonical RQ1 fixtures.

Raw checker results are `NNNN-opt.json` and `NNNN-lit.json`; each contains
per-step results and counts. `run-case.sh NNNN` reproduces builds/checks against
these fixtures. `source-hashes.json` records SHA-256 of kernels, circuits and
fixtures. Fixture agreement is not an exhaustive full-domain correctness proof.
No shared gadget, checker, case-set, kernel or repository fixture was modified.
The harmless Python sys.prefix warning caused by the relative .venv executable
also appears in logs; all required Python operations completed.

`P` below means full --no-prove PASS including domain rejection. `D` means
A/B outputs, mock/execute, and counts passed, but the checker reports exactly
"no input element to move out of its domain". The four status letters are
optimized Halo2, optimized Noir, literal Halo2, literal Noir. For 0373 explicit
grades:0 domain tampering was additionally rejected by all four circuits;
metadata-only and nullary cases have no applicable input-element domain test.

Counts are Halo2 advice + lookup (total), and unpadded Noir gates. These are
**automatic sizing** counts, not fixed-lookup-width comparison results.

| Case | Optimized Halo2 | Literal Halo2 | Optimized Noir | Literal Noir | Four-file status | (k, lookup bits), opt / lit |
|---|---:|---:|---:|---:|---|---|
| ds1000_case0291 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | D/D/D/D | (5, 4) / (5, 4) |
| ds1000_case0349 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | D/D/D/D | (5, 4) / (5, 4) |
| ds1000_case0373 | 1350 + 400 (1750) | 133650 + 31025 (164675) | 3029 | 28125 | D/D/D/D | (9, 8) / (10, 8) |
| ds1000_case0400 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | D/D/D/D | (5, 4) / (5, 4) |
| ds1000_case0401 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | D/D/D/D | (5, 4) / (5, 4) |
| ds1000_case0451 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | D/D/D/D | (5, 4) / (5, 4) |
| ds1000_case0413 | 1732 + 240 (1972) | 4161 + 200 (4361) | 3149 | 3480 | P/P/P/P | (7, 6) / (8, 7) |
| ds1000_case0467 | 640 + 192 (832) | 640 + 192 (832) | 3084 | 3084 | P/P/P/P | (7, 6) / (7, 6) |
| ds1000_case0468 | 640 + 192 (832) | 640 + 192 (832) | 3084 | 3084 | P/P/P/P | (7, 6) / (7, 6) |

## Circuit files

| Case | Optimized Halo2 | Optimized Noir | Literal Halo2 | Literal Noir |
|---|---|---|---|---|
| ds1000_case0291 | [target.rs](../datasets/ds1000/ds1000_case0291/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0291/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0291/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0291/main.literal.nr) |
| ds1000_case0349 | [target.rs](../datasets/ds1000/ds1000_case0349/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0349/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0349/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0349/main.literal.nr) |
| ds1000_case0373 | [target.rs](../datasets/ds1000/ds1000_case0373/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0373/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0373/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0373/main.literal.nr) |
| ds1000_case0400 | [target.rs](../datasets/ds1000/ds1000_case0400/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0400/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0400/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0400/main.literal.nr) |
| ds1000_case0401 | [target.rs](../datasets/ds1000/ds1000_case0401/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0401/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0401/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0401/main.literal.nr) |
| ds1000_case0451 | [target.rs](../datasets/ds1000/ds1000_case0451/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0451/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0451/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0451/main.literal.nr) |
| ds1000_case0413 | [target.rs](../datasets/ds1000/ds1000_case0413/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0413/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0413/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0413/main.literal.nr) |
| ds1000_case0467 | [target.rs](../datasets/ds1000/ds1000_case0467/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0467/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0467/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0467/main.literal.nr) |
| ds1000_case0468 | [target.rs](../datasets/ds1000/ds1000_case0468/target.rs) | [main.nr](../datasets/ds1000/ds1000_case0468/main.nr) | [target.literal.rs](../datasets/ds1000/ds1000_case0468/target.literal.rs) | [main.literal.nr](../datasets/ds1000/ds1000_case0468/main.literal.nr) |

## Interpretations and per-case review

- 0291: signature shape is [2,5,5,2]. Reads array metadata, not values;
  CONVENTIONS 1.6 gives no input-element domain checks. Constants are declared
  in output layouts, omitted from public instances. Literal retains the source
  assignment. Equal optimized/literal counts are expected for metadata only.
  No twin reused. Evidence: 0291-opt.json and 0291-lit.json in the work directory.

- 0349: metadata-only column count 6; same interpretation as 0291, equal sizes.
  Reused 0291 plumbing with its own [8,6] signature and scalar output.

- 0400: empty rank-one array; no inputs or public cells. Both variants retain
  the empty construction. Equal counts are expected. No twin reused.

- 0401: shape (3,0), flattened empty output; no inputs or public cells.
  Adapted 0400 construction comment and shape semantics; equal counts expected.

- 0451: all 4000 zeros of shape (20,10,10,2) are declared, with no public
  instances. Literal materializes the constant array; equal counts expected.
  No twin reused.

- 0373: the source reads all 50 grades in np.sort, so both variants check all
  input domains. Optimized drops the unused sort; literal keeps the helper call
  and all 1225 bubble-sort comparisons, with fresh bounds per comparison.
  The shape-only arange/division is folded with floor(i * 2^32 / 50) in both
  variants, not nearest rounding. Literal is larger in both backends.
  Explicit grades:0 tampering is rejected by both mocks and both Noir executions;
  evidence 0373-explicit-domain.json supplements the checker's failed search.
  No twin reused. No new witness hints or extra output assertions.

- 0413: complex fixture values use [real,imag] pairs; both parts are Q32 and
  checked in the signed 64-bit window. This follows Inputs::complexes,
  hw_check.reference and prover_toml conventions. No magnitude approximation
  or dropping the imaginary part: the test is exact imaginary-code != 0.
  The public length counts complex elements (0..10), then 20 interleaved
  component cells with zero padding, declared dynamic float[10x2].
  Optimized uses shared two-component compaction; imaginary payload is already
  zero for rejected items, so its redundant mask multiplication is omitted.
  Literal materializes the projection/mask, running counts, and all 100
  input/output equality pairs, summing both components. It is larger on both
  backends. Input/output plumbing adapted from 0412; literal scan idiom from
  0442, computation derived from this kernel. No duplicate twin assumed.
  Extra no-proof checks pass with zero/all selected values, negative imaginary
  parts, and +/-2^-32 imaginary values. Both variants reject explicit x:1
  imaginary-domain tampering in addition to the standard real-part tamper.
  Evidence: 0413-{opt,lit,edge-opt,edge-lit}.json and
  0413-imaginary-domain.json. Historical case-set zinnia_rejects status is
  retained by the checker; no compiler was run to reassess it.

- 0467: 16 clipped windows, outer row then column; each window is flattened
  row-major, giving 100 integer output cells in list order. Every one of the
  16 private integers is checked once, and repeated copies alias those wires.
  Literal retains window(arr, size), all bound assignments in their source
  positions and the slice append. Bounds, floor and integer casts depend only
  on shape (positive 3/2 floors to 1) and are folded at compile time.
  Optimized Rust aliases static slices; optimized Noir uses the static index
  list. Literal and optimized counts are equal because data movement adds no
  arithmetic. No twin reused; 0467 was completed before adapting 0468.

- 0468: adapted all four completed 0467 circuits. The outer loop is now column
  then row, and ymin/ymax precede the inner loop while xmin/xmax are inside it.
  Each individual slice stays row-major. Regenerated the optimized Noir index
  list for this ordering. Shape, 100 outputs, domain checks, and sizes agree
  with the twin. Both variants passed their own A/B fixtures and domain tests.

## Final audit and outstanding limits

Both variants' hw_check.lint runs across all nine cases report **0 errors**.
The sole warning is 0373's widening cast of a compile-time loop index (0..49)
to u64, used to fold the positive constant Q32 division; it is safe.
Literal sources were reviewed against L1-L7: helpers remain helpers, full sort
and masks are retained, no custom hints or forbidden literal gadgets, fresh
comparison bounds, fixed source loop order, and identical output/domain plumbing.

No literal circuit is smaller. Equal in both backends: **0291, 0349, 0400,
0401, 0451, 0467, 0468** (metadata/constants or static wire copies).
Both 0373 and 0413 are strictly larger in both backends.

Not finishable under the permitted file scope: obtaining unqualified checker
PASS for **0291, 0349, 0373, 0400, 0401, 0451** requires the checker to handle
constant-output/no-input cases. Its tools were deliberately left unmodified.
No proving or tampered-public verification was performed; only mocks, witness
execution, input-domain rejection and counts were verified. Canonical-fixture
integration and proof validation remain outside this report's verified scope.
