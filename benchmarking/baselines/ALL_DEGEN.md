# All accepted programs: degen batch

All four baselines per case are authored from kernel.py, not compiler output.
Checks use --no-prove, the assigned lock-b, 8 GiB cap, dev-fast builds (three
Cargo jobs), and external work directories under eval-redo/all-work/degen.
No setup, proving, verification, timing benchmark, scaling, or compiler run.

The nine case folders initially had no fixture.json. External authoring fixtures
are in all-work/degen/fixtures/<case>/fixture.json and passed via --fixture-root.
Input A comes from the matching record.json generate_test_case (the test named
in kernel.py); B uses seeded NumPy RandomState draws (91000 + numeric case id).
These fixtures are not committed and are not claimed to be canonical RQ1 fixtures.

The unmodified checker unconditionally fails when no output-sensitive input
can be found. This affects every degenerate case, including nullary kernels.
Such results are reported as TOOL-BLOCKED, never as a full checker pass.

Each row names its folder under benchmarking/datasets/ds1000; its four files
are target.rs, main.nr, target.literal.rs, main.literal.nr. Counts are automatic
sizing, advice + lookup (total), and unpadded Noir gates, respectively.

| Case | Optimized Halo2 | Literal Halo2 | Optimized Noir | Literal Noir | Four-file check status |
|---|---:|---:|---:|---:|---|
| ds1000_case0291 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED domain-target discovery |

| ds1000_case0349 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED domain-target discovery |

| ds1000_case0400 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED domain-target discovery |

| ds1000_case0401 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED domain-target discovery |

| ds1000_case0451 | 0 + 0 (0) | 0 + 0 (0) | 34 | 34 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED domain-target discovery |

| ds1000_case0373 | 1350 + 400 (1750) | 133650 + 31025 (164675) | 3029 | 28125 | All A/B outputs, mocks/executes, counts pass; TOOL-BLOCKED automatic domain-target discovery; explicit tamper rejected in all four |

| ds1000_case0413 | 1732 + 240 (1972) | 4161 + 200 (4361) | 3149 | 3480 | PASS all four, including domain rejection |

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
