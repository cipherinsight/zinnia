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
