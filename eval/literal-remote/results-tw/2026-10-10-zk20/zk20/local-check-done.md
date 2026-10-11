# Local correctness checks — zkvm20

Working tree: `/home/zhantong/Projects/zinnia-project/wt/literal`
Branch: `all-merged`; commit: `a65574b5`.

All runs used execute mode only (no proofs), with outputs checked against NumPy by `benchmarking/baselines/zkvm_common/check_outputs.py` using repository fixtures A and B and the checker’s configured tolerances. Builds and checks used `CARGO_BUILD_JOBS=3` and `MEMRUN_MAX=16G`. Updated the prepared `build-local.sh` from 4 to 3 build jobs.

## Final status

| System | Fixture A | Fixture B | Passed | Failed | Skipped |
|---|---|---|---|---|---|
| Cairo | 17/17 | 17/17 | 34 | 0 | 0 |
| SP1 | 20/20 | 20/20 | 40 | 0 | 0 |
| RISC0 | 20/20 | 20/20 | 40 | 0 | 0 |
| Total | 57/57 | 57/57 | 114 | 0 | 0 |

The previously completed native checks (40/40) are not included in the total above. Cairo does not cover the two NPBench entries or the Pythran entry.

## Cairo results

| Case | Fixture A | Fixture B |
|---|---|---|
| ds1000_case0313 | ok | ok |
| ds1000_case0330 | ok | ok |
| ds1000_case0356 | ok | ok |
| ds1000_case0357 | ok | ok |
| ds1000_case0365 | ok | ok |
| ds1000_case0370 | ok | ok |
| ds1000_case0382 | ok | ok |
| ds1000_case0413 | ok | ok |
| ds1000_case0417 | ok | ok |
| ds1000_case0418 | ok | ok |
| ds1000_case0419 | ok | ok |
| ds1000_case0430 | ok | ok |
| ds1000_case0456 | ok | ok |
| ds1000_case0457 | ok | ok |
| ds1000_case0458 | ok | ok |
| ds1000_case0463 | ok | ok |
| ds1000_case0502 | ok | ok |

Detailed results: `cairo-A.json`, `cairo-B.json`, `cairo-A.log`, `cairo-B.log`.

The prior Scarb build lacked executable artifacts for cases 0456, 0457, 0458, 0463, and 0502. Built these five targets successfully (`cairo-build-missing.log`) and reran all 17 cases on both fixtures. Initial artifact failures are preserved in `cairo-{A,B}.initial.{json,log}`; they were build-artifact issues, not output mismatches.

## SP1 and RISC0 results

Both guest/host builds completed successfully. Build logs: `sp1-build.log`, `risc0-build.log`.

| Case | SP1 A | SP1 B | RISC0 A | RISC0 B |
|---|---|---|---|---|
| ds1000_case0313 | ok | ok | ok | ok |
| ds1000_case0330 | ok | ok | ok | ok |
| ds1000_case0356 | ok | ok | ok | ok |
| ds1000_case0357 | ok | ok | ok | ok |
| ds1000_case0365 | ok | ok | ok | ok |
| ds1000_case0370 | ok | ok | ok | ok |
| ds1000_case0382 | ok | ok | ok | ok |
| ds1000_case0413 | ok | ok | ok | ok |
| ds1000_case0417 | ok | ok | ok | ok |
| ds1000_case0418 | ok | ok | ok | ok |
| ds1000_case0419 | ok | ok | ok | ok |
| ds1000_case0430 | ok | ok | ok | ok |
| ds1000_case0456 | ok | ok | ok | ok |
| ds1000_case0457 | ok | ok | ok | ok |
| ds1000_case0458 | ok | ok | ok | ok |
| ds1000_case0463 | ok | ok | ok | ok |
| ds1000_case0502 | ok | ok | ok | ok |
| npbench_deep_learning_resnet_resnet/resnet_basicblock | ok | ok | ok | ok |
| npbench_polybench_floyd_warshall_floyd_warshall/kernel | ok | ok | ok | ok |
| pythran_ccopy/CCOPY | ok | ok | ok | ok |

Detailed results: `sp1-A.json`, `sp1-B.json`, `sp1-A.log`, `sp1-B.log`, `risc0-A.json`, `risc0-B.json`, `risc0-A.log`, `risc0-B.log`.

Fixed cases: **none**. No program edits, commits, or pushes were necessary. No remote measurements need to be redone due to these checks.

CAIRO_CHECKS_PASSED
