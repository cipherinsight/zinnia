Completed [REPORT.md](/home/zhantong/Projects/zinnia-project/eval-redo/paper-numbers/C/REPORT.md), including remote timing statistics, SMT shares, per-program evidence, caveats, and reproduction scripts.

| Corpus | SMT queries | Budget exhausted |
|---|---:|---:|
| DS-1000 | 4,193 | 0 |
| NPBench | 33,541 | 0 |
| Pythran | 5,216 | 178 |

All exhaustion occurred in `CCOPY`, with another 4,505 calls skipped after its compilation-wide budget was spent. Observer verification passed; no proofs, keys, or local timing benchmarks were run.