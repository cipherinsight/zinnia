# Moderate-difficulty task revision

Six tasks, 600 seconds each. Keep the 12-person Williams-square crossover
allocation, two untimed practices, public examples, and private scoring of
all attempts. Every task supplies its implementation approach and manual links.

| Task | Simplification | Remaining operations |
| --- | --- | --- |
| Channel energy | No batch axis; one matrix and shared column parameters | Calibration, clipping, minimum, squared sum |
| Cross products | No batch axis or higher-rank products | Column centering, transpose, matrix multiply |
| Weighted distances | No projection; exactly three coordinates | Point pairing, weighting, reduction |
| Image filter | One image/channel/filter, stride 1, no dilation | Contiguous windows, elementwise multiply, sum |
| Block energy | 2×2 blocks; no channel-pair covariance | Local minimum, square, sum per channel |
| Masked diffusion | Scalar grid; no batches or channel mixing | Mask, neighbors, three synchronous rounds |

The reduction removes simultaneous relationships rather than merely reducing
input sizes. Five tasks use at most two input dimensions; only block energy
requires a rank-three input. (Point pairing can use an optional rank-three
intermediate.) This is a design judgement, not evidence of human completion
time: the pilot should establish whether ten minutes is now adequate.

All v5 contracts receive new IDs. Previous task IDs and saved historical
assignments continue to dispatch to their original contracts. Do not combine
v4 and v5 outcomes as if they were the same task. P010's first-task reset is
backed up and explicitly recorded; this assisted pilot should be excluded
from unassisted participant comparisons.

Run compatibility gate:

```sh
.venv/bin/python -m studies.user_study.revision5.validate
npm run test:crossover --prefix studies/user_study/site
```

The gate uses scalar oracles, public and hidden shape/value cases, Zinnia
array/loop/chained/default-allocation/while/other natural solutions, and Rust
array and loop solutions. Participant execution uses compiled Zinnia IR or
RISC Zero. A finite compatibility inventory cannot establish support for all
possible valid implementations; discovered natural alternatives should be
added as regressions.
