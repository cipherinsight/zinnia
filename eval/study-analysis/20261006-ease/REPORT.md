# Python/Zinnia versus Rust/RISC0: all 12 valid participants

The researcher confirmed that **P0 is a valid study participant**, and that all 12 surveys represent different people. Zhaoyu distributed P3/P11 to two different participants. This report supersedes the earlier 11-person analysis and its mistaken pilot classification.

Source: immutable `all-results-20261006-173802/study.sqlite3` backup; 72 tasks, 36 per language. Latest retained submissions are used, including authorized redos. No live results were modified. Reproduce with `python analyze.py` (system Python with NumPy). Source checksum and full results are in `results.json`; token-free task data are in `task_data.csv`.

## Participant results

Times are average recorded active time per task, including tasks reaching the 600-second limit. Correctness means passing all final hidden tests, which is distinct from finishing before timeout.

| Participant | Python mean time | Rust mean time | Python tasks passed | Rust tasks passed |
|---|---:|---:|---:|---:|
| P0 | 3:16 | 8:51 | 3/3 | 1/3 |
| P1 | 4:55 | 6:00 | 2/3 | 3/3 |
| P2 | 5:33 | 5:58 | 3/3 | 3/3 |
| P3 | 4:25 | 9:12 | 3/3 | 2/3 |
| P4 | 4:23 | 4:57 | 3/3 | 3/3 |
| P5 | 3:58 | 5:16 | 3/3 | 3/3 |
| P6 | 7:11 | 9:11 | 3/3 | 3/3 |
| P7 | 5:25 | 8:05 | 3/3 | 3/3 |
| P8 | 2:25 | 7:21 | 3/3 | 3/3 |
| P9 | 3:01 | 7:38 | 3/3 | 3/3 |
| P10 | 2:38 | 6:33 | 3/3 | 3/3 |
| P11 | 1:17 | 9:35 | 3/3 | 3/3 |

**All 12 participants averaged less active time with Python/Zinnia.**

## Paired hypothesis tests

For each participant, average their three tasks in each language and compare Rust minus Python. Participants, not the 72 individual task observations, are the units of inference.

The directional research hypothesis is lower programming effort with Python/Zinnia. The main two-sided exact sign test has null probability 0.5 that a participant's time difference favors either language; under a continuous paired distribution this is a median-difference test. All 12 favor Python, giving **p = 0.00048828125**. This is exploratory post-hoc analysis, not a preregistered confirmatory test.

| Endpoint | Paired participants | Python/Zinnia | Rust/RISC0 | Two-sided sign-flip p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Active time per task | 12 | 242.24 s (4:02) | 443.01 s (7:23) | 0.000488 | 0.002441 |
| First passing version, capped time | 12 | 217.57 s (3:38) | 362.71 s (6:03) | 0.001465 | 0.005859 |
| Test attempts per task | 12 | 2.03 | 4.03 | 0.003418 | 0.010254 |
| Difficulty, 1 easy–7 difficult | 8 | 3.375 | 4.042 | 0.023438 | 0.046875 |
| Final hidden-test success | 12 | 35/36 (97.2%) | 33/36 (91.7%) | 0.750000 | 0.750000 |

The sign-flip tests enumerate all paired sign flips under a symmetric paired-null assumption. They are not randomization tests of the actual schedule assignment. For active time, the sign-flip and exact sign-test p-values coincide. Holm correction covers all five endpoints above.

Mean active-time saving is **200.77 seconds/task (3:21)**, a **45.3% ratio-of-means reduction**. A participant-paired percentile bootstrap (100,000 resamples, fixed seed) gives a 95% interval of **125.64–282.40 seconds** for the mean saving. These intervals are exploratory, not exact small-sample intervals.

Time to first passing version is **145.14 seconds lower**, bootstrap interval **81.26–211.65**. It requires both visible/custom and linked hidden tests to pass, or a passing final submission. A task with no passing version is assigned 600 seconds; this is a capped effort-to-success measure, not an uncensored time-to-completion estimate. Test timestamps are at request, excluding that run's subsequent execution latency. An intermediate correct version counts even if the final version regresses.

Difficulty includes only eight participants with all six ratings present: P0, P1, P3 and P11 have unrated timeouts. Missingness is related to difficulty and may bias the complete-case result. The conservative sign test for the rating difference gives **p = 0.0703**, so subjective ease remains suggestive rather than robust stand-alone evidence. Raw ordinal scores are averaged as an exploratory summary.

Attempts count visible and custom runs only; hidden/final grading is not counted again. Fewer attempts can also reflect different testing habits. Final correctness does **not** show a significant language difference (p = 0.75).

## Sensitivity analyses

| Analysis | Participants | Python/Rust mean active time (seconds) | Two-sided exact sign-test p | Five-endpoint Holm-adjusted time p |
|---|---:|---:|---:|---:|
| Latest retained results | 12 | 242.24 / 443.01 | 0.000488 | 0.002441 |
| Restore original P1/P6 Rust timeout cases | 12 | 242.24 / 450.63 | 0.000488 | 0.002441 |
| Exclude known reset schedules P1, P2, P4, P6 | 8 | 198.06 / 468.86 | 0.007813 | 0.039063 |

The reset-exclusion subset is a sensitivity calculation, not a claim that remaining records have no assistance, earlier exposure or procedural changes. P0 remains included as a valid participant. Restoring two original cases uses the pre-reset database and does not reconstruct every participant's first exposure. All retained participants in each scenario favor Python on active time.

A descriptive participant/task/period fixed-effects model estimates 200.77 extra Rust seconds in the full balanced cohort, matching the unadjusted difference. No unclustered task-level significance test is used.

## Claim and limitations

Recommended wording:

> In our exploratory crossover study of 12 participants and six array-programming tasks, Python/Zinnia required less recorded active task time than Rust/RISC0 for every participant. Mean time was 242 versus 443 seconds per task, a 45.3% reduction (two-sided exact paired sign test, p = 0.00049; five-endpoint Holm-adjusted p = 0.00244). Time to the first passing version and test-attempt counts also favored Python/Zinnia. Final correctness did not differ significantly.

This supports **lower recorded programming effort for these participants and tasks**. Eight of 12 participants report greater Python familiarity, three equal familiarity, and one greater Rust familiarity. The within-person design controls stable participant differences but cannot remove language-specific familiarity. Consequently the results do not isolate an intrinsic compiler-architecture advantage or establish superiority at equal language experience.

Other qualifications: active time includes reading and testing as well as coding, and some tasks have long intervals after passing; timeouts are capped; redos, assistance and evolving descriptions/manuals complicate first-exposure comparisons. P6 encountered a likely Zinnia guarded-indexing compiler defect. Hidden diffusion tests have known blind spots, so test-suite success is measured correctness rather than proof over every input. The entire analysis is exploratory and not preregistered.
