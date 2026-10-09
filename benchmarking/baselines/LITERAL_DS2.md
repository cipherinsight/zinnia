# DS2 literal baselines

In progress. All checks use `--no-prove`; no keys or proofs are generated.

| Case | Literal halo2 advice + lookup = total | Literal Noir gates | Optimized halo2 | Optimized Noir | Verification / notes |
|---|---:|---:|---:|---:|---|
| 0340 | 1200 + 360 = 1560 | 3004 | not run | not run | Both pass A/B and input-domain checks. Reused unchanged after line-by-line audit; flattening is a row-major wire copy. |
| 0345 | 5374 + 628 = 6002 | 3515 | not run | not run | halo2 pass; Noir pass. Reused; corrected augmented assignment to gather, operate on slice, scatter.  |
| 0346 | 4692 + 472 = 5164 | 3497 | not run | not run | halo2 pass; Noir pass. Reused; corrected augmented assignment to gather, operate on slice, scatter.  |
| 0347 | 5624 + 706 = 6330 | 3566 | not run | not run | halo2 FAIL; Noir pass. Reused; corrected augmented assignment to gather, operate on slice, scatter. 6 inputs checked twice |
| 0359 | 276 + 84 = 360 | 2854 | not run | not run | halo2 pass; Noir pass. Reused; closure capture repaired.  |
| 0360 | 552 + 168 = 720 | 2891 | not run | not run | halo2 pass; Noir pass. Reused and audited.  |
| 0361 | 960 + 288 = 1248 | 2966 | not run | not run | halo2 pass; Noir pass. Reused; closure capture repaired.  |
| 0362 | 1600 + 480 = 2080 | 3066 | not run | not run | halo2 pass; Noir pass. Reused; closure capture repaired. Added missing Noir file.  |
