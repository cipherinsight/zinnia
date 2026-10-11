# Epoch 2026-10-10-zk42-retry1-ds1000_case0386-zkvm

Ratios are Zinnia / baseline of per-case medians; below 1 means Zinnia is faster.
Only cases with status `ok` (proved, verified, outputs equal to NumPy, tampered proof rejected) enter a comparison.


## Coverage

| system | cases | ok | other statuses |
|---|---|---|---|
| sp1 | 1 | 0 | out_of_memory: 1 |
| risc0 | 1 | 0 | crashed: 1 |
| cairo | 1 | 1 | - |

## Headline: all-supported subset without copy-only cases

0 cases completed by every system that do more than copy inputs: (none)

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|

## Copy-only cases (reported apart)

0 all-supported cases that only copy inputs: (none)

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|

## All-supported subset

0 cases completed by every system of the epoch (sp1, risc0, cairo): (none)

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|

## Pairwise subsets

Each pair on the cases both of its systems completed.

| Zinnia | baseline | clock | n | geomean ratio | 95 % CI | Zinnia faster / slower |
|---|---|---|---|---|---|---|

## Excluded cases

(none)

## Cases without a program

No program for the case on that system (e.g. no handwritten baseline yet); not a failure.

(none)

## Failures by system

- sp1 ds1000_case0386: out_of_memory at warmup: exit -9 after 167.0s
- risc0 ds1000_case0386: crashed at warmup: exit -15 after 209.5s
