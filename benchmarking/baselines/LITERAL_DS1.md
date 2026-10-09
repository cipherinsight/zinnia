# Literal ds1 baselines

In progress. Checks always use `--no-prove`; no keys or proofs.

| Case | Literal Halo2 advice + lookup | Literal Noir gates | Optimized Halo2 | Optimized Noir | Verdict / notes |
|---|---:|---:|---:|---:|---|
| 0298 | 4889 + 360 = 5249 | 3370 | not run | not run | Noir pass; Halo2 mocks, output comparison and domain rejection pass, but checker flags 5 inputs checked twice. |

0298 reuses the unverified draft, replacing forbidden bounded-input/range-proof reuse with fresh bounds per comparison, materializing unique values, and separating matrix scatter from output packing. Stable bubble sort; equality-test compaction; padded matrix packed to dynamic row stride.

Tool conflict: L5 explicitly requires `q.bound` on each comparison operand, L6 forbids bounded loaders/reuse, while `hw_check.check` still rejects `redundant_input_checks > 0`. Shared tools are not modified. This prevents a clean checker pass for comparisons of loaded inputs; it does not represent a failed constraint or output check.

Arg scan group checked: 0309, 0310, 0311, 0312, 0320, 0321. All Noir checks pass; all Halo2 constraints/output/domain checks pass, with only the redundant-input policy conflict. Fresh bounds are intentional under L5.
