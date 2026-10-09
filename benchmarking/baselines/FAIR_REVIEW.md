# Integrated NPBench/Pythran fairness review

All79 original1x entries have been statically reviewed against the same source
algorithm and the selected frozen backend arithmetic. 39 entry pairs changed
from the optimized d47bb13f snapshot. No remaining known algorithm/arithmetic or
accepted-domain mismatch was identified in this review, within the typed-input
and observable-output policy in NPB_FAIRNESS.md. This is a review conclusion,
not formal equivalence or proof verification.

Coverage partitions all79 without gaps:53 numerical entries
(FAIR_REDUCTION_AUDIT.md and fair_reduction_inventory.json),14 algorithm/control
entries (FAIR_ALGORITHMS.md and FAIR_ALGORITHM_DOMAINS.md),6 root-owned entries
(FAIR_ROOT_DOMAINS.md), and6 canonical-transcendental callers (shared arithmetic
report at workspace eval-redo/npb-hw/fairness/shared-arithmetic.md).

Historical inventory entries that say pending/not_certified describe earlier
audit stages. The integrated per-entry inventory and final count provenance are
at workspace eval-redo/npb-hw/fairness/inventory.json and corrected-counts.json.
No final-source proof validation is claimed. The user's current instruction is
compile first, count original-size circuits, no local proving, and no remote
proving until explicit approval. No compiler, kernel, fixture or shape changed.

Fairness includes backend-specific constraints: Halo2 and Noir differ in some
signed-width and gap-comparison domains. Each handwritten circuit matches its
own frozen backend. Omitted unused typed inputs and exact CSE follow the shared
observable-output policy; these are not alternate source algorithms. Noir
canonical functions explicitly reuse the frozen arithmetic library, while its
case algorithms remain handwritten. Halo2 arithmetic schedules/coefficients
match the frozen contract with independently authored constraints.

The old1.501x/1.300x size ratios remain withdrawn as fair headlines. Use only
fresh counts whose source and shared-library hashes match this audited snapshot.
Circuit sizes do not establish proving speed. Remote proof validation, when
approved, must use the final audited source hashes and unchanged1x fixtures.
