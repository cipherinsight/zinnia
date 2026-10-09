# Faithful algorithm revisions

These six entries now follow the source algorithm rather than substituting an
algebraically equivalent task. This revision does not change compiler code.
It must be combined with the canonical shared exp/log/tanh gadget revision
before its sizes are used in a same-arithmetic comparison.

| Entry | Restored computation | Permitted local choices |
|---|---|---|
| `pythran_fibo_seq__fibo` | 1024 bounded sequential `(a,b)=(a+b,a)` steps, conditional on private n | Source precondition 0<=n<=1024; no additional state range assertions beyond the frozen compiler. No n<=90 shortcut or matrix powers. |
| `pythran_log_likelihood__log_likelihood` | Per-element square, divide, exp, density normalization divide, log, then sum | Common scalar denominator and sqrt(2*pi) constant reused; no log-density identity. |
| `pythran_ising__ising_step` | Sequential site updates and literal `exp(-dE*beta)>0.5` predicate | Constant neighbor indices; inactive exp argument replaced by zero. No logarithmic threshold; float32 shares canonical Q32 as in the selected compiler. |
| `npbench_crc16_crc16__crc16` | Each byte's eight sequential conditional CRC shift/XOR steps | State carried as 64 Boolean bits, input bit decompositions reused, arithmetic right shift retains sign. No GF2 history/parity reformulation. |
| `pythran_insertion_sort__insertion_sort` | Source insertion saves, comparisons, shifts, and final placement | While active, j=i-step by induction. Constant indices therefore replace dynamic addressing without changing compared pairs or loop order. |
| `pythran_bubble_sort__bubble_sort` | Nested adjacent comparison/swap passes | The source sets swap_test=True unconditionally inside every nonempty pass, so its early break is unreachable. |

Sorts use no sorting network or unconstrained permutation. Source shapes and
fixtures are unchanged. CRC keeps the private polynomial fully signed int64;
it is not specialized to the fixture polynomial.

## Validation performed

Under an 8 GiB memory cap and at most two Cargo jobs:

- All six Rust circuits compiled together with `cargo build --profile dev-fast`.
- Static lint: zero errors for all six entry pairs.
- Noir compile and witness execution succeeded on inputs A and B for all six
  entries. All 12 decoded outputs matched the existing NumPy reference and
  tolerance. This run used the inherited shared transcendental implementation,
  so it validates this algorithm rewrite, not the forthcoming canonical gadget.
- No setup, proof generation, verification, MockProver, or scaling run occurred.

Raw compilation/witness execution evidence is preserved outside the source tree
at `eval-redo/npb-hw/fairness/algorithms-witness-inherited-gadgets.json`.
Revalidate log-likelihood and Ising by compilation/witness execution after
canonical gadget integration. Old optimized size records and old proof results
are historical and do not validate these revised circuits.

## Numerical scope

The selected comparison is the compiler's Q32 arithmetic, not literal binary64
or binary32. In particular the older Ising binary32-boundary regression tests a
different statement and must not be used as an expected NumPy-equivalence claim
for this canonical-Q32 comparison. Log-likelihood now materializes the density;
Q32 underflow before log is no longer bypassed. Fibonacci uses field additions as the frozen compiler does; the intended int64
result scope n<=90 is not an enforced output range in that compiler. All1024
conditional source steps remain represented. See FAIR_ALGORITHMS_INVENTORY.md
for exact IR evidence and the remaining accepted-domain qualification.

Shared gadget arithmetic remains a separately integrated dependency. These
changes alone do not assert that the other 73 entries have passed a complete
same-algorithm/same-arithmetic audit, and no revised aggregate is claimed here.

## Further restorations

Deriche now computes exp(-2*alpha) directly. GoodExpo now materializes the
rounded Q32 hundredths before multiplying by255, truncating, and wrapping to
uint8. Periodic-distance now computes every directed difference, applies both
periodic masks in sequence, and squares every directed value independently.
The four owned transcendental callers (log-likelihood, Ising, Deriche,
goodExpo) explicitly call canonical shared APIs. Compilation/witness results
before those APIs arrive do not validate this follow-up revision.
