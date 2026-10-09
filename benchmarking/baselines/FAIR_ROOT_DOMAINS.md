# Six-entry source and arithmetic audit

Original1x; compiler frozen at4115cecb. No compiler code is changed.

## Corrections

- `deuxd_convolution`: each loop product rounds before addition; repeated
  identical clamped products and output computations use exact CSE.
- `make_decision`: componentwise rounded squares, sum, sqrt, rounded square,
  sequential first argmin and selected symbol. No monotone-distance shortcut.
- `hdiff`: flux predicate compares the rounded product with zero, not raw sign.
- `tax`: preserve four masks and compacted arrays, then each source segment's
  formula. Input sum used as divisor is not prematurely restricted to signed64;
  retain the source nonzero assertion. Halo2 dynamic division and Noir qdiv
  use their respective frozen accepted divisor ranges.
- `clip` and `CROTG`: abs uses individually rounded component squares before
  sqrt. Complex/real division retains rounded x*d and d*d, without cancellation.
  CROTG conjugates before rounding each signed component product: floor(-x)
  is not generally -floor(x). Branch on rounded abs(CA), not exact raw squares.

## Inactive branches and domain

Frozen IR executes division and multiplication nodes even when their source
branch is inactive, while nonzero assertions are guarded. Thus replacing all
inactive operands with harmless constants changes the accepted domain. The
updated clip/CROTG retain original intermediate products; division maps zero
to zero via denominator+iszero and numerator*(1-iszero), and asserts nonzero
only on the source's active branch. Clip selects the normalized value to zero
before the final multiplication, matching the source's guarded function return.
CROTG computes all intermediate products before selecting branch outputs.

All dynamic denominators in these two cases are nonnegative Q32 product
results, bounded below2^63. Positive division constraints therefore need no
extra divisor sign check. Two component squares have individual signed64
quotient checks. Noir's complex-abs sum is consequently nonnegative below2^64,
within qsqrt's sign/magnitude domain. Halo2's generic input square loses
nonnegative range information (independent interval multiplication exceeds
LIMIT_BITS120), so sqrt uses its signed64 sign fallback. The handwritten
Halo2 complex-magnitude helper now enforces that same sum window. Explicit
source sqrt assertions later in CROTG also use signed64 comparisons.

Unused initial C/S parameters in CROTG are omitted under the preexisting
observable-output contract; their fixed-dtype input checks in compiler IR are
reported as layout/checking overhead, not missing arithmetic. Original kernels,
fixtures and their declared parameter types are unchanged.

## Validation status

The earlier stage passed exact integer-Q32 A/B comparisons by Noir witness
execution for all six entries. After the user's renewed compile-first direction,
validation is compilation/count-only, with no witness execution, MockProver,
setup, proving or remote runs. Earlier witness results do not certify subsequent
domain corrections. Current source hashes and counts are in
`eval-redo/npb-hw/fairness/corrected-counts.json`; every row records provenance.
