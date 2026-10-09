# Algorithm-case accepted-domain audit

Static comparison against the frozen compiler, 2026-10-09. This revision
changes handwritten targets only. No witness, MockProver, setup, proof, or
remote job was run for these changes.

## Concrete repairs

| Cases | Mismatch | Correction and boundary example |
|---|---|---|
| Log-likelihood | An extra signed-64 check on `2 * sigma²`. | Reuse the known nonnegative divisor as its magnitude without a signed64 check. With all data equal to mean and sigma=40000, sigma² fits Q32 but twice its encoding exceeds signed64. |
| GoodExpoMeasure | Direct integer squaring skipped the intermediate MulF window after FloatCast/SubF. | Restore cast, subtraction, rounded Q32 square, then division. Channel value50128 gives d=50000: d² exceeds the Q32 window. At sigma=1000000 the old shortcut reached exp(-2500)=0 and accepted a value that the frozen square rejects. |

Dynamic DivF evidence: `fixed_div` in
`src/prove/halo2_base/synthesizer/fixed_point.rs` dispatches unknown factors to
`div_rem`; its unknown-zero branch masks the numerator and replaces the divisor
by one. Nonzero dynamic divisors use a hinted sign and two unsigned64
remainder/gap checks, without a signed64 divisor assertion. The source frontend additionally asserts nonzero denominators: the log-likelihood
repair retains rejection of zero through its remainder bounds. The existing
`div_by_abs` constraint is valid here: twice the nonnegative signed64 square
is below 2^64, so all integer products remain far below the field modulus.
GoodExpo evidence: `sub_f` calls `sub_i`, which does not propagate the
`float_casts` factor alias. `fixed_mul` therefore rescales the square and bounds
its quotient; exact integer squaring was insufficient to reproduce that check.

These boundary examples follow directly from the constraints. They were not
executed, and are not being presented as witness-validation results.

## Other examined operations

- Deriche and pivoted LU have frontend-inserted nonzero division guards even
  without explicit source preconditions. Fresh IR shows one such guard for
  Deriche and four pivot guards for each LU variant. Their zero-denominator
  rejection is correct; primitive totalization alone does not describe the
  whole circuit domain. Log-likelihood has three nonzero assertions: source
  sigma, twice sigma-squared, and density scale.
- LU's absolute-value limit is intentional: compiler `magnitude` constrains
  the result below 2^63, including when a prior sign is known. `abs_int64`
  excluding the magnitude of -2^63 is therefore not a discovered mismatch.
- Ising uses signed64 dE in Halo2, but Noir's `<=0` is a 64-bit gap
  comparison. Noir now uses that canonical comparator. All sites=1000000000
  and beta=0 give first dE=16000000000000000000, within that gap but outside
  signed64. Both targets now retain unmasked `-dE*beta` for exp: saved IR
  ExpF45/65/85/104 consumes MulI directly, even in an inactive source branch.
- Periodic-distance evaluates and bounds L/2 and -L/2 unconditionally, as
  DivF1218/2186 in saved IR. All flags false, L=2^40, zero coordinates must
  still reject. Noir comparisons use 64-bit gaps, not operand windows:
  L=2^31 and x alternating +/-2^30 can produce a raw delta outside signed64
  which the periodic correction brings to zero. Halo2 retains operand bounds.
  The final source sqrt nonnegative assertion bounds its sum to signed64
  in Halo2 and unsigned64 in Noir; these guards are now explicit. Three
  coordinate differences40000 give individual squares1.6e9 but sum4.8e9,
  exposing the missing sum guard without overflowing any component square.
- GoodExpo's final IntCast differs between frozen backends: Halo2 uses
  signed64 `sign_and_abs`, while Noir uses `sign_abs` with a 64-bit magnitude.
  Halo2 retains its signed64 check. Noir multiplies the rounded value by255
  exactly, checks the nonnegative dividend to 64 bits, then divides by SCALE;
  truncation equals floor for this nonnegative value. For example, all three
  channels129 and sigma=-0.1875 produce a product near exp(16), whose value
  times255 lies between 2^31 and 2^32. This differentiates the input windows.
  The earlier nonnegative exp product is below 2^63, so the 38-bit hundredths integer
  decomposition (100*x / 2^32 plus rounding) is sufficient.
- Insertion/bubble comparisons operate on selected original bounded values.
  Value-index's packed stable keys span fewer than KEY_BITS bits and preserve
  all signed64 input keys. Its one-hot permutation constraints retain the
  compiler's stable ties; ignored score_t is covered by typed input policy.
- Projection-simplex now compares the actual source `u-quotient` gap with
  zero. Halo2 bounds this gap to signed64; Noir uses the 64-bit gap comparator
  both here and for the final maximum. All DivF quotients remain bounded.
  In Noir, v=[2^30,-2^30,...,-2^30], z=2^31+1 can have final top gap slightly
  greater than 2^31 in Q32; an extra signed64 gap window wrongly rejected it.
- Weights now uses the frozen backend's comparator domains instead of a
  66/70-bit extended comparator: Halo2 signed64 threshold-times-length,
  Noir 64-bit operand gap. At length2, threshold=-2^30-1 exposes the former
  mismatch; threshold=-2^31 exposes the latter. Equality match counts,
  symmetric reuse, and bounded reciprocal for total1..5 remain exact. Total0
  is rejected as required by the frontend's five nonzero division assertions.
- CRC's 64-bit signed two's-complement shift/XOR domain is explicit in the
  compiler bitwise backend. Fibonacci remains field addition without a
  selected-state/output int64 check, as documented in the algorithm inventory.

This is a focused audit of the operations above, not a certification of all
79 cases. Other owners cover nonpivoted LU/ludcmp/cholesky and shared numerical
primitives. No aggregate is reinstated by these corrections.

## Validation

Changed Noir targets compile with `nargo compile`. Rust validation
uses `cargo check --offline --profile dev-fast` restricted to the affected cases,
with `RUSTC_BOOTSTRAP=1` required by the pinned dependency's feature attributes.
Heavy commands serialize with `/tmp/hw_check_heavy.lock`, an 8 GiB systemd cap,
and at most two Cargo jobs. Compile-only records are under
`eval-redo/npb-hw/fairness/domain-algorithms-compile/`.

## Completed review scope

After the corrections above, this finite static audit identified no remaining
algorithm, arithmetic-schedule, or accepted-domain mismatch in these14 entries
under the typed-input contract and each frozen backend's own constraints:
Fibonacci, log-likelihood, Ising, CRC16, insertion-sort, bubble-sort, Deriche,
GoodExpoMeasure, periodic-distance, pivoted LU0, pivoted LU1, projection-simplex,
value-index, and weights. Unused typed inputs follow the project's documented
observable-output policy. This is static review plus compilation, not a formal
proof or full-domain witness experiment. Nonpivoted LU/ludcmp and other
numerical kernels are tracked by their separate audit owner.

The final four-case comparator/control compilation record is
`eval-redo/npb-hw/fairness/domain-algorithms-control-compile/results.json`.
No execution, MockProver, setup or proving was performed.
