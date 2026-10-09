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
- Ising's dE comparison uses `lte_i`/`lt_i` with the signed64 operand domain;
  its handwritten dE window matches that comparison. The canonical exp
  inv-ln2 product has magnitude greater than its argument, so its signed64
  rescale already excludes arguments outside signed64.
- Periodic-distance comparisons use the signed64 domain as well. Disabled
  branches' extra difference windows do not reject otherwise-valid results:
  every uncorrected difference must subsequently have a signed64 Q32 square.
  Both directed masks remain sequential, and each directed square remains
  separately rounded.
- GoodExpo's final IntCast uses truncation and `sign_and_abs` on its dividend
  in the frozen backend. The signed64 check after multiplying by255 therefore
  reproduces a downstream requirement; it is not the lost square check above.
  Its nonnegative product is below 2^63, so the 38-bit hundredths integer
  decomposition (100*x / 2^32 plus rounding) is sufficient.
- Insertion/bubble comparisons operate on selected original bounded values.
  Sorting primitive uses in projection-simplex and value-index retain the
  compiler's signed64 comparisons and stable argsort ties. Projection-simplex
  checks its quotient in a comparison, and its final gap in a maximum; their
  signed64 windows are not additional restrictions.
- CRC's 64-bit signed two's-complement shift/XOR domain is explicit in the
  compiler bitwise backend. Fibonacci remains field addition without a
  selected-state/output int64 check, as documented in the algorithm inventory.

This is a focused audit of the operations above, not a certification of all
79 cases. Other owners cover nonpivoted LU/ludcmp/cholesky and shared numerical
primitives. No aggregate is reinstated by these corrections.

## Validation

Both changed Noir targets compile with `nargo compile`. Rust validation
uses `cargo check --offline --profile dev-fast` restricted to the two changed cases and three audited control cases,
with `RUSTC_BOOTSTRAP=1` required by the pinned dependency's feature attributes.
Heavy commands serialize with `/tmp/hw_check_heavy.lock`, an 8 GiB systemd cap,
and at most two Cargo jobs. Compile-only records are under
`eval-redo/npb-hw/fairness/domain-algorithms-compile/`.
