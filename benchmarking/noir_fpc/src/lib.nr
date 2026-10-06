// noir_fpc -- the Q32 fixed-point library of Zinnia's Noir backend.
//
// Encoding. A float x is the field element round(x * 2^32) ("Q32"); an
// integer n is the field element n. A negative value v is stored as p - |v|
// (p is the BN254 scalar modulus, about 2^253.6). Helpers interpret a field
// element as the signed integer of smallest magnitude it represents.
//
// Value window. Zinnia's integers are int64 and its Q32 floats are int64
// fixed-point values, so every value of a program lives in the signed 64-bit
// window [-2^63, 2^63). The helpers that need an order (comparisons, abs,
// sign, division, rescaling) accept the window (and slightly more, as each
// doc comment states) and are unsatisfiable (the proof fails) outside it,
// as Zinnia's halo2-base backend. All range checks are 64 bits or narrower,
// far below p/2, which is what makes the uniqueness arguments below work.
//
// Soundness convention. A helper that asks the prover for a value through an
// `unconstrained` hint then constrains that value so that it is the ONLY one
// that satisfies the constraints: the output is a function of the inputs, so
// a malicious prover cannot substitute another one. Each helper states the
// argument in its doc comment, and the tests at the bottom of this file feed
// the constraint functions malicious hint values (`#[test(should_fail)]`).
//
// Semantics. The helpers reproduce the reference semantics of Zinnia's mock
// backend, `src/prove/kernel.rs`: `//` and `%` round toward negative infinity
// (Python), fixed-point `/` is Euclidean, float -> int truncates, Q32 products
// are floored, and a zero divisor gives 0. sqrt, sin, cos, tan and arcsin run
// the kernel's own fixed-point algorithms, so they agree with the mock bit for
// bit; exp, log, pow, sinh, cosh, tanh, arccos and arctan2 are polynomial
// approximations of functions the mock evaluates in f64 (a few Q32 units).
//
// Range checks. `x.assert_max_bit_size::<N>()` constrains 0 <= x < 2^N. A
// truncating cast `x as uN` on a Field is NOT a range check: it silently keeps
// the low N bits. Use `checked_u8` .. `checked_u64` to convert a Field to an
// unsigned integer type; after the range check Noir elides the truncation, so
// the checked form is also cheaper than the bare cast. Barretenberg pays a
// lookup table per distinct width, so the helpers use 64 and 32 bits wherever
// a wider check would also be sound.

pub global Q: Field = 4294967296;
global Q_U128: u128 = 4294967296;

// 2^63: x + 2^63 is a 64-bit value exactly when x lies in [-2^63, 2^63).
global WINDOW_OFFSET: Field = 9223372036854775808;

// pi, 2 pi, pi/2 in Q32 (kernel::pi_q, half_pi_q).
global PI_Q: Field = 13493037705;
global TWO_PI_Q: Field = 26986075410;
global HALF_PI_Q: Field = 6746518852;

// ---------------------------------------------------------------------------
// Range checks
// ---------------------------------------------------------------------------

/// Field -> u8 that fails unless 0 <= x < 2^8.
pub fn checked_u8(x: Field) -> u8 {
    x.assert_max_bit_size::<8>();
    x as u8
}

/// Field -> u16 that fails unless 0 <= x < 2^16.
pub fn checked_u16(x: Field) -> u16 {
    x.assert_max_bit_size::<16>();
    x as u16
}

/// Field -> u32 that fails unless 0 <= x < 2^32.
pub fn checked_u32(x: Field) -> u32 {
    x.assert_max_bit_size::<32>();
    x as u32
}

/// Field -> u64 that fails unless 0 <= x < 2^64.
pub fn checked_u64(x: Field) -> u64 {
    x.assert_max_bit_size::<64>();
    x as u64
}

/// Field -> u128 that fails unless 0 <= x < 2^128.
pub fn checked_u128(x: Field) -> u128 {
    x.assert_max_bit_size::<128>();
    x as u128
}

/// Constrains x to the signed 64-bit window [-2^63, 2^63): x + 2^63 must be a
/// 64-bit value. Any other x wraps to a field element of about 254 bits.
pub fn assert_in_window(x: Field) {
    (x + WINDOW_OFFSET).assert_max_bit_size::<64>();
}

/// `idx` as the index of an array of `last + 1` elements, after proving
/// 0 <= idx <= last, for `last < 2^(W-1)` and W <= 32. The checks are on
/// 2 * idx and last - idx, never on idx itself: ACVM drops a range check of a
/// memory index (it trusts the backend's memory check), and Barretenberg
/// before its 2026-05-13 fix (the pinned bb 5.0.0-nightly.20260324 included)
/// admits a read at index p - 1 because a sorted memory chain may start
/// there. Soundness: 2 * idx in [0, 2^W) leaves idx in [0, 2^(W-1)) or a
/// field element (p + t) / 2 with t odd, about p / 2; last - idx in
/// [0, 2^W) rules out the latter and every idx > last. Callers pick W = 8 or
/// W = 28, widths whose lookup tables bb already builds for 64-bit checks
/// (4 limbs of 14 bits and one of 8). The final range check only lets Noir
/// drop the truncation of the cast (which is then exact).
pub fn array_index<let W: u32>(idx: Field, last: Field) -> u32 {
    assert(W <= 32, "array_index: W must be <= 32");
    assert(last.lt(2.pow_32((W - 1) as Field)), "array_index: array too long for W");
    (2 * idx).assert_max_bit_size::<W>();
    (last - idx).assert_max_bit_size::<W>();
    idx.assert_max_bit_size::<32>();
    idx as u32
}

// ---------------------------------------------------------------------------
// Sign and magnitude
// ---------------------------------------------------------------------------

// Hint: whether v is negative in its signed reading, i.e. whether its
// additive inverse is the smaller field element of the two.
unconstrained fn hint_is_negative(v: Field) -> bool {
    (-v).lt(v)
}

// Hint: (negative, |v|) for the signed reading of v.
unconstrained fn hint_sign_magnitude(v: Field) -> (bool, Field) {
    if hint_is_negative(v) {
        (true, -v)
    } else {
        (false, v)
    }
}

/// Constraints for `sign_abs`; returns |v| for the claimed sign. The
/// magnitude m = (negative ? -v : v) must satisfy m - negative in [0, 2^64).
/// Uniqueness: with the wrong sign m is the field negation of a value in
/// [0, 2^64] (v = 0 claimed negative gives m - 1 = -1), a field element
/// above p - 2^65, which fails. So the sign and |v| are functions of v, and
/// exactly the v in [-2^64, 2^64) are accepted.
fn constrain_sign_abs(v: Field, negative: bool) -> Field {
    let mag = if negative { -v } else { v };
    (mag - (negative as Field)).assert_max_bit_size::<64>();
    mag
}

/// (v < 0, |v|) for v in [-2^64, 2^64); the proof fails for any other v.
/// One 64-bit range check.
pub fn sign_abs(v: Field) -> (bool, Field) {
    // Safety: constrain_sign_abs pins the sign (see there).
    let negative = unsafe { hint_is_negative(v) };
    (negative, constrain_sign_abs(v, negative))
}

/// |v| for v in (-2^63, 2^63), the int64 values whose magnitude is itself
/// an int64; the proof fails for any other v (|i64::MIN| = 2^63 overflows
/// int64 and, like every int64 overflow, has no proof). With the hinted sign
/// and m = (negative ? -v : v), one 64-bit range check of t = 2m + negative
/// (64 bits, a whole number of the backend's range limbs; a 63-bit check of m
/// costs more gates). For negative = false that is m in [0, 2^63); for
/// negative = true, 2m + 1 < 2^64, i.e. m in [0, 2^63 - 1]: together exactly
/// |v| <= 2^63 - 1. Uniqueness: for v != 0 the wrong sign gives
/// m = p - |v| and t = p - 2|v| (+1), far above 2^64. At v = 0 both signs give
/// m = 0, so the result is a function of v even where the sign hint is not
/// (which is why the sign is not returned).
pub fn abs_int64(v: Field) -> Field {
    // Safety: the range check pins m (see above).
    let negative = unsafe { hint_is_negative(v) };
    let mag = if negative { -v } else { v };
    (2 * mag + (negative as Field)).assert_max_bit_size::<64>();
    mag
}

pub fn is_negative(v: Field) -> bool {
    sign_abs(v).0
}

pub fn qabs(v: Field) -> Field {
    sign_abs(v).1
}

pub fn qabs_i(v: Field) -> Field {
    qabs(v)
}

/// Integer sign: -1, 0 or 1.
pub fn qsign_i(v: Field) -> Field {
    let (negative, mag) = sign_abs(v);
    sign_of(negative, mag)
}

/// -1, 0 or 1 from a pair returned by `sign_abs` (which never returns a
/// negative zero). The `*_of` helpers take such a pair, so that one
/// `sign_abs` serves every helper that needs the sign of the same value.
pub fn sign_of(negative: bool, mag: Field) -> Field {
    if mag == 0 {
        0
    } else if negative {
        -1
    } else {
        1
    }
}

/// Float sign as a Q32 value: -1.0, 0.0 or 1.0.
pub fn qsign(v: Field) -> Field {
    qsign_i(v) * Q
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

// Hint: a < b in the signed reading of a - b.
unconstrained fn hint_lt(a: Field, b: Field) -> bool {
    hint_is_negative(a - b)
}

/// Constraints for `qlt_bits::<W>`: the claimed answer c shifts the
/// difference, and t = a - b + c * 2^W must be a W-bit value. For c = false
/// that is a - b in [0, 2^W), for c = true it is a - b in [-2^W, 0): the two
/// ranges are disjoint, so at most one c passes, and one does exactly when
/// a - b reads as an integer in [-2^W, 2^W) (W <= 250 keeps |t| far below p,
/// so no other field element of a - b lands in either range). t is linear in
/// a, b and c, one arithmetic gate before the range check (the form Noir uses
/// for its own integer comparisons).
fn constrain_lt<let W: u32>(a: Field, b: Field, c: bool) {
    let gap = a - b + (c as Field) * 2.pow_32(W as Field);
    gap.assert_max_bit_size::<W>();
}

/// a < b for signed values whose difference lies in [-2^W, 2^W) (W <= 250);
/// the proof fails otherwise, never returns a wrong answer. The mock's
/// `lt_i` reads the sign of a - b, so the two agree wherever this is
/// accepted. One W-bit range check, no check of a or b themselves; a caller
/// that knows the operands' ranges picks a small W (a W below 14 avoids
/// Barretenberg's 14-bit lookup table).
pub fn qlt_bits<let W: u32>(a: Field, b: Field) -> bool {
    assert(W <= 250, "qlt_bits: W must be <= 250");
    // Safety: constrain_lt pins c (see there).
    let c = unsafe { hint_lt(a, b) };
    constrain_lt::<W>(a, b, c);
    c
}

/// a < b for any two values of the signed 64-bit window (any pair whose
/// difference lies in [-2^64, 2^64)).
pub fn qlt(a: Field, b: Field) -> bool {
    qlt_bits::<64>(a, b)
}

pub fn qle(a: Field, b: Field) -> bool {
    !qlt(b, a)
}

pub fn qgt(a: Field, b: Field) -> bool {
    qlt(b, a)
}

pub fn qge(a: Field, b: Field) -> bool {
    !qlt(a, b)
}

pub fn qadd(a: Field, b: Field) -> Field {
    a + b
}

pub fn qsub(a: Field, b: Field) -> Field {
    a - b
}

// ---------------------------------------------------------------------------
// Bitwise operations (two's complement, as the mock's i64 operators)
// ---------------------------------------------------------------------------
//
// For integers in two's complement, a | b = a + b - (a & b),
// a ^ b = a + b - 2 (a & b) and a & m = a - (a & ~m) with ~m = -m - 1, so
// every operator reduces to an AND; an AND with a non-negative mask m only
// reads the low bits of a that are set in m.

/// The 8 * NB low bits of the two's complement of x (least significant
/// first), for x in the signed 64-bit window; the proof fails otherwise.
/// x + 2^63 is split into 8 bytes, which proves it is a 64-bit value (8-bit
/// lookups only), and its NB low bytes into bits. x + 2^63 and x agree
/// modulo 2^(8 NB) for NB <= 7, and every step is a unique decomposition, so
/// no hint is involved.
pub fn low_bits<let NB: u32>(x: Field) -> [bool; NB * 8] {
    assert(NB <= 7, "low_bits: at most 56 bits");
    let bytes: [u8; 8] = (x + WINDOW_OFFSET).to_le_bytes();
    let mut low: Field = 0;
    let mut scale: Field = 1;
    for i in 0..NB {
        low += (bytes[i] as Field) * scale;
        scale *= 256;
    }
    low.to_le_bits()
}

/// a & b for two values of the signed 64-bit window (the mock's i64 `&`);
/// the proof fails outside it. w = x + 2^63 is the two's complement of x
/// with bit 63 flipped, so the native u64 AND of the two w's has the low 63
/// bits of a & b, and the sign bit of a & b is set iff both bits 63 of the
/// w's are clear (both values negative).
pub fn bitand(a: Field, b: Field) -> Field {
    let wa = checked_u64(a + WINDOW_OFFSET);
    let wb = checked_u64(b + WINDOW_OFFSET);
    let a_nonneg = wa >= 9223372036854775808;
    let b_nonneg = wb >= 9223372036854775808;
    let low63 = ((wa & wb) as Field) - ((a_nonneg & b_nonneg) as Field) * WINDOW_OFFSET;
    let negative = !a_nonneg & !b_nonneg;
    low63 - (negative as Field) * WINDOW_OFFSET
}

/// a | b (two's complement): a + b - (a & b).
pub fn bitor(a: Field, b: Field) -> Field {
    a + b - bitand(a, b)
}

/// a ^ b (two's complement): a + b - 2 (a & b).
pub fn bitxor(a: Field, b: Field) -> Field {
    a + b - 2 * bitand(a, b)
}

// ---------------------------------------------------------------------------
// Integer division
// ---------------------------------------------------------------------------
//
// The conventions of kernel::div_rem_or_zero: `n == q*d + r` with |r| < |d|,
//   floor  (Python `//`, `%`; FloorDivI, ModI): r is 0 or has the sign of d;
//   Euclid (fixed-point `/`, DivF):              0 <= r < |d|;
//   trunc  (Python `int()`, IntCast):            r is 0 or has the sign of n;
// and a zero divisor gives (0, 0). Quotients lie in the signed 64-bit window;
// an input whose quotient leaves it has no proof.

// Hint: (q, r) of the signed readings of n and d with the remainder made
// non-negative (`remainder_negative == false`) or non-positive; (0, 0) when
// d == 0 (the constraints then fail).
unconstrained fn hint_divmod(n: Field, d: Field, remainder_negative: bool) -> (Field, Field) {
    let (n_neg, n_mag) = hint_sign_magnitude(n);
    let (d_neg, d_mag) = hint_sign_magnitude(d);
    let nm = n_mag as u128;
    let dm = d_mag as u128;
    if dm == 0 {
        (0, 0)
    } else {
        // Truncated division first: r has the sign of n.
        let qm = nm / dm;
        let rm = nm % dm;
        let q_t: Field = if n_neg != d_neg { -(qm as Field) } else { qm as Field };
        let r_t: Field = if n_neg { -(rm as Field) } else { rm as Field };
        let d_sign: Field = if d_neg { -1 } else { 1 };
        if (rm == 0) | (n_neg == remainder_negative) {
            (q_t, r_t)
        } else if remainder_negative {
            // r_t > 0 must become negative: r = r_t - |d|, q = q_t + sign(d).
            (q_t + d_sign, r_t - (dm as Field))
        } else {
            // r_t < 0 must become non-negative: r = r_t + |d|, q = q_t - sign(d).
            (q_t - d_sign, r_t + (dm as Field))
        }
    }
}

/// Constraints shared by the signed divisions, for a divisor of magnitude
/// `d_mag` (from `sign_abs`, so at most 2^64) and a remainder whose sign the
/// caller fixes (`remainder_negative`). With r_mag = |r|: r_mag and
/// d_mag - 1 - r_mag are 64-bit values, so 0 <= r_mag < d_mag (a zero divisor
/// fails), and q lies in the signed 64-bit window. Uniqueness: |q*d + r| <
/// 2^63 * 2^64 + 2^64 < p / 2, so `q*d + r == n (mod p)` forces q*d + r to
/// equal the signed reading of n, and integer division with a remainder of
/// fixed sign and |r| < |d| has exactly one (q, r).
fn constrain_divmod(n: Field, d: Field, q: Field, r: Field, d_mag: Field, remainder_negative: bool) {
    assert_in_window(q);
    let r_mag = if remainder_negative { -r } else { r };
    r_mag.assert_max_bit_size::<64>();
    (d_mag - 1 - r_mag).assert_max_bit_size::<64>();
    assert(q * d + r == n, "divmod: reconstruction failed");
}

/// Python `(n // d, n % d)` for d != 0: the remainder takes d's sign.
fn floor_divmod_nonzero(n: Field, d: Field) -> (Field, Field) {
    let (d_negative, d_mag) = sign_abs(d);
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, d_negative) };
    constrain_divmod(n, d, q, r, d_mag, d_negative);
    (q, r)
}

/// Euclidean `(q, r)`, 0 <= r < |d|, for d != 0.
fn euclid_divmod_nonzero(n: Field, d: Field) -> (Field, Field) {
    let (_d_negative, d_mag) = sign_abs(d);
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, false) };
    constrain_divmod(n, d, q, r, d_mag, false);
    (q, r)
}

/// Python `(n // d, n % d)` for a divisor the circuit already proves
/// non-zero (an asserted `d != 0`), from `d`'s `sign_abs` pair: no zero
/// mapping, and one `sign_abs` shared by every division by `d`. The pair is
/// pinned by `sign_abs`'s own constraint; `constrain_divmod` then pins
/// (q, r) as in `floor_divmod_nonzero`.
pub fn floor_divmod_nonzero_of(n: Field, d: Field, d_negative: bool, d_mag: Field) -> (Field, Field) {
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, d_negative) };
    constrain_divmod(n, d, q, r, d_mag, d_negative);
    (q, r)
}

/// `qdiv(a, b)` for a divisor the circuit already proves non-zero, from
/// `b`'s `sign_abs` magnitude (see `floor_divmod_nonzero_of`).
pub fn qdiv_nonzero_of(a: Field, b: Field, b_mag: Field) -> Field {
    let n = a * Q;
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, b, false) };
    constrain_divmod(n, b, q, r, b_mag, false);
    q
}

/// (n, d), or (0, 1) when d == 0: dividing the substitute gives the zero
/// divisor's (0, 0) without a special case in the division constraints.
fn zero_divisor_to_unit(n: Field, d: Field) -> (Field, Field) {
    let d_is_zero = d == 0;
    (if d_is_zero { 0 } else { n }, if d_is_zero { 1 } else { d })
}

/// Python `(n // d, n % d)` for any signs; a zero divisor gives (0, 0).
pub fn floor_divmod(n: Field, d: Field) -> (Field, Field) {
    let (n1, d1) = zero_divisor_to_unit(n, d);
    floor_divmod_nonzero(n1, d1)
}

/// Euclidean `(q, r)` with 0 <= r < |d| for any signs; a zero divisor gives
/// (0, 0).
pub fn euclid_divmod(n: Field, d: Field) -> (Field, Field) {
    let (n1, d1) = zero_divisor_to_unit(n, d);
    euclid_divmod_nonzero(n1, d1)
}

/// Constraints for `divmod_by_positive`. `r` and `d - 1 - r` are BITS-bit
/// values, which gives 0 <= r <= d - 1 (so d >= 1 and d < 2^(BITS+1)); with
/// q in the signed 64-bit window and BITS <= 120 the integer q*d + r stays
/// below p/2 in magnitude, so the argument of `constrain_divmod` applies.
fn constrain_divmod_by_positive<let BITS: u32>(n: Field, d: Field, q: Field, r: Field) {
    assert(BITS <= 120, "divmod_by_positive: BITS must be <= 120");
    r.assert_max_bit_size::<BITS>();
    (d - 1 - r).assert_max_bit_size::<BITS>();
    assert_in_window(q);
    assert(q * d + r == n, "divmod_by_positive: reconstruction failed");
}

/// `(n // d, n % d)` for a divisor 1 <= d <= 2^BITS (typically a constant),
/// where floor and Euclidean division agree: 0 <= r < d. The remainder check
/// is two BITS-bit range checks instead of a sign projection of d, and the
/// constraints themselves enforce d >= 1, so a wrong BITS can only make the
/// proof fail, never accept a wrong quotient.
pub fn divmod_by_positive<let BITS: u32>(n: Field, d: Field) -> (Field, Field) {
    // Safety: constrain_divmod_by_positive pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, false) };
    constrain_divmod_by_positive::<BITS>(n, d, q, r);
    (q, r)
}

/// Constraints for `divmod_by_pow2`: r is a K-bit value and q lies in the
/// signed 64-bit window, so |q * 2^K + r| < 2^(63+K) + 2^K < p / 2 for
/// K <= 120 and `q * 2^K + r == n` pins the unique floor quotient and
/// remainder (as `constrain_divmod_by_positive` with d = 2^K, where
/// d - 1 - r >= 0 already follows from r < 2^K).
fn constrain_divmod_by_pow2<let K: u32>(n: Field, q: Field, r: Field) {
    assert(K <= 120, "divmod_by_pow2: K must be <= 120");
    r.assert_max_bit_size::<K>();
    assert_in_window(q);
    assert(q * 2.pow_32(K as Field) + r == n, "divmod_by_pow2: reconstruction failed");
}

/// `(n // 2^K, n % 2^K)`: one K-bit check of the remainder and the window
/// check of the quotient.
pub fn divmod_by_pow2<let K: u32>(n: Field) -> (Field, Field) {
    // Safety: constrain_divmod_by_pow2 pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, 2.pow_32(K as Field), false) };
    constrain_divmod_by_pow2::<K>(n, q, r);
    (q, r)
}

/// Integer `n // d` (Python), any signs; 0 for d == 0.
pub fn q_floor_div(n: Field, d: Field) -> Field {
    let (q, _r) = floor_divmod(n, d);
    q
}

/// Integer `n % d` (Python: the sign of d), any signs; 0 for d == 0.
pub fn q_mod(n: Field, d: Field) -> Field {
    let (_q, r) = floor_divmod(n, d);
    r
}

/// Constraints for `q_floor_div_bounded`: q, r and d - r - 1 are BITS-bit
/// values, i.e. q >= 0 and 0 <= r < d. With BITS <= 120 the integer q*d + r is
/// below 2^242 < p, so the reconstruction holds over the integers and the
/// quotient is unique.
fn constrain_floor_div_bounded<let BITS: u32>(n: Field, d: Field, q: Field, r: Field) {
    assert(BITS <= 120, "q_floor_div_bounded: BITS must be <= 120");
    assert(q * d + r == n, "q_floor_div_bounded: reconstruction failed");
    q.assert_max_bit_size::<BITS>();
    r.assert_max_bit_size::<BITS>();
    (d - r - 1).assert_max_bit_size::<BITS>();
}

/// `n // d` for operands the caller knows to be non-negative BITS-bit values
/// (0 <= n, 1 <= d). If that premise is false the proof fails.
pub fn q_floor_div_bounded<let BITS: u32>(n: Field, d: Field) -> Field {
    // Safety: constrain_floor_div_bounded pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, false) };
    constrain_floor_div_bounded::<BITS>(n, d, q, r);
    q
}

// Hint: (q, r) = (floor(m / Q), m mod Q) for 0 <= m < 2^128.
unconstrained fn hint_divmod_q(m: Field) -> (Field, Field) {
    let mm = m as u128;
    ((mm / Q_U128) as Field, (mm % Q_U128) as Field)
}

/// Constraints for `q_div_q`: q < 2^64, r < 2^32 and q*Q + r == |v| (no wrap:
/// q*Q + r < 2^97), so (q, r) = (floor(|v| / Q), |v| mod Q) is unique.
fn constrain_trunc_div_q(mag: Field, q: Field, r: Field) {
    q.assert_max_bit_size::<64>();
    r.assert_max_bit_size::<32>();
    assert(q * Q + r == mag, "q_div_q: reconstruction failed");
}

/// Float -> int cast: trunc(v / Q), rounding toward zero (Python `int()`),
/// i.e. sign(v) * floor(|v| / Q).
pub fn q_div_q(v: Field) -> Field {
    let (negative, mag) = sign_abs(v);
    q_div_q_of(negative, mag)
}

/// `q_div_q` from the `sign_abs` pair of v.
pub fn q_div_q_of(negative: bool, mag: Field) -> Field {
    // Safety: constrain_trunc_div_q pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod_q(mag) };
    constrain_trunc_div_q(mag, q, r);
    if negative {
        -q
    } else {
        q
    }
}

// Hint: d^-1, or 0 for d == 0.
unconstrained fn hint_inverse_or_zero(d: Field) -> Field {
    if d == 0 {
        0
    } else {
        1 / d
    }
}

/// Constraints for `field_div_or_zero`: for d != 0, d * inv == 1 pins
/// inv = d^-1; for d == 0, inv is forced to 0.
fn constrain_inverse_or_zero(d: Field, inv: Field) {
    let d_is_zero = d == 0;
    assert(d * inv == if d_is_zero { 0 } else { 1 }, "field_div_or_zero: bad inverse");
    assert(!d_is_zero | (inv == 0), "field_div_or_zero: inverse of zero must be 0");
}

/// Field division a * d^-1, and 0 for d == 0 (the IR's `DivI`, as the mock).
pub fn field_div_or_zero(a: Field, d: Field) -> Field {
    // Safety: constrain_inverse_or_zero pins inv uniquely.
    let inv = unsafe { hint_inverse_or_zero(d) };
    constrain_inverse_or_zero(d, inv);
    a * inv
}

// ---------------------------------------------------------------------------
// Fixed-point multiplication and division
// ---------------------------------------------------------------------------

/// Constraints for `qscale_down_raw`: 0 <= r < 2^32 = Q and q in the signed
/// 64-bit window keep |q*Q + r| below 2^96, so `q*Q + r == v` holds over the
/// integers and q = floor(v / Q) is unique.
fn constrain_floor_div_q(v: Field, q: Field, r: Field) {
    r.assert_max_bit_size::<32>();
    assert_in_window(q);
    assert(q * Q + r == v, "qscale_down_raw: reconstruction failed");
}

/// floor(v / Q) for a raw (Q^2-scaled) product or sum of products; the
/// result must lie in the signed 64-bit window.
pub fn qscale_down_raw(v: Field) -> Field {
    // Safety: constrain_floor_div_q pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(v, Q, false) };
    constrain_floor_div_q(v, q, r);
    q
}

/// Q32 product: floor(a * b / Q).
pub fn qmul(a: Field, b: Field) -> Field {
    qscale_down_raw(a * b)
}

/// Q32 quotient a / b: the Euclidean quotient of a * Q by b (floor for b > 0,
/// ceiling for b < 0), as kernel::fp_div_prescale; 0 for b == 0.
pub fn qdiv(a: Field, b: Field) -> Field {
    let (q, _r) = euclid_divmod(a * Q, b);
    q
}

// ---------------------------------------------------------------------------
// Square root
// ---------------------------------------------------------------------------

// Hint: floor(sqrt(n)) for 0 <= n < 2^128.
unconstrained fn hint_isqrt_floor(n: Field) -> Field {
    let target = n as u128;
    let mut g: u128 = target;
    if target > 1 {
        // Newton iteration from above converges to floor(sqrt(target)).
        for _ in 0..130 {
            let next = (g + target / g) / 2;
            if next < g {
                g = next;
            }
        }
    }
    g as Field
}

/// Constraints for `isqrt_floor` (0 <= n < 2^126): y, d = n - y^2 and 2y - d
/// are 64-bit values. y^2 < 2^128 and n < 2^126 keep both differences, read
/// as integers, inside (-2^128, 2^126]; a negative one is a field element
/// above p - 2^128 and fails, so y^2 <= n <= y^2 + 2y < (y + 1)^2: y is the
/// unique floor root (and honest values pass: y < 2^63, d <= 2y < 2^64).
fn constrain_isqrt_floor(n: Field, y: Field) {
    y.assert_max_bit_size::<64>();
    let d = n - y * y;
    d.assert_max_bit_size::<64>();
    (2 * y - d).assert_max_bit_size::<64>();
}

/// floor(sqrt(n)) for 0 <= n < 2^126.
fn isqrt_floor(n: Field) -> Field {
    // Safety: constrain_isqrt_floor pins y uniquely.
    let y = unsafe { hint_isqrt_floor(n) };
    constrain_isqrt_floor(n, y);
    y
}

/// Q32 square root rounded down: floor(sqrt(x * Q)), the largest Q32 value
/// whose square does not exceed x (kernel::fp_sqrt). A negative input gives
/// 0 (the kernel's convention). Fails for x >= 2^32 in real units (raw
/// x >= 2^64), outside the kernel's fixed-point window.
pub fn qsqrt(x: Field) -> Field {
    let (negative, mag) = sign_abs(x);
    qsqrt_of(negative, mag)
}

/// `qsqrt` from the `sign_abs` pair of x.
pub fn qsqrt_of(negative: bool, mag: Field) -> Field {
    let m = if negative { 0 } else { mag };
    isqrt_floor(m * Q)
}

// ---------------------------------------------------------------------------
// Polynomial evaluation (Horner, Q32, leading coefficient first)
// ---------------------------------------------------------------------------

fn horner_q<let N: u32>(x: Field, coefs: [Field; N]) -> Field {
    let mut acc: Field = coefs[0];
    for i in 1..N {
        acc = qmul(x, acc) + coefs[i];
    }
    acc
}

// ---------------------------------------------------------------------------
// sin / cos / tan
// ---------------------------------------------------------------------------

// kernel::sin_odd_coefs_q(32): sin(s)/s as a degree-4 polynomial in u = s^2
// on [0, pi/2] (kernel::SIN_ODD_U_COEFS), leading coefficient first.
global SIN_ODD_Q: [Field; 5] = [11136, -(850500 as Field), 35789648, -(715827153 as Field), 4294967215];

/// Q32 sine by kernel::fp_sin's algorithm (bit for bit the mock's value):
/// r = |x| mod 2*pi_q (an exact checked division); fold r >= pi_q onto
/// r - pi_q (sign flip); mirror r > pi_q/2 onto pi_q - r, so s lies in
/// [0, pi/2]; then s * P(s^2) with flooring products. Error against libm:
/// about 9e-9 on [-20, 20]. Fails for raw |x| >= 2^64, outside the kernel's
/// fixed-point window. The two foldings compare r < 2 pi_q < 2^35 with
/// constants below 2^34 through `qlt_bits::<42>` (whole 14-bit limbs).
pub fn qsin(x: Field) -> Field {
    let (negative, m) = sign_abs(x);
    m.assert_max_bit_size::<64>();
    let (_k, r) = divmod_by_positive::<42>(m, TWO_PI_Q);
    let folded = !qlt_bits::<42>(r, PI_Q);
    let r1 = if folded { r - PI_Q } else { r };
    let mirrored = qlt_bits::<42>(HALF_PI_Q, r1);
    let s = if mirrored { PI_Q - r1 } else { r1 };
    let p = qmul(s, horner_q(qmul(s, s), SIN_ODD_Q));
    if negative != folded {
        -p
    } else {
        p
    }
}

/// Q32 cosine: cos x = sin(x + pi_q / 2) (kernel::fp_cos).
pub fn qcos(x: Field) -> Field {
    qsin(x + HALF_PI_Q)
}

/// Q32 tangent: qsin(x) / qcos(x) (kernel::fp_tan).
pub fn qtan(x: Field) -> Field {
    qdiv(qsin(x), qcos(x))
}

// ---------------------------------------------------------------------------
// exp / log / pow
// ---------------------------------------------------------------------------
//
//   exp(x) = 2^(x / ln 2),  log(x) = log2(x) * ln 2.
// The polynomial coefficient layouts mirror EXP2_COEFS / LOG2_COEFS in
// src/prove/kernel.rs.

global INV_LN2_Q: Field = 6196328019; // round(2^32 / ln 2)

// exp2 on [0, 1), degree 12, quantized to Q32.
global EXP2_COEFS_Q: [Field; 13] = [
    0, 2, 31, 437, 5676, 65510, 661577, 5726720, 41309550, 238388332, 1031764991, 2977044472,
    4294967296,
];

// 2^e for 0 <= e < 128, from the bits of e.
fn pow2(e: Field) -> Field {
    let bits: [bool; 7] = e.to_le_bits();
    let mut acc: Field = 1;
    let mut square: Field = 2;
    for i in 0..7 {
        if bits[i] {
            acc = acc * square;
        }
        square = square * square;
    }
    acc
}

// Hint: (q, r) with v == q * m + r, 0 <= r < m, for 0 <= v < 2^126, m >= 1.
unconstrained fn hint_divmod_nonneg(v: Field, m: Field) -> (Field, Field) {
    let vv = v as u128;
    let mm = m as u128;
    ((vv / mm) as Field, (vv % mm) as Field)
}

/// The index `i` of a packed sort key `K = v * 2^B + i` (Zinnia's
/// `IR::PackKey`: `v` in the signed 64-bit window, `0 <= i < 2^B`, B <= 32).
/// Uniqueness: with `o = 2^63 * 2^B`, `K + o = (v + 2^63) * 2^B + i`; the
/// hint `(q, i)` is checked by `q < 2^64`, `i < 2^B` and `q * 2^B + i == K + o`,
/// whose left side stays below `2^(64+B) < p`, so the field equation is an
/// integer equation with the single solution `(v + 2^63, i)`.
pub fn packed_key_index<let B: u32>(key: Field) -> Field {
    let scale = 2.pow_32(B as Field);
    let shifted = key + 9223372036854775808 * scale;
    // Safety: the three checks below pin (q, i) (see above).
    let (q, i) = unsafe { hint_divmod_nonneg(shifted, scale) };
    i.assert_max_bit_size::<B>();
    q.assert_max_bit_size::<64>();
    assert(q * scale + i == shifted, "packed_key_index: reconstruction failed");
    i
}

#[test]
fn test_packed_key_index() {
    for v in [0, 5, -5, 9223372036854775807, -9223372036854775808] {
        for i in [0, 1, 99, 127] {
            assert(packed_key_index::<7>(v * 128 + i) == i);
        }
    }
}

/// floor(v / m) for 0 <= v < 2^40 and a power of two 1 <= m <= 2^127.
/// Uniqueness: q < 2^40, 0 <= r <= m - 1 (two 128-bit range checks) and
/// q * m + r == v, which stays below 2^168 < p, so (q, r) is unique.
fn floor_div_pow2(v: Field, m: Field) -> Field {
    // Safety: the checks below pin (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod_nonneg(v, m) };
    q.assert_max_bit_size::<40>();
    r.assert_max_bit_size::<128>();
    (m - 1 - r).assert_max_bit_size::<128>();
    assert(q * m + r == v, "floor_div_pow2: reconstruction failed");
    q
}

/// Q32 exp(x). Splits y = x / ln 2 into floor(y) + f with f in [0, 1), takes
/// 2^f from the polynomial and scales by 2^floor(y) exactly (multiplication,
/// or flooring division for negative powers). Fails when |floor(y)| >= 128.
pub fn qexp(x: Field) -> Field {
    let y = qmul(x, INV_LN2_Q);
    let (i, f) = divmod_by_pow2::<32>(y);
    let two_to_f = horner_q(f, EXP2_COEFS_Q); // in [Q, 2Q]
    let (i_negative, e) = sign_abs(i);
    e.assert_max_bit_size::<7>();
    let scale = pow2(e);
    // Both are well defined for every e; the sign picks one.
    let scaled_down = floor_div_pow2(two_to_f, scale);
    let scaled_up = two_to_f * scale;
    if i_negative {
        scaled_down
    } else {
        scaled_up
    }
}

// round(ln 2 * 2^64): k ln 2 is formed in Q64 and rounded once to Q32.
global LN2_Q64: Field = 12786308645202655660;

// round(2^32 / (2j + 1)) for j = 11 down to 0: the series
// atanh(s) / s = sum_j s^(2j) / (2j + 1), leading coefficient first.
global ATANH_SERIES_Q: [Field; 12] = [
    186737709, 204522252, 226050910, 252645135, 286331153, 330382100, 390451572, 477218588,
    613566757, 858993459, 1431655765, 4294967296,
];

// Hint: (k, m) with m = floor(x / 2^k) in [Q, 2Q) (k may be negative, then
// m = x * 2^-k exactly), for 1 <= x < 2^126.
unconstrained fn hint_log2_decompose(x: Field) -> (Field, Field) {
    let xx = x as u128;
    // k = up - down.
    let mut up: u32 = 0;
    let mut down: u32 = 0;
    let mut m: u128 = xx;
    for _ in 0..128 {
        if m < Q_U128 {
            m = m * 2;
            down = down + 1;
        } else if m >= 2 * Q_U128 {
            m = m / 2;
            up = up + 1;
        }
    }
    ((up as Field) - (down as Field), m as Field)
}

/// Q32 natural log of x > 0 (fails for x <= 0).
///
/// x = (m / Q) * 2^k with m in [Q, 2Q): m = floor(x / 2^k) for k >= 0 and
/// m = x * 2^-k exactly for k < 0. These constraints pin k (halving m leaves
/// [Q, 2Q)) and m uniquely. Then ln x = k ln 2 + 2 atanh(s) with
/// s = (m - Q) / (m + Q) in [0, 1/3); the series through s^23 has truncation
/// error below 1e-12, and each Horner level adds at most one Q32 ulp, damped
/// by s^2 < 1/9. Error: a few ulps.
pub fn qlog(x: Field) -> Field {
    let (negative, x_f) = sign_abs(x);
    assert(!negative & (x_f != 0), "qlog: non-positive input");
    // Safety: the checks below pin (k, m) uniquely.
    let (k, m) = unsafe { hint_log2_decompose(x_f) };
    (m - Q).assert_max_bit_size::<32>(); // Q <= m < 2Q
    let (k_negative, e) = sign_abs(k);
    e.assert_max_bit_size::<7>();
    let scale = pow2(e);
    // k < 0: x * 2^-k == m exactly; k >= 0: x - m * 2^k in [0, 2^k).
    let lhs = if k_negative { x_f * scale } else { x_f };
    let m_scaled = if k_negative { m } else { m * scale };
    let bound = if k_negative { 1 } else { scale };
    let rem = lhs - m_scaled;
    rem.assert_max_bit_size::<64>();
    (bound - 1 - rem).assert_max_bit_size::<64>();
    let s = qdiv(m - Q, m + Q);
    let series = horner_q(qmul(s, s), ATANH_SERIES_Q);
    let ln_m = 2 * qmul(s, series);
    let (k_ln2, _r) = divmod_by_pow2::<32>(k * LN2_Q64 + 2147483648);
    k_ln2 + ln_m
}

/// a^b = exp(b * log a) for a > 0.
pub fn qpow(base: Field, exp_q: Field) -> Field {
    qexp(qmul(exp_q, qlog(base)))
}

/// Float base raised to a non-negative integer exponent below 2^32 (not
/// Q-scaled), by square-and-multiply with Q32 rescaling. 0^0 = 1.
pub fn qpow_int_q32(base: Field, exp: u64) -> Field {
    assert(exp >> 32 == 0, "qpow_int_q32: exponent must be below 2^32");
    let mut result: Field = Q;
    let mut b: Field = base;
    let mut e: u64 = exp;
    for _ in 0..32 {
        if (e & 1) == 1 {
            result = qmul(result, b);
        }
        e = e >> 1;
        if e != 0 {
            b = qmul(b, b);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// arcsin, arccos, arctan, arctan2
// ---------------------------------------------------------------------------

// kernel::atan_odd_coefs_q(32): atan(t)/t as a polynomial in u = t^2 on
// [-1, 1] (fit error 1.8e-9), leading first.
global ATAN_ODD_Q: [Field; 10] = [
    -(7489138 as Field),
    45873335,
    -(131930687 as Field),
    246804093,
    -(359577404 as Field),
    469877966,
    -(612529925 as Field),
    858917466,
    -(1431653573 as Field),
    4294967285,
];

/// atan(t) for |t| <= 1 (Q32): t * P(t^2), flooring products.
pub fn qarctan(t: Field) -> Field {
    qmul(t, horner_q(qmul(t, t), ATAN_ODD_Q))
}

/// Q32 arcsin by kernel::fp_arcsin's algorithm (bit for bit the mock's
/// value): arcsin x = 2 atan(x / (1 + sqrt(1 - x^2))), with the exact integer
/// root floor(sqrt(2^64 - x_q^2)) and a flooring division by the positive
/// 2^32 + root. Outside [-1, 1] the result is 0 (the kernel's convention).
/// Error against libm: about 6e-9.
pub fn qarcsin(x: Field) -> Field {
    let (_negative, mag) = sign_abs(x);
    let in_domain = !qlt(Q, mag);
    let v = if in_domain { x } else { 0 };
    let v_mag = if in_domain { mag } else { 0 };
    // 2^64 - v^2 <= 2^64, so the root is at most 2^32.
    let root = isqrt_floor(Q * Q - v_mag * v_mag);
    // Q + root <= 2^33; 42 bits (whole 14-bit limbs) bound the remainder
    // just as 33 would.
    let (t, _r) = divmod_by_positive::<42>(v * Q, Q + root);
    let half = qarctan(t);
    half + half
}

/// Q32 arccos = pi/2 - arcsin x on [-1, 1]; 0 outside (where the mock's f64
/// acos gives nan, which it encodes as 0).
pub fn qarccos(x: Field) -> Field {
    let (_negative, mag) = sign_abs(x);
    if qlt(Q, mag) {
        0
    } else {
        HALF_PI_Q - qarcsin(x)
    }
}

/// Q32 atan2(y, x): the polynomial on min(|x|,|y|) / max(|x|,|y|) in [0, 1],
/// then the octant and quadrant adjustments. atan2(0, 0) = 0.
pub fn qarctan2(y: Field, x: Field) -> Field {
    let (sy, my) = sign_abs(y);
    let (sx, mx) = sign_abs(x);
    let swap: bool = qlt(mx, my);
    let num: Field = if swap { mx } else { my };
    let den: Field = if swap { my } else { mx };
    let poly: Field = qarctan(qdiv(num, den));
    let a0: Field = if swap { HALF_PI_Q - poly } else { poly };
    let a1: Field = if sx { PI_Q - a0 } else { a0 };
    if sy {
        -a1
    } else {
        a1
    }
}

// ---------------------------------------------------------------------------
// Hyperbolic functions
// ---------------------------------------------------------------------------

// floor(v / 2), as the mock's fp_div_prescale(v, 2.0).
fn half_floor(v: Field) -> Field {
    let (q, _r) = divmod_by_pow2::<1>(v);
    q
}

pub fn qsinh(x: Field) -> Field {
    half_floor(qexp(x) - qexp(-x))
}

pub fn qcosh(x: Field) -> Field {
    half_floor(qexp(x) + qexp(-x))
}

pub fn qtanh(x: Field) -> Field {
    qdiv(qsinh(x), qcosh(x))
}

// ---------------------------------------------------------------------------
// Sorting hints (Zinnia's IR::Hint)
// ---------------------------------------------------------------------------
//
// These return prover-chosen values and constrain NOTHING. Zinnia's IR pins
// every lane it uses (strictly increasing (value, index) keys read through an
// argsort, or counts against an order statistic; see the Rust compiler's
// `lower::sort`), so a circuit accepts only the lanes computed here.
// They reproduce `kernel::hint_lane` exactly: the same order (`signed_lt`),
// a stable sort, and lane 0 for a rank outside [0, N).

// a < b in the signed reading of each (the integer of smallest magnitude a
// field element represents): kernel::signed_cmp. A total order, equal to the
// integer order on the signed 64-bit window.
unconstrained fn signed_lt(a: Field, b: Field) -> bool {
    let a_negative = hint_is_negative(a);
    let b_negative = hint_is_negative(b);
    if a_negative != b_negative {
        a_negative
    } else {
        a.lt(b)
    }
}

// The stable ascending argsort of `a` under `signed_lt` (bottom-up merge
// sort; on equal values the left run goes first).
unconstrained fn stable_argsort<let N: u32>(a: [Field; N]) -> [u32; N] {
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        order[i] = i;
    }
    let mut merged: [u32; N] = [0; N];
    let mut width: u32 = 1;
    while width < N {
        let mut lo: u32 = 0;
        while lo < N {
            let mid = if lo + width < N { lo + width } else { N };
            let hi = if lo + 2 * width < N { lo + 2 * width } else { N };
            let mut i = lo;
            let mut j = mid;
            for k in lo..hi {
                let take_left = if i >= mid {
                    false
                } else if j >= hi {
                    true
                } else {
                    !signed_lt(a[order[j]], a[order[i]])
                };
                if take_left {
                    merged[k] = order[i];
                    i += 1;
                } else {
                    merged[k] = order[j];
                    j += 1;
                }
            }
            lo = hi;
        }
        order = merged;
        width *= 2;
    }
    order
}

unconstrained fn hint_argsort<let N: u32>(a: [Field; N]) -> [Field; N] {
    stable_argsort(a).map(|i: u32| i as Field)
}

unconstrained fn hint_sorted<let N: u32>(a: [Field; N]) -> [Field; N] {
    stable_argsort(a).map(|i: u32| a[i])
}

// The ascending sorted values of the active elements (flag 1) of a
// runtime-length array, then 0s: kernel::hint_values, MaskedSortedValues.
// The active elements, kept in index order, are sorted by the same stable
// merge sort.
unconstrained fn hint_masked_sorted<let N: u32>(a: [Field; N], active: [Field; N]) -> [Field; N] {
    let mut order: [u32; N] = [0; N];
    let mut count: u32 = 0;
    for i in 0..N {
        if active[i] != 0 {
            order[count] = i;
            count += 1;
        }
    }
    let mut compact: [Field; N] = [0; N];
    for k in 0..N {
        if k < count {
            compact[k] = a[order[k]];
        }
    }
    let sorted_order = stable_argsort_prefix(compact, count);
    let mut lanes: [Field; N] = [0; N];
    for k in 0..N {
        if k < count {
            lanes[k] = compact[sorted_order[k]];
        }
    }
    lanes
}

// The stable argsort of the first `count` entries of `a` (the rest stay in
// place after them).
unconstrained fn stable_argsort_prefix<let N: u32>(a: [Field; N], count: u32) -> [u32; N] {
    let mut order: [u32; N] = [0; N];
    for i in 0..N {
        order[i] = i;
    }
    let mut merged: [u32; N] = order;
    let mut width: u32 = 1;
    while width < count {
        let mut lo: u32 = 0;
        while lo < count {
            let mid = if lo + width < count { lo + width } else { count };
            let hi = if lo + 2 * width < count { lo + 2 * width } else { count };
            let mut i = lo;
            let mut j = mid;
            for k in lo..hi {
                let take_left = if i >= mid {
                    false
                } else if j >= hi {
                    true
                } else {
                    !signed_lt(a[order[j]], a[order[i]])
                };
                if take_left {
                    merged[k] = order[i];
                    i += 1;
                } else {
                    merged[k] = order[j];
                    j += 1;
                }
            }
            lo = hi;
        }
        order = merged;
        width *= 2;
    }
    order
}

// Lane `rank` of `lanes`, 0 for a rank outside [0, N).
unconstrained fn lane_or_zero<let N: u32>(lanes: [Field; N], rank: Field) -> Field {
    if rank.lt(N as Field) {
        lanes[rank as u32]
    } else {
        0
    }
}

unconstrained fn hint_count_at_most<let N: u32>(a: [Field; N], t: Field) -> Field {
    let mut count = 0;
    for i in 0..N {
        if !signed_lt(t, a[i]) {
            count += 1;
        }
    }
    count
}

/// `#{i : a[i] <= t}` in the signed order (kernel::hint_lane, CountAtMost).
/// A prover hint: the caller must pin it.
pub fn count_at_most_hint<let N: u32>(a: [Field; N], t: Field) -> Field {
    // Safety: a prover hint; Zinnia's IR pins every use.
    unsafe { hint_count_at_most(a, t) }
}

#[test]
fn test_count_at_most_hint() {
    let a = [-5, 0, 0, 7, 9223372036854775807];
    assert(count_at_most_hint(a, -6) == 0);
    assert(count_at_most_hint(a, 0) == 3);
    assert(count_at_most_hint(a, 6) == 3);
    assert(count_at_most_hint(a, 9223372036854775807) == 5);
}

/// The stable ascending argsort of `a` (kernel::hint_values, StableArgsort).
/// A prover hint: the caller must pin every lane it uses.
pub fn argsort_hint<let N: u32>(a: [Field; N]) -> [Field; N] {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { hint_argsort(a) }
}

/// The ascending sorted values of `a` (kernel::hint_values, SortedValues).
/// A prover hint: the caller must pin every lane it uses.
pub fn sorted_hint<let N: u32>(a: [Field; N]) -> [Field; N] {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { hint_sorted(a) }
}

/// The ascending sorted values of the active elements (`active[i] == 1`),
/// then 0s (kernel::hint_values, MaskedSortedValues). A prover hint: the
/// caller must pin every lane it uses.
pub fn masked_sorted_hint<let N: u32>(a: [Field; N], active: [Field; N]) -> [Field; N] {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { hint_masked_sorted(a, active) }
}

/// Lane `rank` of `masked_sorted_hint(a, active)`, 0 for a rank outside
/// [0, N). A prover hint: the caller must pin it.
pub fn masked_sorted_hint_at<let N: u32>(a: [Field; N], active: [Field; N], rank: Field) -> Field {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { lane_or_zero(hint_masked_sorted(a, active), rank) }
}

/// Lane `rank` of `argsort_hint(a)`, 0 for a rank outside [0, N). A prover
/// hint: the caller must pin it.
pub fn argsort_hint_at<let N: u32>(a: [Field; N], rank: Field) -> Field {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { lane_or_zero(hint_argsort(a), rank) }
}

/// Lane `rank` of `sorted_hint(a)`, 0 for a rank outside [0, N). A prover
/// hint: the caller must pin it.
pub fn sorted_hint_at<let N: u32>(a: [Field; N], rank: Field) -> Field {
    // Safety: a prover hint; Zinnia's IR pins every lane it reads.
    unsafe { lane_or_zero(hint_sorted(a), rank) }
}

// ---------------------------------------------------------------------------
// Tests: honest values
// ---------------------------------------------------------------------------

fn q_from_int(x: u64) -> Field {
    (x as Field) * Q
}

// |a - b| <= tol, for the approximate functions' tests.
fn assert_close(a: Field, b: Field, tol: Field) {
    assert(!qlt(tol, qabs(a - b)), "|a - b| > tol");
}

#[test]
fn test_qmul_basic() {
    let a = q_from_int(2) + Q / 2;
    let p = qmul(a, q_from_int(4));
    assert(p == q_from_int(10));
    // floor for negative products: -1.5 * 2^-32 rounds down.
    assert(qmul(-(3 * Q / 2), 1) == -2);
}

#[test]
fn test_qdiv_signs() {
    assert(qdiv(q_from_int(10), q_from_int(4)) == q_from_int(2) + Q / 2);
    // Euclidean floor: -1/3 = -0.333... rounds down to -1431655766 / Q.
    assert(qdiv(-Q, q_from_int(3)) == -1431655766);
    // Negative divisors.
    assert(qdiv(q_from_int(10), -q_from_int(4)) == -(q_from_int(2) + Q / 2));
    assert(qdiv(-q_from_int(10), -q_from_int(4)) == q_from_int(2) + Q / 2);
}

#[test]
fn test_divisions_by_nonzero_divisors() {
    for d in [q_from_int(4), -q_from_int(4), q_from_int(3), -q_from_int(7)] {
        let (_, mag) = sign_abs(d);
        assert(qdiv_nonzero_of(q_from_int(10), d, mag) == qdiv(q_from_int(10), d));
        assert(qdiv_nonzero_of(-q_from_int(10), d, mag) == qdiv(-q_from_int(10), d));
    }
    for d in [4, -4, 3, -7] {
        let (negative, mag) = sign_abs(d);
        let (q, r) = floor_divmod_nonzero_of(-9, d, negative, mag);
        let (q2, r2) = floor_divmod(-9, d);
        assert((q == q2) & (r == r2));
    }
}

#[test(should_fail)]
fn test_qdiv_nonzero_of_rejects_zero_divisor() {
    let _ = qdiv_nonzero_of(Q, 0, 0);
}

#[test(should_fail)]
fn test_divmod_of_rejects_a_shifted_quotient() {
    // The remainder bound pins q: q + 1 with r - d fails it.
    let (negative, mag) = sign_abs(7);
    let (q, r) = floor_divmod_nonzero_of(100, 7, negative, mag);
    constrain_divmod(100, 7, q + 1, r - 7, mag, negative);
}

#[test]
fn test_qdiv_zero_is_zero() {
    // kernel::fp_div_prescale: a zero divisor gives 0.
    assert(qdiv(Q, 0) == 0);
}

#[test]
fn test_neg_arithmetic() {
    assert(qadd(-q_from_int(3), q_from_int(2)) == -q_from_int(1));
}

#[test]
fn test_sign_abs() {
    let (s, m) = sign_abs(-q_from_int(5));
    assert(s & (m == 4294967296 * 5));
    let (s0, m0) = sign_abs(0);
    assert(!s0 & (m0 == 0));
    assert(qabs(-q_from_int(5)) == q_from_int(5));
    // The ends of the accepted range [-2^64, 2^64).
    let (s1, m1) = sign_abs(-18446744073709551616);
    assert(s1 & (m1 == 18446744073709551616));
    let (s2, m2) = sign_abs(18446744073709551615);
    assert(!s2 & (m2 == 18446744073709551615));
}

#[test(should_fail)]
fn test_sign_abs_rejects_2_pow_64() {
    let _ = sign_abs(18446744073709551616);
}

#[test(should_fail)]
fn test_sign_abs_rejects_below_minus_2_pow_64() {
    let _ = sign_abs(-18446744073709551617);
}

#[test]
fn test_qsign() {
    // np.sign on floats returns -1.0 / 0.0 / 1.0 (Q32), on integers -1/0/1.
    assert(qsign(-q_from_int(5)) == -Q);
    assert(qsign(0) == 0);
    assert(qsign(3) == Q);
    assert(qsign_i(-7) == -1);
    assert(qsign_i(0) == 0);
    assert(qsign_i(7) == 1);
}

#[test]
fn test_qlt() {
    assert(qlt(-q_from_int(3), -q_from_int(2)));
    assert(qlt(-q_from_int(2), q_from_int(0)));
    assert(qlt(q_from_int(1), q_from_int(2)));
    assert(!qlt(q_from_int(5), q_from_int(5)));
    assert(!qlt(0, 0));
    assert(qle(5, 5) & qge(5, 5) & qgt(6, 5) & !qgt(5, 5));
    // The ends of the signed 64-bit window.
    let min = -9223372036854775808;
    let max = 9223372036854775807;
    assert(qlt(min, max) & !qlt(max, min) & qlt(min, 0) & qlt(-1, max));
}

#[test(should_fail)]
fn test_qlt_rejects_a_difference_of_2_pow_64() {
    let _ = qlt(18446744073709551616, 0);
}

#[test]
fn test_qlt_bits() {
    assert(qlt_bits::<3>(-4, 3) & !qlt_bits::<3>(3, -4) & !qlt_bits::<3>(2, 2));
}

#[test(should_fail)]
fn test_qlt_bits_rejects_a_difference_beyond_its_width() {
    // 5 - (-4) = 9 needs 4 bits.
    let _ = qlt_bits::<3>(5, -4);
}

#[test(should_fail)]
fn test_lt_narrow_rejects_a_wrong_answer() {
    constrain_lt::<3>(2, 1, true);
}

#[test(should_fail)]
fn test_lt_narrow_rejects_a_missed_less() {
    constrain_lt::<3>(1, 2, false);
}

#[test(should_fail)]
fn test_lt_narrow_rejects_equal_claimed_less() {
    constrain_lt::<3>(2, 2, true);
}

#[test]
fn test_qlt_bits_accepts_the_edges_of_its_width() {
    // a - b = -2^3 and 2^3 - 1, the ends of [-2^W, 2^W).
    assert(qlt_bits::<3>(-4, 4) & !qlt_bits::<3>(4, -3));
}

#[test(should_fail)]
fn test_qlt_bits_rejects_a_difference_of_2_pow_w() {
    let _ = qlt_bits::<3>(4, -4);
}

#[test]
fn test_q_floor_div_signs() {
    // Python floor division: -7 // 2 == -4, 7 // -2 == -4, -7 // -2 == 3.
    assert(q_floor_div(7, 3) == 2);
    assert(q_floor_div(-7, 3) == -3);
    assert(q_floor_div(6, 3) == 2);
    assert(q_floor_div(-6, 3) == -2);
    assert(q_floor_div(7, -2) == -4);
    assert(q_floor_div(-7, -2) == 3);
    assert(q_floor_div(6, -3) == -2);
    assert(q_floor_div(5, 0) == 0);
}

#[test]
fn test_q_mod_signs() {
    // Python: the remainder takes the divisor's sign.
    assert(q_mod(7, 3) == 1);
    assert(q_mod(-7, 3) == 2);
    assert(q_mod(6, 3) == 0);
    assert(q_mod(7, -2) == -1);
    assert(q_mod(-7, -2) == -1);
    assert(q_mod(5, 0) == 0);
}

#[test]
fn test_euclid_divmod_signs() {
    // Euclidean: 0 <= r < |d| (fixed-point `/`).
    let (q1, r1) = euclid_divmod(-7, 2);
    assert((q1 == -4) & (r1 == 1));
    let (q2, r2) = euclid_divmod(7, -2);
    assert((q2 == -3) & (r2 == 1));
    let (q3, r3) = euclid_divmod(-7, -2);
    assert((q3 == 4) & (r3 == 1));
    let (q4, r4) = euclid_divmod(5, 0);
    assert((q4 == 0) & (r4 == 0));
}

#[test]
fn test_field_div_or_zero() {
    assert(field_div_or_zero(21, 3) == 7);
    assert(field_div_or_zero(21, 0) == 0);
}

#[test]
fn test_divmod_by_positive() {
    let (q, r) = divmod_by_positive::<2>(-7, 3);
    assert((q == -3) & (r == 2));
    let (q2, r2) = divmod_by_positive::<1>(5, 2);
    assert((q2 == 2) & (r2 == 1));
}

#[test]
fn test_divmod_by_pow2() {
    let (q, r) = divmod_by_pow2::<3>(-17);
    assert((q == -3) & (r == 7));
    let (q2, r2) = divmod_by_pow2::<1>(9);
    assert((q2 == 4) & (r2 == 1));
}

#[test(should_fail)]
fn test_divmod_by_pow2_rejects_next_quotient() {
    // -17 = -2 * 8 - 1: the remainder -1 is not a 3-bit value.
    constrain_divmod_by_pow2::<3>(-17, -2, -1);
}

#[test(should_fail)]
fn test_divmod_by_pow2_rejects_field_quotient() {
    // 17 = q * 8 + 0 with q = 17 * 8^-1 mod p: a huge field element.
    let q: Field = 17 / 8;
    constrain_divmod_by_pow2::<3>(17, q, 0);
}

#[test]
fn test_q_floor_div_bounded() {
    assert(q_floor_div_bounded::<8>(23, 5) == 4);
}

#[test]
fn test_q_div_q_signs() {
    assert(q_div_q(3 * Q) == 3);
    assert(q_div_q(3 * Q + 5) == 3);
    assert(q_div_q(-(3 * Q)) == -3);
    assert(q_div_q(-(3 * Q) - 5) == -3);
    assert(q_div_q(-5) == 0);
}

#[test]
fn test_qsqrt_is_the_floor_root() {
    // kernel::fp_sqrt: floor(sqrt(x * 2^32)); a negative input gives 0.
    assert(qsqrt(0) == 0);
    assert(qsqrt(q_from_int(4)) == q_from_int(2));
    // sqrt(2) * 2^32 = 6074000999.95 -> 6074000999.
    assert(qsqrt(q_from_int(2)) == 6074000999);
    assert(qsqrt(10737418240) == 6790939565); // sqrt(2.5)
    assert(qsqrt(1) == 65536);
    assert(qsqrt(4) == 131072); // Q32(1e-9) = 4
    assert(qsqrt(-Q) == 0);
    // The largest accepted input, raw 2^64 - 1.
    assert(qsqrt(18446744073709551615) == 281474976710655);
}

#[test(should_fail)]
fn test_qsqrt_rejects_values_outside_the_window() {
    let _ = qsqrt(18446744073709551616); // raw 2^64
}

#[test]
fn test_qsin_qcos_match_the_kernel() {
    // (x, fp_sin(x), fp_cos(x)) from kernel::fp_sin / fp_cos, bit for bit.
    let cases: [(Field, Field, Field); 9] = [
        (4294967296, 3614090346, 2320580746), // 1
        (1189705940992, 2207631691, 3684169691), // 277
        (1391569403904, -1735446858, -3928736213), // 324
        (-1189705940992, -2207631691, 3684169690), // -277
        (2147483648, 2059117017, 3769188388), // 0.5
        (12884901888, 606105810, -4251985386), // 3
        (4294967296000, 3551420484, 2415399879), // 1000
        (0, 0, 4294967285), // 0
        (-7516192768, -4226187466, -765560969), // -1.75
    ];
    for i in 0..9 {
        let (x, want_sin, want_cos) = cases[i];
        assert(qsin(x) == want_sin);
        assert(qcos(x) == want_cos);
    }
}

#[test(should_fail)]
fn test_qsin_rejects_values_outside_the_window() {
    let _ = qsin(18446744073709551616); // raw 2^64
}

#[test]
fn test_qexp_qlog() {
    // e = 11674931555 / 2^32; exp(-1) = 1580030169 / 2^32.
    assert_close(qexp(Q), 11674931555, 64);
    assert_close(qexp(-Q), 1580030169, 64);
    assert_close(qexp(0), Q, 8);
    // round(ln(X / 2^32) * 2^32) for the Q32 inputs X of e, 0.5, 3, 0.001
    // and 12345.678.
    assert_close(qlog(11674931555), Q, 8);
    assert_close(qlog(Q / 2), -2977044472, 8);
    assert_close(qlog(q_from_int(3)), 4718503851, 8);
    assert_close(qlog(4294967), -29668583308, 8);
    assert_close(qlog(53024283256947), 40463150269, 8);
    assert(qlog(Q) == 0);
}

#[test]
fn test_qsinh_qcosh() {
    // round(sinh(1) * 2^32), round(cosh(1) * 2^32).
    assert_close(qsinh(Q), 5047450693, 64);
    assert_close(qcosh(Q), 6627480862, 64);
}

#[test]
fn test_qpow_small() {
    // 2.0 ^ 3.0 = 8 through log/exp.
    assert_close(qpow(q_from_int(2), q_from_int(3)), q_from_int(8), 512);
    assert(qpow_int_q32(q_from_int(2), 3) == q_from_int(8));
    assert(qpow_int_q32(-q_from_int(2), 2) == q_from_int(4));
}

#[test(should_fail_with = "qpow_int_q32: exponent must be below 2^32")]
fn test_qpow_int_large_exponent_fails() {
    let _ = qpow_int_q32(Q, 4294967296);
}

#[test]
fn test_qarctan_against_libm() {
    // (t, round(atan(t) * 2^32)); the polynomial's error is 1.8e-9 (8 ulps).
    let cases: [(Field, Field); 11] = [
        (-4294967296, -3373259426),
        (-3221225472, -2763816217),
        (-2147483648, -1991351318),
        (-1073741824, -1052175346),
        (-429496730, -428073603),
        (0, 0),
        (429496730, 428073603),
        (1073741824, 1052175346),
        (2147483648, 1991351318),
        (3221225472, 2763816217),
        (4294967296, 3373259426),
    ];
    for i in 0..11 {
        let (t, want) = cases[i];
        assert_close(qarctan(t), want, 16);
    }
}

#[test]
fn test_qarcsin_matches_the_kernel() {
    // (x, kernel::fp_arcsin(x)) bit for bit; outside [-1, 1] the result is 0.
    let cases: [(Field, Field); 11] = [
        (-4294967296, -6746518836),
        (-3865470566, -4809373442),
        (-2147483648, -2248839616),
        (-429496730, -430215800),
        (0, 0),
        (429496730, 430215796),
        (2147483648, 2248839612),
        (2576980378, 2763816210),
        (3865470566, 4809373438),
        (4294967296, 6746518836),
        (6442450944, 0),
    ];
    for i in 0..11 {
        let (x, want) = cases[i];
        assert(qarcsin(x) == want);
    }
    // arccos = pi/2 - arcsin on [-1, 1] (libm values within 32 ulps).
    assert_close(qarccos(2147483648), 4497679235, 32); // acos(0.5)
    assert_close(qarccos(-3865470566), 11555892298, 32); // acos(-0.9)
    assert(qarccos(6442450944) == 0); // acos(1.5): outside the domain
}

#[test]
fn test_qarctan2_against_libm_all_quadrants() {
    // (y, x, round(atan2(y, x) * 2^32)) for (+-1, +-2), (+-2, +-1), the axes.
    let cases: [(Field, Field, Field); 12] = [
        (4294967296, 8589934592, 1991351318),
        (8589934592, 4294967296, 4755167535),
        (-4294967296, 8589934592, -1991351318),
        (-8589934592, 4294967296, -4755167535),
        (-4294967296, -8589934592, -11501686387),
        (-8589934592, -4294967296, -8737870170),
        (4294967296, -8589934592, 11501686387),
        (8589934592, -4294967296, 8737870170),
        (0, 4294967296, 0),
        (4294967296, 0, 6746518852),
        (0, -4294967296, 13493037705),
        (-4294967296, 0, -6746518852),
    ];
    for i in 0..12 {
        let (y, x, want) = cases[i];
        assert_close(qarctan2(y, x), want, 32);
    }
    assert(qarctan2(0, 0) == 0);
}

#[test]
fn test_bitwise_matches_twos_complement() {
    // (a, b, a & b, a | b, a ^ b), as Python / i64.
    let cases: [(Field, Field, Field, Field, Field); 7] = [
        (12, 10, 8, 14, 6),
        (-12, 10, 0, -2, -2),
        (-12, -10, -12, -10, 2),
        (0, -1, 0, -1, -1),
        (-9223372036854775808, -1, -9223372036854775808, -1, 9223372036854775807),
        (9223372036854775807, -9223372036854775808, 0, -1, -1),
        (5, 5, 5, 5, 0),
    ];
    for i in 0..7 {
        let (a, b, and, or, xor) = cases[i];
        assert(bitand(a, b) == and);
        assert(bitor(a, b) == or);
        assert(bitxor(a, b) == xor);
    }
    // Low bits of the two's complement: -3 = ...11111101.
    let bits = low_bits::<1>(-3);
    assert(!bits[1] & bits[0] & bits[2] & bits[7]);
    let bits2 = low_bits::<2>(1029); // 0b100_0000_0101
    assert(bits2[0] & !bits2[1] & bits2[2] & bits2[10] & !bits2[11]);
}

#[test(should_fail)]
fn test_low_bits_rejects_values_outside_the_window() {
    let _ = low_bits::<1>(9223372036854775808);
}

#[test(should_fail)]
fn test_bitand_rejects_values_outside_the_window() {
    let _ = bitand(-9223372036854775809, 1);
}

#[test]
fn test_array_index() {
    let a = [10, 11, 12, 13, 14, 15];
    assert(a[array_index::<8>(0, 5)] == 10);
    assert(a[array_index::<8>(5, 5)] == 15);
}

#[test]
fn test_array_index_28() {
    let a = [10, 11, 12, 13, 14, 15];
    assert(a[array_index::<28>(4, 5)] == 14);
}

#[test(should_fail)]
fn test_array_index_28_rejects_minus_one() {
    let _ = array_index::<28>(-1, 5);
}

#[test(should_fail)]
fn test_array_index_rejects_minus_one() {
    let _ = array_index::<8>(-1, 5);
}

#[test(should_fail)]
fn test_array_index_rejects_the_length() {
    let _ = array_index::<8>(6, 5);
}

#[test(should_fail)]
fn test_array_index_rejects_an_odd_half() {
    // 2 * ((p + 1) / 2) = 1: passes the first check, fails the second.
    let _ = array_index::<8>(1 / 2, 5);
}

#[test]
fn test_checked_casts() {
    assert(checked_u8(255) == 255);
    assert(checked_u32(4294967295) == 4294967295);
}

#[test(should_fail)]
fn test_checked_u8_rejects_256() {
    let _ = checked_u8(256);
}

#[test(should_fail)]
fn test_checked_u32_rejects_negative() {
    // A truncating `as u32` would keep the low bits of p - 1.
    let _ = checked_u32(-1);
}

// ---------------------------------------------------------------------------
// Tests: malicious hint values must be rejected
// ---------------------------------------------------------------------------
//
// Each test calls a constraint function with a value a cheating prover could
// return from the hint instead of the honest one.

#[test]
fn test_lt_accepts_honest() {
    constrain_lt::<64>(3, 5, true);
    constrain_lt::<64>(5, 3, false);
    constrain_lt::<64>(5, 5, false);
    constrain_lt::<64>(-5, 3, true);
    constrain_lt::<3>(-4, 3, true);
    constrain_lt::<3>(3, -4, false);
}

#[test(should_fail)]
fn test_lt_rejects_false_for_less() {
    constrain_lt::<64>(3, 5, false);
}

#[test(should_fail)]
fn test_lt_rejects_true_for_greater() {
    constrain_lt::<64>(5, 3, true);
}

#[test(should_fail)]
fn test_lt_rejects_true_for_equal() {
    constrain_lt::<64>(5, 5, true);
}

#[test(should_fail)]
fn test_lt_rejects_true_across_zero() {
    // -1 < 0 is false for (0, -1).
    constrain_lt::<64>(0, -1, true);
}

#[test(should_fail)]
fn test_sign_abs_rejects_negative_zero() {
    let _ = constrain_sign_abs(0, true);
}

#[test(should_fail)]
fn test_sign_abs_rejects_wrong_sign_of_negative() {
    // -5 claimed non-negative: magnitude p - 5.
    let _ = constrain_sign_abs(-5, false);
}

#[test(should_fail)]
fn test_sign_abs_rejects_wrong_sign_of_positive() {
    // 5 claimed negative: magnitude p - 5.
    let _ = constrain_sign_abs(5, true);
}

#[test(should_fail)]
fn test_divmod_rejects_field_inverse_quotient() {
    // 7 = q*3 + 0 with q = 7 * 3^-1 mod p: a huge field element.
    let q: Field = 7 / 3;
    constrain_divmod(7, 3, q, 0, 3, false);
}

#[test(should_fail)]
fn test_divmod_rejects_other_remainder() {
    // 7 % 3 claimed as 2 needs q = 5 * 3^-1 mod p.
    let q: Field = 5 / 3;
    constrain_divmod(7, 3, q, 2, 3, false);
}

#[test(should_fail)]
fn test_divmod_rejects_remainder_too_large() {
    constrain_divmod(7, 3, 1, 4, 3, false);
}

#[test(should_fail)]
fn test_divmod_rejects_remainder_equal_to_divisor() {
    // 6 = 1*3 + 3: the remainder equals the divisor.
    constrain_divmod(6, 3, 1, 3, 3, false);
}

#[test(should_fail)]
fn test_divmod_rejects_remainder_of_the_wrong_sign() {
    // Euclid needs r >= 0: 7 = 3*3 - 2 is rejected ...
    constrain_divmod(7, 3, 3, -2, 3, false);
}

#[test(should_fail)]
fn test_floor_divmod_rejects_euclidean_answer() {
    // ... and Python floor with d < 0 needs r <= 0: 7 = -2*-3 + 1 is rejected.
    constrain_divmod(7, -2, -3, 1, 2, true);
}

#[test(should_fail)]
fn test_divmod_rejects_zero_divisor() {
    constrain_divmod(7, 0, 0, 7, 0, false);
}

#[test(should_fail)]
fn test_divmod_rejects_quotient_outside_the_window() {
    // 2^64 = 2^63 * 2 + 0: the honest quotient 2^63 leaves the window.
    constrain_divmod(18446744073709551616, 2, 9223372036854775808, 0, 2, false);
}

#[test]
fn test_divmod_accepts_honest() {
    constrain_divmod(-7, 3, -3, 2, 3, false);
    constrain_divmod(7, -2, -4, -1, 2, true);
}

#[test(should_fail)]
fn test_divmod_by_positive_rejects_field_inverse_quotient() {
    let q: Field = 7 / 3;
    constrain_divmod_by_positive::<2>(7, 3, q, 0);
}

#[test(should_fail)]
fn test_floor_div_bounded_rejects_field_inverse_quotient() {
    let q: Field = 7 / 3;
    constrain_floor_div_bounded::<8>(7, 3, q, 0);
}

#[test(should_fail)]
fn test_trunc_div_q_rejects_field_inverse_quotient() {
    // |v| = 3Q + 5 claimed as q = |v| / Q (field inverse), r = 0.
    let mag: Field = 3 * Q + 5;
    constrain_trunc_div_q(mag, mag / Q, 0);
}

#[test(should_fail)]
fn test_floor_div_q_rejects_off_by_one() {
    // floor((3Q + 5) / Q) = 3; claim 4 with remainder 5 - Q.
    constrain_floor_div_q(3 * Q + 5, 4, 5 - Q);
}

#[test(should_fail)]
fn test_floor_div_q_rejects_field_inverse_quotient() {
    let v: Field = 3 * Q + 5;
    constrain_floor_div_q(v, v / Q, 0);
}

#[test(should_fail)]
fn test_isqrt_rejects_rounded_up_root() {
    // n = 2^65: floor(sqrt) = 6074000999, the rounded root 6074001000 is rejected.
    constrain_isqrt_floor(2 * Q * Q, 6074001000);
}

#[test(should_fail)]
fn test_isqrt_rejects_rounded_down_root() {
    constrain_isqrt_floor(2 * Q * Q, 6074000998);
}

#[test(should_fail)]
fn test_isqrt_rejects_negative_root() {
    constrain_isqrt_floor(4 * Q * Q, -2 * Q);
}

#[test]
fn test_isqrt_accepts_honest() {
    constrain_isqrt_floor(2 * Q * Q, 6074000999);
    constrain_isqrt_floor(0, 0);
}

#[test(should_fail)]
fn test_field_div_rejects_nonzero_inverse_of_zero() {
    // For d == 0 the hint must be 0; any other value is rejected.
    constrain_inverse_or_zero(0, 5);
}

#[test(should_fail)]
fn test_field_div_rejects_wrong_inverse() {
    constrain_inverse_or_zero(3, 2);
}

// ---------------------------------------------------------------------------
// Tests: sorting hints (same lanes as kernel::hint_lane)
// ---------------------------------------------------------------------------

#[test]
fn test_argsort_hint_is_stable_and_signed() {
    // -1 < 0 < 2 = 2 < 5; the two 2s keep their original order (2, 4).
    let a: [Field; 6] = [5, -1, 2, 0, 2, 0 - 7];
    assert(argsort_hint(a) == [5, 1, 3, 2, 4, 0]);
    assert(sorted_hint(a) == [-7, -1, 0, 2, 2, 5]);
}

#[test]
fn test_hint_lanes_at_runtime_rank() {
    let a: [Field; 5] = [30, 10, 20, 10, -40];
    assert(argsort_hint_at(a, 0) == 4);
    assert(argsort_hint_at(a, 2) == 3);
    assert(sorted_hint_at(a, 4) == 30);
    // Out of range: lane 0 by definition (the IR's pins reject it).
    assert(sorted_hint_at(a, 5) == 0);
    assert(argsort_hint_at(a, -1) == 0);
}

#[test]
fn test_argsort_hint_odd_lengths() {
    let a: [Field; 7] = [3, 3, 1, 3, 0, 1, 3];
    assert(argsort_hint(a) == [4, 2, 5, 0, 1, 3, 6]);
    let one: [Field; 1] = [9];
    assert(argsort_hint(one) == [0]);
}

#[test]
fn test_masked_sorted_hint_sorts_the_active_elements() {
    let a: [Field; 5] = [30, 10, 20, 10, -40];
    let active: [Field; 5] = [1, 0, 1, 0, 1];
    assert(masked_sorted_hint(a, active) == [-40, 20, 30, 0, 0]);
    assert(masked_sorted_hint_at(a, active, 1) == 20);
    assert(masked_sorted_hint_at(a, active, 3) == 0);
    let none: [Field; 5] = [0; 5];
    assert(masked_sorted_hint(a, none) == [0; 5]);
}

#[test]
fn test_abs_int64() {
    assert(abs_int64(-5) == 5);
    assert(abs_int64(7) == 7);
    assert(abs_int64(0) == 0);
    assert(abs_int64(-9223372036854775807) == 9223372036854775807);
}

#[test]
fn test_abs_int64_accepts_both_edges() {
    assert(abs_int64(9223372036854775807) == 9223372036854775807);
    assert(abs_int64(-9223372036854775807) == 9223372036854775807);
}

#[test(should_fail)]
fn test_abs_int64_rejects_i64_min() {
    let _ = abs_int64(-9223372036854775808);
}

#[test(should_fail)]
fn test_abs_int64_rejects_2_pow_63() {
    let _ = abs_int64(9223372036854775808);
}
