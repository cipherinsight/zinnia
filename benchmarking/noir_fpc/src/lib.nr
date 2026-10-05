// noir_fpc -- Q32 fixed-point library shared by Zinnia's Noir backend and the
// handwritten Noir baselines.
//
// Encoding. A float x is the field element round(x * 2^32) ("Q32"); an
// integer n is the field element n. A negative value v is stored as p - |v|
// (p is the BN254 scalar modulus, about 2^253.6). Helpers interpret a field
// element as the signed integer of smallest magnitude it represents.
//
// Value range. Every helper that needs an order (comparisons, division,
// rescaling, abs, sign) accepts operands whose magnitude is below 2^126 and
// is unsatisfiable (the proof fails) otherwise. 2^126 is far above any
// legitimate Q32 intermediate and far below p/2, which is what makes the
// uniqueness arguments below work.
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
// the low N bits. Use `checked_u8` .. `checked_u128` to convert a Field to an
// unsigned integer type; after the range check Noir elides the truncation, so
// the checked form is also cheaper than the bare cast.

pub global Q: Field = 4294967296;
pub global Q_U128: u128 = 4294967296;
pub global HALF_Q: u128 = 2147483648;

// 2^126: offset that maps the signed range [-2^126, 2^126) onto [0, 2^127).
global SIGNED_OFFSET: Field = 85070591730234615865843651857942052864;

// pi, pi/2, pi/4 in Q32 (kernel::pi_q, half_pi_q, quarter_pi_q).
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

/// Constrains x to the signed range [-2^126, 2^126): x + 2^126 must be a
/// 127-bit value. A value outside the range wraps to a field element of about
/// 254 bits and fails the check.
pub fn assert_signed_range(x: Field) {
    (x + SIGNED_OFFSET).assert_max_bit_size::<127>();
}

// ---------------------------------------------------------------------------
// Sign / magnitude
// ---------------------------------------------------------------------------

// Hint: (negative, |v|) for the signed reading of v. v is negative when its
// additive inverse is the smaller field element of the two.
unconstrained fn hint_sign_magnitude(v: Field) -> (bool, Field) {
    if (-v).lt(v) {
        (true, -v)
    } else {
        (false, v)
    }
}

/// Constraints for `to_signed`. Uniqueness: `mag < 2^126` and `v == +-mag`
/// leave one candidate per sign; both signs can only hold together when
/// mag + mag' == 0 mod p, i.e. mag == mag' == 0, and the last check rules out
/// a negative zero. So (negative, mag) is a function of v.
fn constrain_sign_magnitude(v: Field, negative: bool, mag: Field) {
    mag.assert_max_bit_size::<126>();
    let signed: Field = if negative { -mag } else { mag };
    assert(v == signed, "to_signed: sign/magnitude decomposition mismatch");
    assert(!negative | (mag != 0), "to_signed: negative zero");
}

/// Signed reading of v as (negative, |v|). Fails when |v| >= 2^126.
pub fn to_signed(v: Field) -> (bool, u128) {
    // Safety: constrain_sign_magnitude pins the hint uniquely (see there).
    let (negative, mag) = unsafe { hint_sign_magnitude(v) };
    constrain_sign_magnitude(v, negative, mag);
    (negative, mag as u128)
}

pub fn is_negative(v: Field) -> bool {
    let (s, _m) = to_signed(v);
    s
}

pub fn qabs(v: Field) -> Field {
    let (_s, m) = to_signed(v);
    m as Field
}

pub fn qabs_i(v: Field) -> Field {
    qabs(v)
}

/// Integer sign: -1, 0 or 1.
pub fn qsign_i(v: Field) -> Field {
    let (s, m) = to_signed(v);
    if m == 0 {
        0
    } else if s {
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
// Comparisons (signed, for values with magnitude < 2^126)
// ---------------------------------------------------------------------------

pub fn qlt(a: Field, b: Field) -> bool {
    let (sa, ma) = to_signed(a);
    let (sb, mb) = to_signed(b);
    qlt_signed(sa, ma, sb, mb)
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

// Pre-projected forms: callers that compare the same operand many times
// project it once with `to_signed` and feed the pairs here. Correct for any
// pair produced by `to_signed` (zero is never negative there).
pub fn qlt_signed(sa: bool, ma: u128, sb: bool, mb: u128) -> bool {
    if sa {
        if !sb {
            (ma != 0) | (mb != 0)
        } else {
            ma > mb
        }
    } else if sb {
        false
    } else {
        ma < mb
    }
}
pub fn qle_signed(sa: bool, ma: u128, sb: bool, mb: u128) -> bool {
    !qlt_signed(sb, mb, sa, ma)
}
pub fn qgt_signed(sa: bool, ma: u128, sb: bool, mb: u128) -> bool {
    qlt_signed(sb, mb, sa, ma)
}
pub fn qge_signed(sa: bool, ma: u128, sb: bool, mb: u128) -> bool {
    !qlt_signed(sa, ma, sb, mb)
}

pub fn qadd(a: Field, b: Field) -> Field {
    a + b
}
pub fn qsub(a: Field, b: Field) -> Field {
    a - b
}
pub fn qneg(a: Field) -> Field {
    -a
}

// ---------------------------------------------------------------------------
// Integer division
// ---------------------------------------------------------------------------
//
// The conventions of kernel::div_rem_or_zero: `n == q*d + r` with |r| < |d|,
//   floor  (Python `//`, `%`; FloorDivI, ModI): r is 0 or has the sign of d;
//   Euclid (fixed-point `/`, DivF):              0 <= r < |d|;
//   trunc  (Python `int()`, IntCast):            r is 0 or has the sign of n;
// and a zero divisor gives (0, 0).

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
/// `d_mag` and a remainder whose sign the caller fixes (`remainder_negative`).
/// Uniqueness: |q| <= 2^126, |d| < 2^126 and |r| < |d| bound the integer
/// q*d + r by 2^252 + 2^126 < (p - 1) / 2, so `q*d + r == n (mod p)` forces
/// q*d + r to equal the signed reading of n, and integer division with a
/// remainder of fixed sign and |r| < |d| has exactly one (q, r). A zero
/// divisor fails `|r| < |d|`.
fn constrain_divmod(n: Field, d: Field, q: Field, r: Field, d_mag: u128, remainder_negative: bool) {
    assert_signed_range(q);
    let r_mag = if remainder_negative { -r } else { r };
    r_mag.assert_max_bit_size::<126>();
    assert((r_mag as u128) < d_mag, "divmod: |remainder| >= |divisor| (or divisor is zero)");
    assert(q * d + r == n, "divmod: reconstruction failed");
}

/// Python `(n // d, n % d)` for d != 0: the remainder takes d's sign.
fn floor_divmod_nonzero(n: Field, d: Field) -> (Field, Field) {
    let (d_negative, d_mag) = to_signed(d);
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, d_negative) };
    constrain_divmod(n, d, q, r, d_mag, d_negative);
    (q, r)
}

/// Euclidean `(q, r)`, 0 <= r < |d|, for d != 0.
fn euclid_divmod_nonzero(n: Field, d: Field) -> (Field, Field) {
    let (_d_negative, d_mag) = to_signed(d);
    // Safety: constrain_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod(n, d, false) };
    constrain_divmod(n, d, q, r, d_mag, false);
    (q, r)
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
/// |q| <= 2^126 and BITS <= 120 the integer q*d + r stays below p/2 in
/// magnitude, so the argument of `constrain_divmod` applies.
fn constrain_divmod_by_positive<let BITS: u32>(n: Field, d: Field, q: Field, r: Field) {
    assert(BITS <= 120, "divmod_by_positive: BITS must be <= 120");
    r.assert_max_bit_size::<BITS>();
    (d - 1 - r).assert_max_bit_size::<BITS>();
    assert_signed_range(q);
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

/// Constraints for `q_div_q`: q < 2^94, r < 2^32 and q*Q + r == |v| (no wrap:
/// q*Q + r < 2^127), so (q, r) = (floor(|v| / Q), |v| mod Q) is unique.
fn constrain_trunc_div_q(mag: Field, q: Field, r: Field) {
    q.assert_max_bit_size::<94>();
    r.assert_max_bit_size::<32>();
    assert(q * Q + r == mag, "q_div_q: reconstruction failed");
}

/// Float -> int cast: trunc(v / Q), rounding toward zero (Python `int()`),
/// i.e. sign(v) * floor(|v| / Q).
pub fn q_div_q(v: Field) -> Field {
    let (negative, mag) = to_signed(v);
    let mag_f = mag as Field;
    // Safety: constrain_trunc_div_q pins (q, r) uniquely.
    let (q, r) = unsafe { hint_divmod_q(mag_f) };
    constrain_trunc_div_q(mag_f, q, r);
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

/// Constraints for `qscale_down_raw`: 0 <= r < 2^32 = Q and |q| <= 2^126 keep
/// |q*Q + r| below 2^159, so `q*Q + r == v` holds over the integers and
/// q = floor(v / Q) is unique.
fn constrain_floor_div_q(v: Field, q: Field, r: Field) {
    r.assert_max_bit_size::<32>();
    assert_signed_range(q);
    assert(q * Q + r == v, "qscale_down_raw: reconstruction failed");
}

/// floor(v / Q) for a raw (Q^2-scaled) product or sum of products.
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

pub fn q_from_int(x: u64) -> Field {
    (x as Field) * Q
}

pub fn q_from_neg_int(x: u64) -> Field {
    -((x as Field) * Q)
}

pub fn q_eq_with_tol(a: Field, b: Field, tol: u128) {
    let (_s, m) = to_signed(a - b);
    assert(m <= tol, "q_eq_with_tol: |a - b| > tol");
}

pub fn q_close(a: Field, b: Field, tol: u128) -> bool {
    let (_s, m) = to_signed(a - b);
    m <= tol
}

// Default tolerance: 0.001 * Q ~= 4_294_967.
pub global TOL_DEFAULT: u128 = 4294967;

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

/// Constraints for `isqrt_floor` (n < 2^(2 Y_BITS), D_BITS = Y_BITS + 1):
/// y < 2^Y_BITS and both d = n - y^2 and 2y - d are D_BITS-bit values. As
/// every term is below 2^(2 Y_BITS + 1), a negative difference cannot pass,
/// so y^2 <= n <= y^2 + 2y < (y + 1)^2: y is the unique floor root.
fn constrain_isqrt_floor<let Y_BITS: u32, let D_BITS: u32>(n: Field, y: Field) {
    y.assert_max_bit_size::<Y_BITS>();
    let d = n - y * y;
    d.assert_max_bit_size::<D_BITS>();
    (2 * y - d).assert_max_bit_size::<D_BITS>();
}

/// floor(sqrt(n)) for 0 <= n < 2^(2 Y_BITS).
fn isqrt_floor<let Y_BITS: u32, let D_BITS: u32>(n: Field) -> Field {
    // Safety: constrain_isqrt_floor pins y uniquely.
    let y = unsafe { hint_isqrt_floor(n) };
    constrain_isqrt_floor::<Y_BITS, D_BITS>(n, y);
    y
}

/// Q32 square root rounded down: floor(sqrt(x * Q)), the largest Q32 value
/// whose square does not exceed x (kernel::fp_sqrt). A negative input gives
/// 0 (the kernel's convention). Fails for x >= 2^32 in real units (raw
/// x >= 2^64), outside the kernel's fixed-point window.
pub fn qsqrt(x: Field) -> Field {
    let (negative, mag) = to_signed(x);
    let m = if negative { 0 } else { mag as Field };
    m.assert_max_bit_size::<64>();
    isqrt_floor::<48, 49>(m * Q)
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

// kernel::sin_taylor_q(32): sin(s)/s as a polynomial in u = s^2, leading first.
global SIN_TAYLOR_Q: [Field; 7] = [
    1,
    -(108 as Field),
    11836,
    -(852176 as Field),
    35791394,
    -(715827883 as Field),
    4294967296,
];

// kernel::cos_taylor_q(32): cos(s) as a polynomial in u = s^2, leading first.
global COS_TAYLOR_Q: [Field; 8] = [
    0,
    9,
    -(1184 as Field),
    106522,
    -(5965232 as Field),
    178956971,
    -(2147483648 as Field),
    4294967296,
];

/// Q32 sine by kernel::fp_sin's algorithm (bit for bit the mock's value):
/// r = |x| mod 2*pi_q (an exact checked division); fold r >= pi_q onto
/// r - pi_q (sign flip); mirror r > pi_q/2 onto pi_q - r; above pi_q/4 use
/// cos(pi_q/2 - r); evaluate the Taylor polynomial in u = s^2 with flooring
/// products. Error against libm: about 1e-9 on [-20, 20]. Fails for raw
/// |x| >= 2^64, outside the kernel's fixed-point window.
pub fn qsin(x: Field) -> Field {
    let (negative, mag) = to_signed(x);
    let m = mag as Field;
    m.assert_max_bit_size::<64>();
    let (_k, r) = divmod_by_positive::<35>(m, TWO_PI_Q);
    // r < 2 pi_q < 2^35, so the conversion never fails; the reduction then
    // runs on u64 values (13493037705 = pi_q, 6746518852 = pi_q / 2,
    // 3373259426 = pi_q / 4).
    let r0 = checked_u64(r);
    let folded = r0 >= 13493037705;
    let r1 = if folded { r0 - 13493037705 } else { r0 };
    let r2 = if r1 > 6746518852 { 13493037705 - r1 } else { r1 };
    let cofunction = r2 > 3373259426;
    let s = (if cofunction { 6746518852 - r2 } else { r2 }) as Field;
    let u = qmul(s, s);
    let p = if cofunction {
        horner_q(u, COS_TAYLOR_Q)
    } else {
        qmul(s, horner_q(u, SIN_TAYLOR_Q))
    };
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
    let (i, f) = divmod_by_positive::<32>(y, Q);
    let two_to_f = horner_q(f, EXP2_COEFS_Q); // in [Q, 2Q]
    let (i_negative, i_mag) = to_signed(i);
    let e = i_mag as Field;
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
    let (negative, mag) = to_signed(x);
    assert(!negative & (mag != 0), "qlog: non-positive input");
    let x_f = mag as Field;
    // Safety: the checks below pin (k, m) uniquely.
    let (k, m) = unsafe { hint_log2_decompose(x_f) };
    (m - Q).assert_max_bit_size::<32>(); // Q <= m < 2Q
    let (k_negative, k_mag) = to_signed(k);
    let e = k_mag as Field;
    e.assert_max_bit_size::<7>();
    let scale = pow2(e);
    // k < 0: x * 2^-k == m exactly; k >= 0: x - m * 2^k in [0, 2^k).
    let lhs = if k_negative { x_f * scale } else { x_f };
    let m_scaled = if k_negative { m } else { m * scale };
    let bound = if k_negative { 1 } else { scale };
    let rem = lhs - m_scaled;
    rem.assert_max_bit_size::<126>();
    (bound - 1 - rem).assert_max_bit_size::<126>();
    let s = qdiv(m - Q, m + Q);
    let series = horner_q(qmul(s, s), ATANH_SERIES_Q);
    let ln_m = 2 * qmul(s, series);
    let (k_ln2, _r) = divmod_by_positive::<32>(k * LN2_Q64 + 2147483648, Q);
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
    let (_negative, mag) = to_signed(x);
    let in_domain = mag <= Q_U128;
    let v = if in_domain { x } else { 0 };
    let v_mag = if in_domain { mag as Field } else { 0 };
    // 2^64 - v^2 <= 2^64, so the root is at most 2^32.
    let root = isqrt_floor::<33, 34>(Q * Q - v_mag * v_mag);
    let (t, _r) = divmod_by_positive::<33>(v * Q, Q + root);
    let half = qarctan(t);
    half + half
}

/// Q32 arccos = pi/2 - arcsin x on [-1, 1]; 0 outside (where the mock's f64
/// acos gives nan, which it encodes as 0).
pub fn qarccos(x: Field) -> Field {
    let (_negative, mag) = to_signed(x);
    if mag <= Q_U128 {
        HALF_PI_Q - qarcsin(x)
    } else {
        0
    }
}

/// Q32 atan2(y, x): the polynomial on min(|x|,|y|) / max(|x|,|y|) in [0, 1],
/// then the octant and quadrant adjustments. atan2(0, 0) = 0.
pub fn qarctan2(y: Field, x: Field) -> Field {
    let (sy, my) = to_signed(y);
    let (sx, mx) = to_signed(x);
    let swap: bool = my > mx;
    let num: Field = if swap { mx as Field } else { my as Field };
    let den: Field = if swap { my as Field } else { mx as Field };
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
    let (q, _r) = divmod_by_positive::<1>(v, 2);
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
// Tests: honest values
// ---------------------------------------------------------------------------

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
    // Negative divisor (the old helper failed here).
    assert(qdiv(q_from_int(10), -q_from_int(4)) == -(q_from_int(2) + Q / 2));
    assert(qdiv(-q_from_int(10), -q_from_int(4)) == q_from_int(2) + Q / 2);
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
fn test_to_signed() {
    let (s, m) = to_signed(-q_from_int(5));
    assert(s & (m == 4294967296 * 5));
    let (s0, m0) = to_signed(0);
    assert(!s0 & (m0 == 0));
    assert(qabs(-q_from_int(5)) == q_from_int(5));
}

#[test(should_fail)]
fn test_to_signed_rejects_large_magnitude() {
    let _ = to_signed(SIGNED_OFFSET);
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
}

#[test(should_fail)]
fn test_qsqrt_rejects_values_outside_the_window() {
    let _ = qsqrt(18446744073709551616); // raw 2^64
}

#[test]
fn test_qsin_qcos_match_the_kernel() {
    // (x, fp_sin(x), fp_cos(x)) from kernel::fp_sin / fp_cos, bit for bit.
    let cases: [(Field, Field, Field); 9] = [
        (4294967296, 3614090360, 2320580733), // 1
        (1189705940992, 2207631680, 3684169708), // 277
        (1391569403904, -1735446857, -3928736219), // 324
        (-1189705940992, -2207631680, 3684169707), // -277
        (2147483648, 2059117009, 3769188403), // 0.5
        (12884901888, 606105819, -4251985396), // 3
        (4294967296000, 3551420499, 2415399866), // 1000
        (0, 0, 4294967296), // 0
        (-7516192768, -4226187461, -765560979), // -1.75
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
    q_eq_with_tol(qexp(Q), 11674931555, 64);
    q_eq_with_tol(qexp(-Q), 1580030169, 64);
    q_eq_with_tol(qexp(0), Q, 8);
    // round(ln(X / 2^32) * 2^32) for the Q32 inputs X of e, 0.5, 3, 0.001
    // and 12345.678.
    q_eq_with_tol(qlog(11674931555), Q, 8);
    q_eq_with_tol(qlog(Q / 2), -2977044472, 8);
    q_eq_with_tol(qlog(q_from_int(3)), 4718503851, 8);
    q_eq_with_tol(qlog(4294967), -29668583308, 8);
    q_eq_with_tol(qlog(53024283256947), 40463150269, 8);
    assert(qlog(Q) == 0);
}

#[test]
fn test_qsinh_qcosh() {
    // round(sinh(1) * 2^32), round(cosh(1) * 2^32).
    q_eq_with_tol(qsinh(Q), 5047450693, 64);
    q_eq_with_tol(qcosh(Q), 6627480862, 64);
}

#[test]
fn test_qpow_small() {
    // 2.0 ^ 3.0 = 8 through log/exp.
    q_eq_with_tol(qpow(q_from_int(2), q_from_int(3)), q_from_int(8), 512);
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
        q_eq_with_tol(qarctan(t), want, 16);
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
    q_eq_with_tol(qarccos(2147483648), 4497679235, 32); // acos(0.5)
    q_eq_with_tol(qarccos(-3865470566), 11555892298, 32); // acos(-0.9)
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
        q_eq_with_tol(qarctan2(y, x), want, 32);
    }
    assert(qarctan2(0, 0) == 0);
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
// Each test calls a constraint function with the values a cheating prover
// would return from the hint. The old helpers accepted the first one of each
// group (no bound on the quotient).

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
fn test_divmod_rejects_remainder_of_the_wrong_sign() {
    // Euclid needs r >= 0: 7 = 3*3 - 2 is rejected ...
    constrain_divmod(7, 3, 3, -2, 3, false);
}

#[test(should_fail)]
fn test_floor_divmod_rejects_euclidean_answer() {
    // ... and Python floor with d < 0 needs r <= 0: 7 = -2*-3 + 1 is rejected.
    constrain_divmod(7, -2, -3, 1, 2, true);
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
fn test_sign_magnitude_rejects_negative_zero() {
    constrain_sign_magnitude(0, true, 0);
}

#[test(should_fail)]
fn test_sign_magnitude_rejects_wrong_sign() {
    // -5 claimed as +(p - 5): magnitude far above 2^126.
    constrain_sign_magnitude(-5, false, -5);
}

#[test(should_fail)]
fn test_isqrt_rejects_rounded_up_root() {
    // n = 2^65: floor(sqrt) = 6074000999, the rounded root 6074001000 is rejected.
    constrain_isqrt_floor::<48, 49>(2 * Q * Q, 6074001000);
}

#[test(should_fail)]
fn test_isqrt_rejects_negative_root() {
    constrain_isqrt_floor::<48, 49>(4 * Q * Q, -2 * Q);
}

#[test]
fn test_isqrt_accepts_honest() {
    constrain_isqrt_floor::<48, 49>(2 * Q * Q, 6074000999);
    constrain_isqrt_floor::<48, 49>(0, 0);
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
