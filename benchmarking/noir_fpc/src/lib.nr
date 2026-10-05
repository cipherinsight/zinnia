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
// Semantics. The arithmetic helpers reproduce the reference semantics of
// Zinnia's mock backend (`src/prove/kernel.rs`): Euclidean integer division,
// truncating float->int casts, flooring Q32 rescales. Transcendental
// functions are polynomial approximations; the mock evaluates them in f64.
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

// Constants for the trigonometric helpers. The range reduction runs in Q64 so
// that the representation error of pi stays negligible for large arguments.
global PI_Q: Field = 13493037705; // round(pi * 2^32)
global HALF_PI_Q: Field = 6746518852; // round(pi / 2 * 2^32)
global PI_Q64: Field = 57952155664616982739; // round(pi * 2^64), 66 bits
global HALF_PI_Q64: Field = 28976077832308491370; // round(pi / 2 * 2^64)

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
// Integer division (Euclidean: n == q*d + r with 0 <= r < |d|)
// ---------------------------------------------------------------------------
//
// Euclidean division is the IR's FloorDivI/ModI semantics (`div_euclid`,
// `rem_euclid` in src/prove/kernel.rs). It equals Python's floor division
// whenever d > 0; for d < 0 Python's remainder takes the sign of d instead.

// Hint: Euclidean (q, r) of the signed readings of n and d; (0, 0) when d == 0
// (the constraints then fail).
unconstrained fn hint_euclid_divmod(n: Field, d: Field) -> (Field, Field) {
    let (n_neg, n_mag) = hint_sign_magnitude(n);
    let (d_neg, d_mag) = hint_sign_magnitude(d);
    let nm = n_mag as u128;
    let dm = d_mag as u128;
    if dm == 0 {
        (0, 0)
    } else {
        let qm = nm / dm;
        let rm = nm % dm;
        // n < 0 with a non-zero remainder rounds the magnitude up so that the
        // remainder becomes |d| - rm >= 0.
        let round_up = n_neg & (rm != 0);
        let q_mag: Field = if round_up { (qm + 1) as Field } else { qm as Field };
        let r: Field = if round_up { (dm - rm) as Field } else { rm as Field };
        let q: Field = if n_neg != d_neg { -q_mag } else { q_mag };
        (q, r)
    }
}

/// Constraints for `euclid_divmod`. Uniqueness: |q| <= 2^126, |d| < 2^126 and
/// 0 <= r < |d| bound the integer q*d + r by 2^252 + 2^126 < (p - 1) / 2, so
/// `q*d + r == n (mod p)` forces q*d + r to equal the signed reading of n as
/// an integer, and integer Euclidean division has exactly one (q, r). A zero
/// divisor fails `r < |d|`.
fn constrain_euclid_divmod(n: Field, d: Field, q: Field, r: Field) {
    let (_d_neg, d_mag) = to_signed(d);
    assert_signed_range(q);
    r.assert_max_bit_size::<126>();
    assert((r as u128) < d_mag, "euclid_divmod: remainder >= |divisor| (or divisor is zero)");
    assert(q * d + r == n, "euclid_divmod: reconstruction failed");
}

/// Euclidean (q, r) with n == q*d + r and 0 <= r < |d|, for any signs.
pub fn euclid_divmod(n: Field, d: Field) -> (Field, Field) {
    // Safety: constrain_euclid_divmod pins (q, r) uniquely.
    let (q, r) = unsafe { hint_euclid_divmod(n, d) };
    constrain_euclid_divmod(n, d, q, r);
    (q, r)
}

/// Constraints for `euclid_divmod_small`. `r` and `d - 1 - r` are BITS-bit
/// values, which gives 0 <= r <= d - 1 (so d >= 1 and d < 2^(BITS+1)); with
/// |q| <= 2^126 and BITS <= 120 the integer q*d + r stays below p/2 in
/// magnitude, so the argument of `constrain_euclid_divmod` applies.
fn constrain_euclid_divmod_small<let BITS: u32>(n: Field, d: Field, q: Field, r: Field) {
    assert(BITS <= 120, "euclid_divmod_small: BITS must be <= 120");
    r.assert_max_bit_size::<BITS>();
    (d - 1 - r).assert_max_bit_size::<BITS>();
    assert_signed_range(q);
    assert(q * d + r == n, "euclid_divmod_small: reconstruction failed");
}

/// Euclidean division by a positive divisor d with d <= 2^BITS (typically a
/// compile-time constant). Same result as `euclid_divmod`, but the remainder
/// check is two BITS-bit range checks instead of a sign projection of d. The
/// constraints themselves enforce 1 <= d, so a wrong BITS can only make the
/// proof fail, never accept a wrong quotient.
pub fn euclid_divmod_small<let BITS: u32>(n: Field, d: Field) -> (Field, Field) {
    // Safety: constrain_euclid_divmod_small pins (q, r) uniquely.
    let (q, r) = unsafe { hint_euclid_divmod(n, d) };
    constrain_euclid_divmod_small::<BITS>(n, d, q, r);
    (q, r)
}

/// Integer `n // d` (Euclidean), any signs.
pub fn q_floor_div(n: Field, d: Field) -> Field {
    let (q, _r) = euclid_divmod(n, d);
    q
}

/// Integer `n % d` (Euclidean, in [0, |d|)), any signs.
pub fn q_mod(n: Field, d: Field) -> Field {
    let (_q, r) = euclid_divmod(n, d);
    r
}

/// Constraints for `q_floor_div_bounded`: q, r and d - r - 1 are BITS-bit
/// values, i.e. q >= 0 and 0 <= r < d. With BITS <= 120 the integer q*d + r is
/// below 2^242 < p, so the reconstruction holds over the integers and the
/// Euclidean (q, r) is unique.
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
    let (q, r) = unsafe { hint_euclid_divmod(n, d) };
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
    let (q, r) = unsafe { hint_euclid_divmod(v, Q) };
    constrain_floor_div_q(v, q, r);
    q
}

/// Q32 product: floor(a * b / Q).
pub fn qmul(a: Field, b: Field) -> Field {
    qscale_down_raw(a * b)
}

/// Q32 quotient: floor(a * Q / b) (Euclidean, as the mock's `div_euclid`), any
/// signs. Fails when b == 0.
pub fn qdiv(a: Field, b: Field) -> Field {
    assert(b != 0, "qdiv: divide by zero");
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

// Hint: round(sqrt(xq)) for 0 <= xq < 2^128.
unconstrained fn hint_isqrt_round(xq: Field) -> Field {
    let target = xq as u128;
    let mut s: u128 = 0;
    if target != 0 {
        // Newton iteration from above converges to floor(sqrt(target)).
        let mut g: u128 = target;
        for _ in 0..130 {
            let next = (g + target / g) / 2;
            if next < g {
                g = next;
            }
        }
        s = g;
        // Round half up: sqrt(target) >= s + 1/2  <=>  target > s^2 + s.
        if target - s * s > s {
            s = s + 1;
        }
    }
    s as Field
}

/// Constraints for `qsqrt` with xq = x * Q: y < 2^64 and
///   y^2 - y + 1 - [y == 0] <= xq <= y^2 + y,
/// both checked as 129-bit range checks on the differences (every term is
/// below 2^129, so a negative difference cannot pass). The integer intervals
/// [y^2 - y + 1, y^2 + y] for y >= 1, and {0} for y == 0, partition the
/// non-negative integers, so y = round(sqrt(xq)) is unique.
fn constrain_isqrt_round(xq: Field, y: Field) {
    y.assert_max_bit_size::<64>();
    let y_is_zero: Field = if y == 0 { 1 } else { 0 };
    (xq - y * y + y - 1 + y_is_zero).assert_max_bit_size::<129>();
    (y * y + y - xq).assert_max_bit_size::<129>();
}

/// Q32 square root rounded to nearest: round(sqrt(x * Q)), i.e. the Q32 value
/// nearest to sqrt(x). Fails for x < 0 and for x >= 2^96 (|x| >= 2^64 in
/// real units).
pub fn qsqrt(x: Field) -> Field {
    let (negative, mag) = to_signed(x);
    assert(!negative, "qsqrt: negative input");
    let mag_f = mag as Field;
    mag_f.assert_max_bit_size::<96>();
    let xq = mag_f * Q;
    // Safety: constrain_isqrt_round pins y uniquely.
    let y = unsafe { hint_isqrt_round(xq) };
    constrain_isqrt_round(xq, y);
    y
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

// sin(x) for x in [0, pi/2] (Q32) via the nested Taylor form
//   sin x = x (1 - x^2/(2*3) (1 - x^2/(4*5) (... (1 - x^2/(14*15)))))
// (terms through x^15; truncation error < 7e-12 on [0, pi/2]). Each level is
// one flooring division of x^2 * t by c * Q, so each level adds at most one
// Q32 ulp and the error is damped by x^2/c < 0.42 per level.
fn sin_first_quadrant(x: Field) -> Field {
    let x2 = qmul(x, x);
    let divisors: [Field; 7] = [210, 156, 110, 72, 42, 20, 6];
    let mut t: Field = Q;
    for i in 0..7 {
        // x2 < 2^34 and t <= Q, so x2 * t < 2^67; c * Q < 2^40.
        let (u, _r) = euclid_divmod_small::<40>(x2 * t, divisors[i] * Q);
        t = Q - u;
    }
    qmul(x, t)
}

// sin(a) for a non-negative angle a in Q64 (a < 2^128), as a Q32 value:
// a = k*pi + r with 0 <= r < pi (exact checked division by the Q64 constant
// pi), sin a = (-1)^k sin r, and sin r = sin(pi - r) folds r into [0, pi/2]
// before it is rounded to Q32.
fn sin_nonneg_q64(a: Field) -> Field {
    let (k, r) = euclid_divmod_small::<66>(a, PI_Q64);
    let (_half, k_odd) = euclid_divmod_small::<1>(k, 2);
    // r < 2^66 (checked above), so checked_u128 never fails here.
    let r_past_half_pi = checked_u128(r) > 28976077832308491370; // HALF_PI_Q64
    let folded = if r_past_half_pi { PI_Q64 - r } else { r };
    let (x, _rem) = euclid_divmod_small::<32>(folded + 2147483648, Q); // round to Q32
    let s = sin_first_quadrant(x);
    if k_odd == 1 {
        -s
    } else {
        s
    }
}

/// Q32 sine for any argument with |x| < 2^64 (real units); sin(-x) = -sin x.
/// Error: a few Q32 ulps (the Taylor levels and the final rounding).
pub fn qsin(x: Field) -> Field {
    let (negative, mag) = to_signed(x);
    let s = sin_nonneg_q64((mag as Field) * Q);
    if negative {
        -s
    } else {
        s
    }
}

/// Q32 cosine: cos x = sin(|x| + pi/2), with the shift done in Q64.
pub fn qcos(x: Field) -> Field {
    let (_negative, mag) = to_signed(x);
    sin_nonneg_q64((mag as Field) * Q + HALF_PI_Q64)
}

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

global LN2_Q: Field = 2977044472; // round(ln 2 * 2^32)
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
    let (i, f) = euclid_divmod_small::<32>(y, Q);
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

// log2 on [2, 4), degree 14, quantized to Q32.
global LOG2_COEFS_Q: [Field; 15] = [
    -(143 as Field),
    6424,
    -(134647 as Field),
    1741789,
    -(15555660 as Field),
    101635445,
    -(502162301 as Field),
    1912293918,
    -(5667543301 as Field),
    13107436182,
    -(23581357307 as Field),
    32769990622,
    -(35069831204 as Field),
    30661759115,
    -(13717002621 as Field),
];

// Hint: (k, m) with m = floor(x / 2^k) in [2Q, 4Q) (k may be negative, then
// m = x * 2^-k exactly), for 1 <= x < 2^126.
unconstrained fn hint_log2_decompose(x: Field) -> (Field, Field) {
    let two_q: u128 = 2 * Q_U128;
    let four_q: u128 = 4 * Q_U128;
    let xx = x as u128;
    // k = up - down.
    let mut up: u32 = 0;
    let mut down: u32 = 0;
    let mut m: u128 = xx;
    for _ in 0..128 {
        if m < two_q {
            m = m * 2;
            down = down + 1;
        } else if m >= four_q {
            m = m / 2;
            up = up + 1;
        }
    }
    ((up as Field) - (down as Field), m as Field)
}

/// Q32 natural log of x > 0 (fails for x <= 0). x = m * 2^k exactly up to the
/// floor: m = floor(x / 2^k) in [2Q, 4Q) for k >= 0, m = x * 2^-k for k < 0;
/// this pins k (halving m leaves [2Q, 4Q)) and m uniquely. Accuracy is
/// limited by the Q32-quantized degree-14 coefficients (about 1e-2 near
/// m = 4); see `test_horner_eval_log2_in_domain` in src/prove/kernel.rs.
pub fn qlog(x: Field) -> Field {
    let (negative, mag) = to_signed(x);
    assert(!negative & (mag != 0), "qlog: non-positive input");
    let x_f = mag as Field;
    // Safety: the checks below pin (k, m) uniquely.
    let (k, m) = unsafe { hint_log2_decompose(x_f) };
    (m - 2 * Q).assert_max_bit_size::<33>(); // 2Q <= m < 4Q
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
    let log2_m = horner_q(m, LOG2_COEFS_Q);
    let k_q = k * Q;
    qmul(k_q + log2_m, LN2_Q)
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
// arctan2, arcsin, arccos
// ---------------------------------------------------------------------------

// atan on [-1, 1], degree 15, quantized to Q32, leading coefficient first
// (ATAN_COEFS in src/prove/kernel.rs lists the same values constant-term
// first; the odd-power coefficients are kept, the ~1e-13 even ones are 0).
global ATAN_COEFS_Q: [Field; 16] = [
    -(17009175 as Field),
    0,
    92381544,
    0,
    -(237862581 as Field),
    0,
    412388002,
    0,
    -(596658626 as Field),
    0,
    856548815,
    0,
    -(1431492759 as Field),
    0,
    4294964071,
    0,
];

pub fn qarctan(t: Field) -> Field {
    horner_q(t, ATAN_COEFS_Q)
}

pub fn qarctan2(y: Field, x: Field) -> Field {
    let (sy, my) = to_signed(y);
    let (sx, mx) = to_signed(x);
    let abs_y: Field = my as Field;
    let abs_x: Field = mx as Field;
    let swap: bool = my > mx;
    let num: Field = if swap { abs_x } else { abs_y };
    let den: Field = if swap { abs_y } else { abs_x };
    // One ulp keeps the divisor non-zero at the origin (atan2(0, 0) = 0).
    let t: Field = qdiv(num, den + 1);
    let poly: Field = qarctan(t);
    let a0: Field = if swap { HALF_PI_Q - poly } else { poly };
    let a1: Field = if sx { PI_Q - a0 } else { a0 };
    if sy {
        -a1
    } else {
        a1
    }
}

pub fn qarcsin(x: Field) -> Field {
    // arcsin(x) = arctan2(x, sqrt(1 - x*x))
    let s: Field = qsqrt(Q - qmul(x, x));
    qarctan2(x, s)
}

pub fn qarccos(x: Field) -> Field {
    // arccos(x) = arctan2(sqrt(1 - x*x), x)
    let s: Field = qsqrt(Q - qmul(x, x));
    qarctan2(s, x)
}

// ---------------------------------------------------------------------------
// Hyperbolic functions
// ---------------------------------------------------------------------------

// floor(v / 2), as the mock's fp_div_prescale(v, 2.0).
fn half_floor(v: Field) -> Field {
    let (q, _r) = euclid_divmod_small::<1>(v, 2);
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

#[test(should_fail_with = "qdiv: divide by zero")]
fn test_qdiv_zero_fails() {
    let _ = qdiv(Q, 0);
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
fn test_qlt() {
    assert(qlt(-q_from_int(3), -q_from_int(2)));
    assert(qlt(-q_from_int(2), q_from_int(0)));
    assert(qlt(q_from_int(1), q_from_int(2)));
    assert(!qlt(q_from_int(5), q_from_int(5)));
    assert(!qlt(0, 0));
}

#[test]
fn test_q_floor_div_signs() {
    assert(q_floor_div(7, 3) == 2);
    assert(q_floor_div(-7, 3) == -3);
    assert(q_floor_div(6, 3) == 2);
    assert(q_floor_div(-6, 3) == -2);
    // d < 0: Euclidean convention (as the mock), not Python's.
    assert(q_floor_div(7, -3) == -2);
    assert(q_floor_div(-7, -3) == 3);
}

#[test]
fn test_q_mod_signs() {
    assert(q_mod(7, 3) == 1);
    assert(q_mod(-7, 3) == 2);
    assert(q_mod(6, 3) == 0);
    assert(q_mod(7, -3) == 1);
    assert(q_mod(-7, -3) == 2);
}

#[test(should_fail)]
fn test_q_floor_div_by_zero_fails() {
    let _ = q_floor_div(7, 0);
}

#[test]
fn test_euclid_divmod_small() {
    let (q, r) = euclid_divmod_small::<3>(-7, 3);
    assert((q == -3) & (r == 2));
    let (q2, r2) = euclid_divmod_small::<1>(5, 2);
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
fn test_qsqrt_rounds_to_nearest() {
    assert(qsqrt(0) == 0);
    assert(qsqrt(q_from_int(4)) == q_from_int(2));
    // sqrt(2) * 2^32 = 6074000999.95 -> 6074001000.
    assert(qsqrt(q_from_int(2)) == 6074001000);
    // Smallest positive input: sqrt(2^-32) = 2^-16 -> 65536.
    assert(qsqrt(1) == 65536);
}

#[test(should_fail_with = "qsqrt: negative input")]
fn test_qsqrt_negative_fails() {
    let _ = qsqrt(-Q);
}

#[test]
fn test_qsin_range_reduction() {
    // Reference values: round(sin(v) * 2^32), round(cos(v) * 2^32).
    q_eq_with_tol(qsin(q_from_int(277)), 2207631717, 4);
    q_eq_with_tol(qsin(q_from_int(324)), -1735446903, 4);
    q_eq_with_tol(qsin(-q_from_int(277)), -2207631717, 4);
    q_eq_with_tol(qsin(Q), 3614090360, 4); // sin(1)
    q_eq_with_tol(qcos(Q), 2320580734, 4); // cos(1)
    q_eq_with_tol(qcos(-Q), 2320580734, 4); // cos(-1)
    assert(qsin(0) == 0);
    q_eq_with_tol(qcos(0), Q, 4);
}

#[test]
fn test_qexp_qlog() {
    // e = 11674931555 / 2^32; exp(-1) = 1580030169 / 2^32.
    q_eq_with_tol(qexp(Q), 11674931555, 64);
    q_eq_with_tol(qexp(-Q), 1580030169, 64);
    q_eq_with_tol(qexp(0), Q, 8);
    q_eq_with_tol(qlog(11674931555), Q, 1000 * TOL_DEFAULT);
}

#[test]
fn test_qsinh_qcosh() {
    // round(sinh(1) * 2^32), round(cosh(1) * 2^32).
    q_eq_with_tol(qsinh(Q), 5047450693, 64);
    q_eq_with_tol(qcosh(Q), 6627480862, 64);
}

#[test]
fn test_qpow_small() {
    // 2.0 ^ 3.0 = 8 through log/exp (coarse: qlog's polynomial).
    q_eq_with_tol(qpow(q_from_int(2), q_from_int(3)), q_from_int(8), 200000000);
    assert(qpow_int_q32(q_from_int(2), 3) == q_from_int(8));
    assert(qpow_int_q32(-q_from_int(2), 2) == q_from_int(4));
}

#[test(should_fail_with = "qpow_int_q32: exponent must be below 2^32")]
fn test_qpow_int_large_exponent_fails() {
    let _ = qpow_int_q32(Q, 4294967296);
}

#[test]
fn test_qarctan2_quadrant() {
    // round(atan(1) * 2^32), round(asin(0.5) * 2^32); the polynomial's own
    // error is about 4.3e-8 (~185 ulps).
    q_eq_with_tol(qarctan2(Q, Q), 3373259426, 400); // pi/4
    q_eq_with_tol(qarctan2(Q, -Q), 3 * 3373259426, 400); // 3pi/4
    q_eq_with_tol(qarctan2(-Q, -Q), -3 * 3373259426, 400); // -3pi/4
    q_eq_with_tol(qarcsin(Q / 2), 2248839617, 400); // pi/6
    q_eq_with_tol(qarctan(Q / 2), 1991351318, 400); // atan(0.5)
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
fn test_euclid_rejects_field_inverse_quotient() {
    // 7 = q*3 + 0 with q = 7 * 3^-1 mod p: a huge field element.
    let q: Field = 7 / 3;
    constrain_euclid_divmod(7, 3, q, 0);
}

#[test(should_fail)]
fn test_euclid_rejects_other_remainder() {
    // 7 % 3 claimed as 2 needs q = 5 * 3^-1 mod p.
    let q: Field = 5 / 3;
    constrain_euclid_divmod(7, 3, q, 2);
}

#[test(should_fail)]
fn test_euclid_rejects_remainder_too_large() {
    constrain_euclid_divmod(7, 3, 1, 4);
}

#[test(should_fail)]
fn test_euclid_rejects_negative_remainder() {
    constrain_euclid_divmod(7, 3, 3, -2);
}

#[test]
fn test_euclid_accepts_honest() {
    constrain_euclid_divmod(-7, 3, -3, 2);
}

#[test(should_fail)]
fn test_euclid_small_rejects_field_inverse_quotient() {
    let q: Field = 7 / 3;
    constrain_euclid_divmod_small::<2>(7, 3, q, 0);
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
fn test_isqrt_rejects_floor_when_round_differs() {
    // xq = 2Q^2: round(sqrt) = 6074001000, floor = 6074000999.
    constrain_isqrt_round(2 * Q * Q, 6074000999);
}

#[test(should_fail)]
fn test_isqrt_rejects_negative_root() {
    constrain_isqrt_round(4 * Q * Q, -2 * Q);
}

#[test]
fn test_isqrt_accepts_honest() {
    constrain_isqrt_round(2 * Q * Q, 6074001000);
    constrain_isqrt_round(0, 0);
}
