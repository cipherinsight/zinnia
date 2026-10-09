// exp, log and tanh on Q32 values: the constructions of the halo2-base
// baselines (halo2_hw/src/q32/transcendental.rs, where the derivations and
// the error measurements are written out), with range-check widths chosen
// for Barretenberg (multiples of 14 bits, or 14k + 4 and 14k + 8, whose last
// limbs share the tables of the 32- and 64-bit checks).
//
// CONVENTIONS fixes the rounding of sin and cos only. For these functions the
// rule (NPB_BASELINES.md, section 5, rule 5) is a value within the case's
// tolerance (1e-6 absolute plus 1e-6 relative) of the real function for every
// input whose result fits Q32. Each reduces its argument to a short interval,
// evaluates a minimax polynomial there exactly in the field (integer
// coefficients times integer powers of the reduced argument: an exact integer
// at a known scale), and brings the value back to Q32 with one floor division
// by a power of two. Errors (integer emulation on 3e5 arguments; the tests
// check them):
//
// * exp: x = k ln 2 + r, k the nearest integer to x / ln 2, |r| <= ln 2 / 2,
//   exp(x) = 2^k exp(r), degree-5 polynomial. Relative error below 8.1e-8
//   plus the final floor (2^-32). An argument below -32 ln 2 is clamped
//   there (the result is then at most 1 unit of 2^-32).
// * log: x = 2^j m with m in [1/sqrt 2, sqrt 2), log m = 2 atanh(s) with
//   s = (m - 1) / (m + 1) in [-0.172, 0.172], an odd degree-5 polynomial in
//   s. Absolute error below 2.3e-8.
// * tanh: tanh(|x|) = (1 - e) / (1 + e) with e = exp(-2 |x|), then the sign.
//   Absolute error below 4e-8.
//
// No witness where the result is not finite or leaves the window: log of a
// non-positive value, exp of an argument at or above 31 ln 2 (and, by the
// polynomial's error, within 1e-7 below it).

use crate::{OFFSET, SCALE, window};

global OFFSET_U64: u64 = 9223372036854775808;
// round(ln 2 * 2^32) and floor(LN2_Q / 2).
global LN2_Q: Field = 2977044472;
global HALF_LN2_Q: Field = 1488522236;
// exp clamps its argument at -32 ln 2 (EXP_MIN_Q = -32 LN2_Q).
global EXP_MIN_Q: Field = -95265423104;
global EXP_MIN_W: u64 = 9223371941589352704; // window(EXP_MIN_Q)
global NEG_EXP_MIN_U64: u64 = 95265423104; // -EXP_MIN_Q
global HALF_NEG_EXP_MIN: Field = 47632711552; // -EXP_MIN_Q / 2
// The degree-5 minimax polynomial for exp(r) on [-ln 2 / 2, ln 2 / 2]
// (relative error 7.5e-8; 8.1e-8 with the coefficients rounded),
// coefficients A_i = round(c_i * 2^26) (constant first: 1.0000000717,
// 0.9999996920, 0.4999889485, 0.1666757473, 0.0419153820, 0.0082976551),
// stored as A_i * 2^(32 (5 - i)) so that Horner's rule on the Q32 code r
// gives the polynomial at scale 2^186 (186 = 13 * 14 + 4).
global EXP_COEFS: [Field; 6] = [
    98079721922925073589448800756043951333014402099413254144,
    22835955937365652757224842780325166114991505408,
    2658397204273246161669119710949539840,
    206334580096952292836638720,
    12081287737114624,
    556846,
];
global EXP_SHIFT: Field = 98079714615416886934934209737619787751599303819750539264; // 2^186
// 2^(2^i) - 1: 2^e is the product of 1 + b_i (2^(2^i) - 1) over the bits b_i of e.
global POW_FACTORS: [Field; 6] = [1, 3, 15, 255, 65535, 4294967295];
// ceil(2^63 / sqrt 2) and 2 LOG_L - 1: log normalizes its argument to [LOG_L, 2 LOG_L).
global LOG_L: Field = 6521908912666391107;
global LOG_L_TOP: Field = 13043817825332782213;
// The odd degree-5 minimax polynomial for log((1 + s) / (1 - s)) on
// |s| <= 0.172 (absolute error 2.1e-8), s (c0 + c1 s^2 + c2 s^4), with
// C_i = round(c_i * 2^30) (2.0000008370, 0.6664407804, 0.4151770601) stored
// as C0 2^128, C1 2^64, C2: the polynomial at the Q32 code s is then exact at
// scale 2^190, and log m in Q32 is its floor by 2^158.
global LOG_COEFS: [Field; 3] =
    [730751124579299321025521069931913590987588370432, 13200219611431690480673357824, 445792974];
global LOG_SHIFT: Field = 365375409332725729550921208179070754913983135744; // 2^158
global LOG_SHIFT_HALF: Field = 784637716923335095479473677900958302012794430558004314112; // 2^189
global TWO_31: Field = 2147483648;

// ---------------------------------------------------------------------- hints

// (floor(x / d), x mod d) for a Field read as a signed integer below 2^126
// in magnitude and a positive d below 2^126.
unconstrained fn divmod_hint(x: Field, d: Field) -> (Field, Field) {
    let neg = !x.lt(OFFSET * OFFSET); // x < 0 (|x| < 2^126)
    let m: u128 = if neg { (-x) as u128 } else { x as u128 };
    let dm: u128 = d as u128;
    let q: u128 = m / dm;
    let r: u128 = m - q * dm;
    if !neg {
        (q as Field, r as Field)
    } else if r == 0 {
        (-(q as Field), 0)
    } else {
        (-((q + 1) as Field), (dm - r) as Field)
    }
}

// (floor(x / 2^K), x mod 2^K) for a Field read as an integer in [0, p).
unconstrained fn shift_hint<let K: u32>(x: Field) -> (Field, Field) {
    let bits: [bool; 254] = x.to_le_bits();
    let mut q: Field = 0;
    let mut r: Field = 0;
    let mut weight: Field = 1;
    for i in 0..254 {
        if i == K {
            weight = 1;
        }
        if i < K {
            r += (bits[i] as Field) * weight;
        } else {
            q += (bits[i] as Field) * weight;
        }
        weight *= 2;
    }
    (q, r)
}

// The j in [0, 64) with x 2^j in [LOG_L, 2 LOG_L) for x > 0 (0 otherwise).
unconstrained fn log_norm_hint(x: Field) -> Field {
    let mut j: u32 = 0;
    if x.lt(OFFSET) & (x != 0) {
        let v = x as u128;
        let l = LOG_L as u128;
        while (v << (j as u128)) < l {
            j += 1;
        }
    }
    j as Field
}

// ------------------------------------------------------------------------ exp

// np.exp(x): x clamped at -32 ln 2, then `exp_clamped`, the result
// window-checked (an argument whose result leaves the window has no witness).
pub fn exp(x: Field) -> Field {
    let low = window(x) < EXP_MIN_W;
    let z = if low { EXP_MIN_Q } else { x };
    let y = exp_clamped(z);
    let _ = window(y);
    y
}

// np.exp(-g) for g in [0, 2^64) (checked here; the check is shared with the
// caller's check of the same value, such as a gap m - x of `max_of`). The
// clamp compares g itself, and the result, at most 1, needs a 42-bit check
// (3 limbs of 14 bits) where the window needs 64.
pub fn exp_neg(g: Field) -> Field {
    g.assert_max_bit_size::<64>();
    let far = (g as u64) > NEG_EXP_MIN_U64;
    let z = if far { EXP_MIN_Q } else { -g };
    exp_at_most_one(z)
}

// exp(z) for z in [-32 ln 2, 0]; the result, at most 2^32 (1 + 8e-8), is
// checked to [0, 2^42), which pins it (`check_exp_scale`).
fn exp_at_most_one(z: Field) -> Field {
    let y = exp_clamped(z);
    y.assert_max_bit_size::<42>();
    y
}

// exp(z) for z in [-32 ln 2, 2^63): t = z + 32 ln 2 + ln 2 / 2 = e LN2_Q + R
// (`check_exp_reduction`: e = k + 32, r = R - HALF_LN2_Q the reduced
// argument), then y = floor(P(r) 2^e / 2^186) with P the exact polynomial at
// scale 2^186 (`check_exp_scale`). The caller checks y to the window or
// tighter, which pins it.
fn exp_clamped(z: Field) -> Field {
    let t = z - EXP_MIN_Q + HALF_LN2_Q;
    // Safety: e and R are pinned by `check_exp_reduction`.
    let (e, rem) = unsafe { divmod_hint(t, LN2_Q) };
    let bits = check_exp_reduction(t, e, rem);
    let r = rem - HALF_LN2_Q;
    let mut p = EXP_COEFS[5];
    for i in 1..6 {
        p = p * r + EXP_COEFS[5 - i];
    }
    let x = p * pow2_of_bits(bits);
    // Safety: y and the remainder are pinned by `check_exp_scale` and the
    // caller's range check of y.
    let (y, rem2) = unsafe { shift_hint::<186>(x) };
    check_exp_scale(x, y, rem2);
    y
}

// t = e LN2_Q + R with e six bits (returned) and R in [0, LN2_Q) (two 32-bit
// checks). e LN2_Q + R is an integer in [0, 2^38), so for t in [0, 64 LN2_Q)
// this is the integer floor decomposition and (e, R) is unique; a larger t
// (a result at or above 2^31.5) has no witness.
fn check_exp_reduction(t: Field, e: Field, rem: Field) -> [bool; 6] {
    let bits: [bool; 6] = e.to_le_bits();
    assert(e * LN2_Q + rem == t, "exp: reduction");
    rem.assert_max_bit_size::<32>();
    (LN2_Q - 1 - rem).assert_max_bit_size::<32>();
    bits
}

// 2^e from the little-endian bits of e.
fn pow2_of_bits(bits: [bool; 6]) -> Field {
    let mut power: Field = 1;
    for i in 0..6 {
        power *= 1 + (bits[i] as Field) * POW_FACTORS[i];
    }
    power
}

// x = y 2^186 + rem with rem in [0, 2^186). x = P 2^e is an integer in
// [0, 1.4143 * 2^249), and with y in the window (the caller's check, or a
// tighter one) y 2^186 + rem is an integer in [-2^249, 2^249 + 2^186); the
// two differ by less than p, so the field identity is the integer floor
// decomposition (in particular y >= 0) and (y, rem) is unique.
fn check_exp_scale(x: Field, y: Field, rem: Field) {
    assert(y * EXP_SHIFT + rem == x, "exp: scale");
    rem.assert_max_bit_size::<186>();
}

// ------------------------------------------------------------------------ log

// np.log(x) for x > 0; zero and negative arguments have no witness (NumPy
// returns -inf and nan).
//
// * x in the window (`window`, shared with any other check of x), N = x 2^j
//   in [L, 2L), L = ceil(2^63 / sqrt 2) (two 64-bit checks): x 2^j is an
//   integer of magnitude below 2^126, so a non-positive x fails the checks and
//   a positive one fixes j. Then x = m 2^(31 - j), m = N / 2^63.
// * s = floor((N - 2^63) 2^32 / (N + 2^63)), (m - 1) / (m + 1) in Q32:
//   remainder and divisor gap in [0, 2^70), s + 2^31 in [0, 2^32);
//   |s (N + 2^63) + rem| < 2^97, so (s, rem) is unique.
// * log m = floor(P / 2^158) for P = C0 s 2^128 + C1 s^3 2^64 + C2 s^5
//   (`check_log_scale`), and log x = log m + (31 - j) LN2_Q.
pub fn log(x: Field) -> Field {
    let _ = window(x);
    // Safety: j is pinned by the checks of N in `check_log_norm`.
    let j = unsafe { log_norm_hint(x) };
    let n = check_log_norm(x, j);
    let den = n + OFFSET;
    let num = (n - OFFSET) * SCALE;
    // Safety: s and its remainder are pinned by `check_log_ratio`.
    let (s, rem) = unsafe { divmod_hint(num, den) };
    check_log_ratio(num, den, s, rem);
    let s2 = s * s;
    let p = ((LOG_COEFS[2] * s2 + LOG_COEFS[1]) * s2 + LOG_COEFS[0]) * s;
    let shifted = p + LOG_SHIFT_HALF;
    // Safety: pinned by `check_log_scale`.
    let (v, rem2) = unsafe { shift_hint::<158>(shifted) };
    check_log_scale(shifted, v, rem2);
    v - TWO_31 + (31 - j) * LN2_Q
}

fn check_log_norm(x: Field, j: Field) -> Field {
    let n = x * pow2_of_bits(j.to_le_bits());
    (n - LOG_L).assert_max_bit_size::<64>();
    (LOG_L_TOP - n).assert_max_bit_size::<64>();
    n
}

fn check_log_ratio(num: Field, den: Field, s: Field, rem: Field) {
    assert(s * den + rem == num, "log: ratio");
    rem.assert_max_bit_size::<70>();
    (den - rem - 1).assert_max_bit_size::<70>();
    (s + TWO_31).assert_max_bit_size::<32>();
}

// P + 2^189 = v 2^158 + rem with v in [0, 2^32) and rem in [0, 2^158)
// (158 = 11 * 14 + 4). |P| < 0.35 * 2^190, so P + 2^189 lies in (0, 2^190),
// and v 2^158 + rem below 2^191: the integer floor decomposition, unique;
// v - 2^31 = floor(P / 2^158).
fn check_log_scale(shifted: Field, v: Field, rem: Field) {
    assert(v * LOG_SHIFT + rem == shifted, "log: scale");
    rem.assert_max_bit_size::<158>();
    v.assert_max_bit_size::<32>();
}

// ----------------------------------------------------------------------- tanh

// np.tanh(x): the sign of x (its window), |x|, 2 |x| clamped at 32 ln 2,
// e = exp(-2 |x|) (`exp_at_most_one`), t = floor((1 - e) / (1 + e)) in Q32
// (`check_tanh_ratio`), and the sign.
pub fn tanh(x: Field) -> Field {
    let w = window(x);
    let neg = w < OFFSET_U64;
    let magnitude = if neg { -x } else { x };
    // |x| <= 2^63: the window offset is 2^63 +- |x|.
    let far = (w > OFFSET_U64 + (HALF_NEG_EXP_MIN as u64)) | (w < OFFSET_U64 - (HALF_NEG_EXP_MIN as u64));
    let z = if far { EXP_MIN_Q } else { -2 * magnitude };
    let e = exp_at_most_one(z);
    let num = (SCALE - e) * SCALE;
    let den = SCALE + e;
    // Safety: t and its remainder are pinned by `check_tanh_ratio`.
    let (t, rem) = unsafe { divmod_hint(num, den) };
    check_tanh_ratio(num, den, t, rem);
    if neg { -t } else { t }
}

// num = t den + rem for den = 2^32 + e in [2^32, 2^43): remainder and divisor
// gap in [0, 2^42), t + 2^32 in [0, 2^42) (t is slightly negative when e
// exceeds 2^32 by the polynomial's error). The terms stay below 2^86, so
// (t, rem) is unique.
fn check_tanh_ratio(num: Field, den: Field, t: Field, rem: Field) {
    assert(t * den + rem == num, "tanh: ratio");
    rem.assert_max_bit_size::<42>();
    (den - rem - 1).assert_max_bit_size::<42>();
    (t + SCALE).assert_max_bit_size::<42>();
}

// ---------------------------------------------------------------------- tests

global ULP: Field = 1; // 2^-32

// |a - b| <= tol, all Q32 codes in the window.
fn close(a: Field, b: Field, tol: Field) -> bool {
    let (wa, wb) = (window(a), window(b));
    let d = if wa > wb { wa - wb } else { wb - wa };
    d <= (tol as u64)
}

#[test]
fn test_exp() {
    // Expected values: round(exp(v) * 2^32); tolerance 8.1e-8 relative + 1 ulp
    // (the argument codes are round(v * 2^32)).
    assert(close(exp(0), 4294967296, 349));
    assert(close(exp(4294967296), 11674931555, 947)); // exp(1)
    assert(close(exp(-4294967296), 1580030169, 129)); // exp(-1)
    assert(close(exp(32212254720), 7765483039870, 629006)); // exp(7.5)
    assert(close(exp(-21474836480), 28939262, 4)); // exp(-5)
    assert(close(exp(-128849018880), 0, ULP)); // exp(-30), clamped
    assert(close(exp(-9223372036854775808), 0, ULP)); // exp(-2^31)
    assert(close(exp(1488522236), 6074001000, 493)); // exp(ln 2 / 2), the reduction's edge
    assert(exp_neg(4294967296) == exp(-4294967296));
    assert(exp_neg(128849018880) == exp(-128849018880));
}

#[test(should_fail)]
fn test_exp_overflow() {
    let _ = exp(92288378632); // 31 ln 2
}

#[test(should_fail)]
fn test_exp_neg_out_of_range() {
    let _ = exp_neg(-1);
}

#[test(should_fail)]
fn test_exp_reduction_next_quotient() {
    let t = 5 * LN2_Q + 1000;
    let _ = check_exp_reduction(t, 6, 1000 - LN2_Q);
}

#[test(should_fail)]
fn test_exp_reduction_previous_quotient() {
    // R = 1000 + LN2_Q < 2^32 passes the 32-bit check, not the gap check.
    let t = 5 * LN2_Q + 1000;
    let _ = check_exp_reduction(t, 4, 1000 + LN2_Q);
}

#[test(should_fail)]
fn test_exp_reduction_field_quotient() {
    let t = 5 * LN2_Q + 1000;
    let _ = check_exp_reduction(t, (t - 1001) / LN2_Q, 1001);
}

#[test]
fn test_exp_reduction_honest() {
    let t = 5 * LN2_Q + 1000;
    let _ = check_exp_reduction(t, 5, 1000);
}

#[test(should_fail)]
fn test_exp_scale_next_quotient() {
    let x = 3 * EXP_SHIFT + 12345;
    check_exp_scale(x, 4, 12345 - EXP_SHIFT);
}

#[test(should_fail)]
fn test_exp_scale_field_quotient() {
    // The non-integer y fails the caller's window check.
    let x = 3 * EXP_SHIFT + 12345;
    let y = (x - 12346) / EXP_SHIFT;
    check_exp_scale(x, y, 12346);
    let _ = window(y);
}

#[test]
fn test_log() {
    // Expected values: round(log(v) * 2^32); tolerance 2.3e-8 (99 ulp).
    assert(close(log(4294967296), 0, 99)); // log(1)
    assert(close(log(8589934592), 2977044472, 99)); // log(2)
    assert(close(log(2147483648), -2977044472, 99)); // log(0.5)
    assert(close(log(14173392077), 5127857956, 99)); // log(3.3)
    assert(close(log(1), -95265423098, 99)); // log(2^-32)
    assert(close(log(9223372036854775807), 92288378626, 99)); // log(2^31)
    assert(close(log(3037000500), -1488522236, 99)); // log(1 / sqrt 2)
    assert(close(log(6074001000), 1488522236, 99)); // log(sqrt 2)
}

#[test(should_fail)]
fn test_log_zero() {
    let _ = log(0);
}

#[test(should_fail)]
fn test_log_negative() {
    let _ = log(-4294967296);
}

#[test(should_fail)]
fn test_log_norm_too_small() {
    // log(3.3): the honest j is 29; 28 leaves N below L.
    let _ = check_log_norm(14173392077, 28);
}

#[test(should_fail)]
fn test_log_norm_too_large() {
    let _ = check_log_norm(14173392077, 30);
}

#[test]
fn test_log_norm_honest() {
    let _ = check_log_norm(14173392077, 29);
}

#[test(should_fail)]
fn test_log_ratio_next_quotient() {
    let (num, den) = (-(1000 * SCALE), 3 * OFFSET);
    // Safety: the honest hint, tampered below.
    let (s, rem) = unsafe { divmod_hint(num, den) };
    check_log_ratio(num, den, s + 1, rem - den);
}

#[test(should_fail)]
fn test_log_ratio_field_quotient() {
    let (num, den) = (-(1000 * SCALE), 3 * OFFSET);
    // Safety: the honest hint, tampered below.
    let (_, rem) = unsafe { divmod_hint(num, den) };
    check_log_ratio(num, den, (num - rem - 1) / den, rem + 1);
}

#[test(should_fail)]
fn test_log_scale_next_quotient() {
    let shifted = 12345 * LOG_SHIFT + 678;
    check_log_scale(shifted, 12346, 678 - LOG_SHIFT);
}

#[test]
fn test_tanh() {
    // Expected values: round(tanh(v) * 2^32); tolerance 4e-8 (172 ulp).
    assert(close(tanh(0), 0, 172));
    assert(close(tanh(2147483648), 1984778077, 172)); // tanh(0.5)
    assert(close(tanh(-2147483648), -1984778077, 172)); // tanh(-0.5)
    assert(close(tanh(12884901888), 4273727625, 172)); // tanh(3)
    assert(close(tanh(47244640256), 4294967294, 172)); // tanh(11), clamped
    assert(close(tanh(-9223372036854775808), -4294967296, 172)); // tanh(-2^31)
}

#[test(should_fail)]
fn test_tanh_ratio_next_quotient() {
    let e = 1840700269; // 3/7 in Q32
    let (num, den) = ((SCALE - e) * SCALE, SCALE + e);
    // Safety: the honest hint, tampered below.
    let (t, rem) = unsafe { divmod_hint(num, den) };
    check_tanh_ratio(num, den, t + 1, rem - den);
}

#[test(should_fail)]
fn test_tanh_ratio_field_quotient() {
    let e = 1840700269;
    let (num, den) = ((SCALE - e) * SCALE, SCALE + e);
    // Safety: the honest hint, tampered below.
    let (_, rem) = unsafe { divmod_hint(num, den) };
    check_tanh_ratio(num, den, (num - rem - 1) / den, rem + 1);
}
