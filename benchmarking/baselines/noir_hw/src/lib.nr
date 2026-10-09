// Q32 helpers for the handwritten Noir baselines.
//
// Representation (same as Zinnia and the halo2-base baselines): a float v is
// the Field element of round(v * 2^32), negatives as p - |x|; an integer is
// its own Field element; a boolean is a Noir `bool` or a 0/1 Field.
//
// Orderings use the signed 64-bit window: `window(x)` proves x lies in
// [-2^63, 2^63) with one 64-bit range check of x + 2^63 and returns that
// offset as a `u64`; comparing offsets compares the signed values. Rounded
// arithmetic witnesses a quotient and remainder and pins both with range
// checks tight enough that the field identity is an integer identity, so the
// decomposition is unique and the prover cannot choose it. Rounding matches
// Zinnia: products floor(a * b / 2^32), quotients floor(a * 2^32 / b) with
// remainder in [0, |b|), float-to-int truncation toward zero.
//
// Range checks are `assert_max_bit_size` on the original Field, never a
// truncating `as` cast; a cast to `u64`/`u32` only follows such a check.

// exp, log and tanh, in their own file.
mod transcendental;
pub use transcendental::{exp, exp_neg, log, tanh};

pub global SCALE: Field = 4294967296; // 2^32
pub global OFFSET: Field = 9223372036854775808; // 2^63
global OFFSET_U64: u64 = 9223372036854775808;
global PI_Q: Field = 13493037705; // round(pi * 2^32)
global HALF_PI_Q: Field = 6746518852; // round(pi / 2 * 2^32)
global HALF_PI_U64: u64 = 6746518852;
// round(c * 2^32) of the degree-14 minimax polynomial for sin on [0, pi]
// (highest degree first); the same coefficients as Zinnia and halo2-graph.
pub global SIN_COEFS_Q: [Field; 15] =
    [0, 1, -2, -102, -13, 11858, -28, -852151, -16, 35791402, -2, -715827882, 0, 4294967296, 0];

// ------------------------------------------------------------------ window

// x + 2^63 as a u64, after proving x lies in [-2^63, 2^63).
pub fn window(x: Field) -> u64 {
    let w = x + OFFSET;
    w.assert_max_bit_size::<64>();
    w as u64
}

pub fn lt(a: Field, b: Field) -> bool {
    window(a) < window(b)
}

pub fn le(a: Field, b: Field) -> bool {
    window(a) <= window(b)
}

pub fn gt(a: Field, b: Field) -> bool {
    window(a) > window(b)
}

pub fn ge(a: Field, b: Field) -> bool {
    window(a) >= window(b)
}

pub fn is_neg(x: Field) -> bool {
    window(x) < OFFSET_U64
}

// |x| of a value in the window; |-2^63| = 2^63 comes out of the window
// (for a divisor's magnitude, where that is fine). Use `abs_int64` for a
// kernel's np.abs / abs.
pub fn abs(x: Field) -> Field {
    if is_neg(x) { -x } else { x }
}

// np.abs / abs of an int64 or a Q32 code x: |x| for x in (-2^63, 2^63), no
// proof otherwise. |i64::MIN| = 2^63 overflows int64 (NumPy wraps it back to
// i64::MIN); like Zinnia, the baselines have no proof there. With the hinted
// sign s and m = (s ? -x : x), one 64-bit check of 2m + s: m < 2^63 for
// s = 0, m <= 2^63 - 1 for s = 1, i.e. exactly |x| <= 2^63 - 1 (so x needs no
// window check of its own). For x != 0 the wrong sign gives m = p - |x| and
// 2m + s near p, far above 2^64; at x = 0 both signs give m = 0.
pub fn abs_int64(x: Field) -> Field {
    // Safety: the range check pins m (see above).
    let negative = unsafe { is_neg_hint(x) };
    let m = if negative { -x } else { x };
    (2 * m + negative as Field).assert_max_bit_size::<64>();
    m
}

unconstrained fn is_neg_hint(x: Field) -> bool {
    split(x).0
}

pub fn max(a: Field, b: Field) -> Field {
    if window(a) < window(b) { b } else { a }
}

pub fn min(a: Field, b: Field) -> Field {
    if window(b) < window(a) { b } else { a }
}

unconstrained fn extremum_hint<let N: u32>(xs: [Field; N], is_max: bool) -> Field {
    let mut m = xs[0];
    for i in 1..N {
        if (window(xs[i]) > window(m)) == is_max {
            m = xs[i];
        }
    }
    m
}

// max(xs) of a few values by a hint m, pinned by m in the window, m - x in
// [0, 2^64) for every x and prod (m - x) = 0. The xs need no window check of
// their own: the gap checks make every x an integer in (m - 2^64, m], so
// m >= x as integers, and the product puts m among the xs; m is then their
// integer maximum and lies in the window.
pub fn max_of<let N: u32>(xs: [Field; N]) -> Field {
    // Safety: m is pinned by `check_max`.
    let m = unsafe { extremum_hint(xs, true) };
    check_max(xs, m);
    m
}

// min(xs): `max_of` with the gaps x - m (every x an integer in [m, m + 2^64)).
// Unlike -max_of(-xs), it holds for a minimum of -2^63 (i64::MIN, or the Q32
// code of -2^31), whose negation leaves the window.
pub fn min_of<let N: u32>(xs: [Field; N]) -> Field {
    // Safety: m is pinned by `check_min`.
    let m = unsafe { extremum_hint(xs, false) };
    check_min(xs, m);
    m
}

fn check_max<let N: u32>(xs: [Field; N], m: Field) {
    check_extremum(xs, m, 1);
}

fn check_min<let N: u32>(xs: [Field; N], m: Field) {
    check_extremum(xs, m, -1);
}

// m in the window, sign * (m - x) in [0, 2^64) for every x, prod = 0.
fn check_extremum<let N: u32>(xs: [Field; N], m: Field, sign: Field) {
    let _ = window(m);
    let mut prod: Field = 1;
    for i in 0..N {
        let gap = sign * (m - xs[i]);
        gap.assert_max_bit_size::<64>();
        prod *= gap;
    }
    assert(prod == 0, "extremum: not one of the values");
}

// ------------------------------------------------------------ input domain

// Every input element the circuit reads is checked to its dtype's domain, so a
// proof means "some valid NumPy input gives these outputs": an int64, or a
// float whose Q32 code lies in the same signed 64-bit window. The check is the
// window check (one 64-bit range check of x + 2^63), which Noir shares with any
// later `window(x)` of the same value, so inputs that are compared cost nothing
// more. Inputs the circuit never reads are not checked.
pub fn check_input(x: Field) {
    let _ = window(x);
}

// `check_input` for every element.
pub fn check_inputs<let N: u32>(xs: [Field; N]) {
    for i in 0..N {
        check_input(xs[i]);
    }
}

// `x` after `check_input(x)`: for kernels that read only part of an input.
pub fn checked(x: Field) -> Field {
    check_input(x);
    x
}

// --------------------------------------------------------------- arithmetic

// (negative, magnitude) of a Field encoding a signed integer below 2^127.
unconstrained fn split(x: Field) -> (bool, u128) {
    let pos: u128 = x as u128;
    if (pos as Field) == x {
        (false, pos)
    } else {
        (true, (-x) as u128)
    }
}

// Floor division of a signed n by a positive d: (q, r), n = q * d + r, 0 <= r < d.
unconstrained fn divmod_hint(n: Field, d: Field) -> (Field, Field) {
    let (neg, m) = split(n);
    let dm: u128 = d as u128;
    if dm == 0 {
        (0, 0)
    } else {
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
}

// Floor division by a divisor that must be positive: n = q * d + r,
// 0 <= r < d (so d >= 1), q in the window. With d < 2^64 + 1 implied by the
// remainder checks, |q * d + r| < 2^128, so this is an integer identity and
// (q, r) is unique. A non-positive d has no witness.
pub fn divmod_pos(n: Field, d: Field) -> (Field, Field) {
    // Safety: q and r are pinned by `check_divmod`.
    let (q, r) = unsafe { divmod_hint(n, d) };
    check_divmod(n, d, q, r);
    (q, r)
}

fn check_divmod(n: Field, d: Field, q: Field, r: Field) {
    assert(q * d + r == n, "divmod_pos: recomposition");
    r.assert_max_bit_size::<64>();
    (d - r - 1).assert_max_bit_size::<64>();
    (q + OFFSET).assert_max_bit_size::<64>();
}

// Floor division by the constant 2^K (0 < K < 64): n = q * 2^K + r with r a
// K-bit value and q in the window, which pins (q, r) as in `divmod_pos` with
// one K-bit check in place of the two 64-bit remainder checks.
pub fn floor_div_pow2<let K: u32>(n: Field) -> Field {
    let d = (1 as u64 << (K as u64)) as Field;
    // Safety: q and r are pinned by `check_div_pow2`.
    let (q, r) = unsafe { divmod_hint(n, d) };
    check_div_pow2::<K>(n, q, r);
    q
}

fn check_div_pow2<let K: u32>(n: Field, q: Field, r: Field) {
    assert(q * (1 as u64 << (K as u64)) as Field + r == n, "floor_div_pow2: recomposition");
    r.assert_max_bit_size::<K>();
    (q + OFFSET).assert_max_bit_size::<64>();
}

// Floor division by a small positive constant D (2 <= D <= 256; Python //,
// or a float divided by the integer D, whose quotient floor(v * 2^32 /
// (D * 2^32)) is floor(v / D)): n = q * D + r with r and D - 1 - r 8-bit
// values (so 0 <= r < D) and q in the window, which pins (q, r) as in
// `divmod_pos`; the 8-bit checks use the range table the 64-bit checks
// already build.
pub fn floor_div_const<let D: u32>(n: Field) -> Field {
    // Safety: q and r are pinned by `check_div_const`.
    let (q, r) = unsafe { divmod_hint(n, D as Field) };
    check_div_const::<D>(n, q, r);
    q
}

fn check_div_const<let D: u32>(n: Field, q: Field, r: Field) {
    assert((2 <= D) & (D <= 256), "floor_div_const: D must be in [2, 256]");
    assert(q * (D as Field) + r == n, "floor_div_const: recomposition");
    r.assert_max_bit_size::<8>();
    ((D as Field) - 1 - r).assert_max_bit_size::<8>();
    (q + OFFSET).assert_max_bit_size::<64>();
}

// floor(p / 2^32) for a raw product or sum of products p in [-2^95, 2^95):
// p = q * 2^32 + r, r in [0, 2^32), q in the window.
pub fn rescale(p: Field) -> Field {
    // Safety: q and r are pinned by `check_rescale`.
    let (q, r) = unsafe { divmod_hint(p, SCALE) };
    check_rescale(p, q, r);
    q
}

fn check_rescale(p: Field, q: Field, r: Field) {
    assert(q * SCALE + r == p, "rescale: recomposition");
    r.assert_max_bit_size::<32>();
    (q + OFFSET).assert_max_bit_size::<64>();
}

// Float product floor(a * b / 2^32).
pub fn mul(a: Field, b: Field) -> Field {
    rescale(a * b)
}

// Float dot product with one rescale for the whole sum.
pub fn dot<let N: u32>(a: [Field; N], b: [Field; N]) -> Field {
    let mut acc: Field = 0;
    for i in 0..N {
        acc += a[i] * b[i];
    }
    rescale(acc)
}

// Float division floor(a * 2^32 / b), b != 0, remainder in [0, |b|).
pub fn div(a: Field, b: Field) -> Field {
    let neg = is_neg(b);
    let mag = if neg { -b } else { b };
    let (q, _r) = divmod_pos(a * SCALE, mag);
    if neg { -q } else { q }
}

// div(a, b) when the circuit already holds |b|: abs_b must be |b| proven
// below 2^63 (`abs_int64`, or selected together with b from such pairs), which
// is all `divmod_pos` needs; b is then abs_b or -abs_b, and b == abs_b tells
// the sign. Saves `div`'s window check of b. (A pivot chosen by
// argmax(abs(..)): the maximum magnitude is the pivot's.)
pub fn div_by_abs(a: Field, b: Field, abs_b: Field) -> Field {
    let (q, _r) = divmod_pos(a * SCALE, abs_b);
    if b == abs_b { q } else { -q }
}

// a / b for floats with 0 <= a < b (a ratio in [0, 1), such as a value divided
// by a larger maximum), rounded like `div`: floor(a * 2^32 / b). Any other a, b
// has no witness. a * 2^32 = q * b + r with r and b - r - 1 in [0, 2^64) (so
// 0 <= r < b, and b is an integer in [1, 2^65)) and q in [0, 2^32), which the
// precondition guarantees and which replaces the window check of q. The terms
// are below 2^97, so the identity holds over the integers and (q, r) is unique.
pub fn div_fraction(a: Field, b: Field) -> Field {
    // Safety: q and r are pinned by `check_div_fraction`.
    let (q, r) = unsafe { divmod_hint(a * SCALE, b) };
    check_div_fraction(a, b, q, r);
    q
}

fn check_div_fraction(a: Field, b: Field, q: Field, r: Field) {
    assert(q * b + r == a * SCALE, "div_fraction: recomposition");
    r.assert_max_bit_size::<64>();
    (b - r - 1).assert_max_bit_size::<64>();
    q.assert_max_bit_size::<32>();
}

// int(x) for a float x: truncation toward zero.
pub fn to_int(x: Field) -> Field {
    let (q, r) = divmod_pos(x, SCALE);
    if is_neg(x) & (r != 0) { q + 1 } else { q }
}

// float(n) for an integer n.
pub fn from_int(n: Field) -> Field {
    n * SCALE
}

// np.floor(x) for a float x.
pub fn floor(x: Field) -> Field {
    let (q, _r) = divmod_pos(x, SCALE);
    q * SCALE
}

unconstrained fn isqrt_hint(t: Field) -> Field {
    let n: u128 = t as u128;
    let mut lo: u128 = 0;
    let mut hi: u128 = 1 << 64;
    for _ in 0..65 {
        if hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if mid * mid <= n { lo = mid; } else { hi = mid; }
        }
    }
    lo as Field
}

// sqrt(x) for x >= 0: y = floor(sqrt(x * 2^32)), proved by y in [0, 2^63),
// y^2 <= x * 2^32 and x * 2^32 < (y + 1)^2 (all terms below 2^130, so these
// are integer facts). A negative x has no witness.
pub fn sqrt(x: Field) -> Field {
    isqrt(x * SCALE)
}

// floor(sqrt(n)) of a raw non-negative integer n below 2^126, with the
// constraints of `sqrt`; a negative n has no witness.
pub fn isqrt(n: Field) -> Field {
    // Safety: y is pinned by `check_isqrt`.
    let y = unsafe { isqrt_hint(n) };
    check_isqrt(n, y);
    y
}

fn check_sqrt(x: Field, y: Field) {
    check_isqrt(x * SCALE, y);
}

fn check_isqrt(t: Field, y: Field) {
    y.assert_max_bit_size::<63>();
    (t - y * y).assert_max_bit_size::<64>();
    (y * y + 2 * y - t).assert_max_bit_size::<64>();
}

// Horner evaluation, coefficients (Q32) highest degree first.
pub fn poly<let N: u32>(x: Field, coefs: [Field; N]) -> Field {
    let mut acc = coefs[0];
    for i in 1..N {
        acc = mul(x, acc) + coefs[i];
    }
    acc
}

unconstrained fn half_hint(q: Field) -> Field {
    ((q as u64) / 2) as Field
}

// The parity of a non-negative q below 2^33: q = 2h + odd with odd a bit and
// h in [0, 2^32), which determines odd.
pub fn is_odd(q: Field) -> bool {
    // Safety: h is pinned by `check_parity`.
    let h = unsafe { half_hint(q) };
    check_parity(q, h)
}

fn check_parity(q: Field, h: Field) -> bool {
    let odd = q - 2 * h;
    assert(odd * (odd - 1) == 0, "parity: not a bit");
    h.assert_max_bit_size::<32>();
    odd == 1
}

// sin(x): |x| = q * pi + r with r in [0, pi), r reflected into [0, pi/2]
// (sin(r) = sin(pi - r)), the minimax polynomial there, negated when exactly
// one of (x < 0) and (q odd) holds. With the reflection the error of the
// Q32-quantized polynomial stays below 2.4e-7 (1e-7 from the coefficients, up
// to 14 floors of 2^-32 each); evaluated up to pi it reaches 1.4e-4.
pub fn sin(x: Field) -> Field {
    let neg = is_neg(x);
    let ax = if neg { -x } else { x };
    let (q, r) = divmod_pos(ax, PI_Q);
    let odd = is_odd(q);
    // divmod_pos proved 0 <= r < PI_Q < 2^64, so the cast is exact.
    let r = if (r as u64) > HALF_PI_U64 { PI_Q - r } else { r };
    let y = poly(r, SIN_COEFS_Q);
    if neg != odd { -y } else { y }
}

// cos(x) = sin(x + pi/2).
pub fn cos(x: Field) -> Field {
    sin(x + HALF_PI_Q)
}

// ----------------------------------------------------------------- arctan2

global QUARTER_PI_Q: Field = 3373259426; // round(pi / 4 * 2^32)
// round(c * 2^32) of the degree-9 odd minimax polynomial for atan(t) on
// |t| <= 29/70, as t * P(t^2), P's coefficients highest degree first (Remez
// exchange on the absolute error, 3.52e-9; 3.53e-9 quantized). The same
// polynomial as the halo2 baselines' `q32::ATAN_COEFS`, written for these
// baselines independently of Zinnia's arctan2.
pub global ATAN_COEFS_Q: [Field; 5] = [332157462, -590751289, 857358406, -1431607205, 4294966890];

// np.arctan2(y, x) in [-pi, pi] with NumPy's quadrant rules on Q32 values
// (arctan2(0, x) = pi for x < 0, arctan2(0, 0) = 0), within 1e-8 of the real
// value for every pair of Q32 inputs. The signs come from the windows;
// `arctan2_nonneg` gives the angle of the magnitudes in [0, pi/2], which
// x < 0 reflects to pi - angle and y < 0 negates.
pub fn arctan2(y: Field, x: Field) -> Field {
    let y_neg = is_neg(y);
    let x_neg = is_neg(x);
    // |y|, |x| <= 2^63 (|-2^63| = 2^63).
    let ay = if y_neg { -y } else { y };
    let ax = if x_neg { -x } else { x };
    let angle = arctan2_magnitudes(ay, ax);
    let angle = if x_neg { PI_Q - angle } else { angle };
    if y_neg { -angle } else { angle }
}

// np.arctan2(y, x) for y, x already proven to lie in [0, 2^63) (outputs of
// `sqrt` or `abs_int64`): the angle in [0, pi/2], with the error bound of
// `arctan2`.
//
// Reduction: the octant bit `swapped` (x < y) orders the pair into
// num <= den, so atan(num / den) lies in [0, pi/4] and the angle is that or
// pi/2 minus it. The bit `over` (29 den < 70 num: the ratio above 29/70, just
// above tan(pi/8)) selects t = (num - den) / (num + den) and adds pi/4
// (atan(r) = pi/4 + atan((r - 1) / (r + 1))); otherwise t = num / den. Either
// way |t| <= 29/70, where the polynomial is within 3.6e-9 of atan. den = 0
// (both inputs 0) divides by 1 instead and gives 0.
pub fn arctan2_nonneg(y: Field, x: Field) -> Field {
    arctan2_magnitudes(y, x)
}

unconstrained fn atan_regions_hint(ay: Field, ax: Field) -> (Field, Field) {
    let swapped = ax.lt(ay);
    let (num, den) = if swapped { (ax, ay) } else { (ay, ax) };
    (swapped as Field, (29 * den).lt(70 * num) as Field)
}

// The angle of magnitudes ay, ax in [0, 2^63]. The two region bits are hints
// pinned by one range check each: with s = swapped, s (ay - ax - 1) +
// (1 - s) (ax - ay) in [0, 2^64), and with o = over, o (70 num - 29 den - 1) +
// (1 - o) (29 den - 70 num) in [0, 2^70) (five 14-bit limbs). The honest bit
// makes the checked value a non-negative integer below the bound; the other
// one makes it negative, p minus a small integer, far above it.
fn arctan2_magnitudes(ay: Field, ax: Field) -> Field {
    // Safety: the bits are pinned by `check_atan_regions`.
    let (swapped, over) = unsafe { atan_regions_hint(ay, ax) };
    let (num, den) = check_atan_regions(ay, ax, swapped, over);
    let numerator = num - over * den;
    let denominator = den + over * num + (den == 0) as Field;
    let t = div_half_ratio(numerator, denominator);
    let base = mul(poly(mul(t, t), ATAN_COEFS_Q), t) + over * QUARTER_PI_Q;
    base + swapped * (HALF_PI_Q - 2 * base)
}

// (num, den) = (min, max) of the magnitudes, after checking the region bits.
fn check_atan_regions(ay: Field, ax: Field, swapped: Field, over: Field) -> (Field, Field) {
    assert(swapped * (swapped - 1) == 0, "arctan2: region bit");
    assert(over * (over - 1) == 0, "arctan2: region bit");
    (swapped * (ay - ax - 1) + (1 - swapped) * (ax - ay)).assert_max_bit_size::<64>();
    let num = ay + swapped * (ax - ay);
    let den = ax + swapped * (ay - ax);
    (over * (70 * num - 29 * den - 1) + (1 - over) * (29 * den - 70 * num)).assert_max_bit_size::<70>();
    (num, den)
}

// floor(a * 2^32 / b) for a ratio |a / b| < 1/2 and b in [1, 2^64]:
// a * 2^32 = q * b + r with r and b - r - 1 in [0, 2^64) and q + 2^31 in
// [0, 2^32), which the precondition guarantees and which replaces the window
// check of q. The terms are below 2^97, so (q, r) is the unique floor
// decomposition.
fn div_half_ratio(a: Field, b: Field) -> Field {
    // Safety: q and r are pinned by `check_div_half_ratio`.
    let (q, r) = unsafe { divmod_hint(a * SCALE, b) };
    check_div_half_ratio(a, b, q, r);
    q
}

fn check_div_half_ratio(a: Field, b: Field, q: Field, r: Field) {
    assert(q * b + r == a * SCALE, "div_half_ratio: recomposition");
    r.assert_max_bit_size::<64>();
    (b - r - 1).assert_max_bit_size::<64>();
    (q + SCALE / 2).assert_max_bit_size::<32>();
}

// ---------------------------------------------------------------- clipping

// np.clip(x, lo, hi) for constants lo <= hi with hi - lo < 2^32 (integers, or
// Q32 codes for floats), on an x in the window (its input check). One 64-bit
// range check, where two comparisons with the bounds take two. Region bits l
// (x < lo) and h (x > hi) with l h = 0, and one 64-bit check of
// u = l (lo - 1 - x) + h (x - hi - 1) + m (D K - a (K - 1)), where
// m = 1 - l - h, a = x - lo, D = hi - lo and K = floor(2^64 / (D + 1)) + 1.
// Every term is an integer far below p / 2, so l forces x <= lo - 1, h forces
// x >= hi + 1, and m forces D K - 2^64 < a (K - 1) <= D K; since
// (D + 1) K > 2^64 and K > D + 1, that is -1 < a < D + 1, i.e. lo <= x <= hi.
// The three regions are disjoint, so the bits are pinned.
pub fn clip_const(x: Field, lo: Field, hi: Field) -> Field {
    // Safety: the region bits are pinned by `check_clip`.
    let (l, h) = unsafe { clip_region_hint(x, lo, hi) };
    check_clip(x, lo, hi, l, h)
}

unconstrained fn clip_region_hint(x: Field, lo: Field, hi: Field) -> (Field, Field) {
    let w = window(x);
    ((w < window(lo)) as Field, (w > window(hi)) as Field)
}

fn check_clip(x: Field, lo: Field, hi: Field, l: Field, h: Field) -> Field {
    let d = hi - lo;
    assert(d.lt(SCALE), "clip_const: needs lo <= hi < lo + 2^32");
    // A constant: lo and hi are.
    let k = ((1 as u128 << 64) / (d as u128 + 1) + 1) as Field;
    assert(l * (l - 1) == 0, "clip: region flag not a bit");
    assert(h * (h - 1) == 0, "clip: region flag not a bit");
    assert(l * h == 0, "clip: two regions");
    let m = 1 - l - h;
    let u = l * (lo - 1 - x) + h * (x - hi - 1) + m * (d * k - (x - lo) * (k - 1));
    u.assert_max_bit_size::<64>();
    x + l * (lo - x) + h * (hi - x)
}

// --------------------------------------------------------------- selection

// Boolean-mask selection: the kept items' payloads, in item order, at the
// front of N slots, zeros after them; w[i] is keep[i] * payload[i]. `hot` is
// the one-hot vector of the number of items kept before item i, i.e. of the
// slot item i fills if it is kept: slot j is the sum over i of hot_i[j] w[i],
// and `hot` moves one slot on after each kept item. Item i can only reach
// slots 0..=i, so the network is triangular; it is all Field arithmetic (no
// dynamic array writes, no range checks). keep[i] must be 0 or 1 (a bool
// cast to Field). Cheaper than native dynamic writes at these sizes (0442:
// 807 gates against 7,346).
pub fn compact<let N: u32>(keep: [Field; N], w: [Field; N]) -> [Field; N] {
    let mut slots: [Field; N] = [0; N];
    let mut hot: [Field; N] = [0; N];
    hot[0] = 1;
    for i in 0..N {
        for j in 0..i + 1 {
            slots[j] += hot[j] * w[i];
        }
        if i + 1 < N {
            let mut next: [Field; N] = [0; N];
            next[0] = hot[0] - keep[i] * hot[0];
            for j in 1..i + 2 {
                next[j] = hot[j] + keep[i] * (hot[j - 1] - hot[j]);
            }
            hot = next;
        }
    }
    slots
}

// x[keep] as a dynamic output: the number of kept values, then the kept
// values in order, zero-padded to N (`compact` with payload keep[i] x[i]).
pub fn select_kept<let N: u32>(x: [Field; N], keep: [bool; N]) -> [Field; N + 1] {
    let mut k: [Field; N] = [0; N];
    let mut w: [Field; N] = [0; N];
    let mut count: Field = 0;
    for i in 0..N {
        k[i] = keep[i] as Field;
        w[i] = k[i] * x[i];
        count += k[i];
    }
    let kept = compact(k, w);
    let mut out: [Field; N + 1] = [0; N + 1];
    out[0] = count;
    for j in 0..N {
        out[j + 1] = kept[j];
    }
    out
}

// `compact` for payloads of U cells: w[i][u] is keep[i] * payload_i[u].
pub fn compact_rows<let N: u32, let U: u32>(keep: [Field; N], w: [[Field; U]; N]) -> [[Field; U]; N] {
    let mut slots: [[Field; U]; N] = [[0; U]; N];
    let mut hot: [Field; N] = [0; N];
    hot[0] = 1;
    for i in 0..N {
        for j in 0..i + 1 {
            for u in 0..U {
                slots[j][u] += hot[j] * w[i][u];
            }
        }
        if i + 1 < N {
            let mut next: [Field; N] = [0; N];
            next[0] = hot[0] - keep[i] * hot[0];
            for j in 1..i + 2 {
                next[j] = hot[j] + keep[i] * (hot[j - 1] - hot[j]);
            }
            hot = next;
        }
    }
    slots
}

// ---------------------------------------------------------------- indexing

// idx as the index of an array of last + 1 elements, after proving
// 0 <= idx <= last (for last < 2^(W-1) and W <= 32); the construction of
// Zinnia's Noir library, so both sides pay the same. The two range checks are
// on 2 idx and last - idx, never on idx itself: the Noir compiler can drop a
// range check of a value used as a memory index (it leaves the bound to the
// backend's memory check), and the pinned bb (5.0.0-nightly.20260324) admits a
// read at index p - 1. 2 idx in [0, 2^W) leaves idx in [0, 2^(W-1)) or a
// field element (p + t) / 2 with t odd (about p / 2); last - idx in [0, 2^W)
// rules out the latter and every idx > last. W = 8 (last < 128) or W = 28 use
// range tables that 64-bit checks already build (64 = 4 x 14 + 8 bits). The
// last check only lets the compiler drop the truncation of the u32 cast.
pub fn array_index<let W: u32>(idx: Field, last: Field) -> u32 {
    assert(W <= 32, "array_index: W must be <= 32");
    assert(last.lt(2.pow_32((W - 1) as Field)), "array_index: array too long for W");
    (2 * idx).assert_max_bit_size::<W>();
    (last - idx).assert_max_bit_size::<W>();
    idx.assert_max_bit_size::<32>();
    idx as u32
}

// Hint: (position, negative) of a NumPy index into an axis of length len.
unconstrained fn index_pos_hint(idx: Field, len: Field) -> (Field, bool) {
    let neg = !idx.lt(OFFSET); // idx < 0
    (if neg { idx + len } else { idx }, neg)
}

// Position in [0, len) of a NumPy index idx in [-len, len) (negative counts
// from the end); any other idx has no witness, as NumPy raises IndexError.
// The hint is pinned by idx = pos - len neg with pos in [0, len)
// (`array_index`, W as there): neg = false gives idx in [0, len), neg = true
// gives idx in [-len, 0), disjoint ranges, so a valid idx has one witness.
pub fn index_pos<let W: u32>(idx: Field, len: u32) -> u32 {
    // Safety: pinned by the identity and `array_index`.
    let (pos, neg) = unsafe { index_pos_hint(idx, len as Field) };
    assert(idx == pos - if neg { len as Field } else { 0 }, "index out of bounds");
    array_index::<W>(pos, (len - 1) as Field)
}

// row[idx] with NumPy semantics by a one-hot over the 2K valid index values
// -K..K: position p is hit when idx is p or p - K, i.e. when
// (idx - p) (idx - p + K) is zero (no zero divisors in a field), one zero test
// per position; the assertion that exactly one position is hit pins idx to a
// valid index (no range check, no memory read). Cheaper than `index_pos` for
// short rows.
pub fn pick<let K: u32>(row: [Field; K], idx: Field) -> Field {
    let mut total: Field = 0;
    let mut value: Field = 0;
    for p in 0..K {
        let at = idx - (p as Field);
        let hit = (at * (at + (K as Field)) == 0) as Field;
        total += hit;
        value += hit * row[p];
    }
    assert(total == 1, "index out of bounds");
    value
}

// ------------------------------------------------------------------- tests

#[test]
fn test_window_order() {
    assert(lt(-5, 3));
    assert(!lt(3, -5));
    assert(lt(-5 * SCALE, -4 * SCALE));
    assert(is_neg(-1));
    assert(!is_neg(0));
}

#[test]
fn test_mul_floor() {
    // 2.5 * 4 = 10; (-1.5) * (2^-32) floors to -2^-32 * 2 = -2 ulp
    assert(mul(2 * SCALE + SCALE / 2, 4 * SCALE) == 10 * SCALE);
    assert(mul(-(SCALE + SCALE / 2), 1) == -2);
}

#[test]
fn test_div_and_int() {
    assert(div(SCALE, 4 * SCALE) == SCALE / 4);
    assert(div(SCALE, -4 * SCALE) == -(SCALE / 4));
    assert(to_int(-(SCALE + 1)) == -1);
    assert(to_int(SCALE + 1) == 1);
    assert(to_int(-(2 * SCALE)) == -2);
    assert(floor(-(SCALE / 2)) == -SCALE);
}

#[test]
fn test_div_by_abs() {
    assert(div_by_abs(SCALE, -4 * SCALE, 4 * SCALE) == div(SCALE, -4 * SCALE));
    assert(div_by_abs(-3 * SCALE, 2 * SCALE, 2 * SCALE) == div(-3 * SCALE, 2 * SCALE));
    assert(div_by_abs(7, -3, 3) == div(7, -3));
}

#[test]
fn test_div_fraction() {
    // 0.75 / 2.5 = 0.3
    assert(div_fraction(3 * SCALE / 4, 5 * SCALE / 2) == div(3 * SCALE / 4, 5 * SCALE / 2));
    assert(div_fraction(0, 1) == 0);
    check_div_fraction(1, 3, 1431655765, 1);
}

#[test(should_fail)]
fn test_div_fraction_next_quotient() {
    check_div_fraction(1, 3, 1431655766, -2);
}

#[test(should_fail)]
fn test_div_fraction_previous_quotient() {
    check_div_fraction(1, 3, 1431655764, 4);
}

#[test(should_fail)]
fn test_div_fraction_field_quotient() {
    check_div_fraction(1, 3, (SCALE - 2) / 3, 2);
}

#[test(should_fail)]
fn test_div_fraction_not_below_one() {
    let _ = div_fraction(3, 3);
}

#[test(should_fail)]
fn test_div_fraction_negative() {
    let _ = div_fraction(-1, 3);
}

#[test(should_fail)]
fn test_div_fraction_zero_divisor() {
    check_div_fraction(0, 0, 0, 0);
}

#[test(should_fail)]
fn test_div_by_abs_zero() {
    let _ = div_by_abs(SCALE, 0, 0);
}

#[test]
fn test_sqrt() {
    assert(sqrt(4 * SCALE) == 2 * SCALE);
    assert(sqrt(2 * SCALE) == 6074000999); // floor(sqrt(2) * 2^32)
}

#[test]
fn test_sin_cos() {
    // sin(pi/6) ~ 0.5, cos(pi/3) ~ 0.5, sin(-7pi/6) ~ 0.5, within 1e-6.
    let half = SCALE / 2;
    let e = 4295; // 1e-6 * 2^32
    let s = sin(2248839617); // round(pi/6 * 2^32)
    assert(lt(abs(s - half), e));
    let c = cos(4497679235); // round(pi/3 * 2^32)
    assert(lt(abs(c - half), e));
    let s2 = sin(-15741877322); // round(-7pi/6 * 2^32)
    assert(lt(abs(s2 - half), e));
}

// Malicious hints: every alternative a prover could load is rejected.

#[test]
fn test_divmod_honest() {
    check_divmod(-23, 5, -5, 2);
}

#[test(should_fail)]
fn test_divmod_next_quotient() {
    check_divmod(-23, 5, -4, -3);
}

#[test(should_fail)]
fn test_divmod_previous_quotient() {
    check_divmod(-23, 5, -6, 7);
}

#[test(should_fail)]
fn test_divmod_field_quotient() {
    check_divmod(-23, 5, (-23 - 3) / 5, 3);
}

#[test(should_fail)]
fn test_divmod_zero_divisor() {
    check_divmod(23, 0, 0, 23);
}

#[test(should_fail)]
fn test_divmod_negative_divisor() {
    check_divmod(23, -4, -5, 3);
}

#[test]
fn test_floor_div_pow2() {
    assert(floor_div_pow2::<2>(-7) == -2);
    assert(floor_div_pow2::<2>(7) == 1);
    check_div_pow2::<2>(-7, -2, 1);
}

#[test(should_fail)]
fn test_div_pow2_next_quotient() {
    check_div_pow2::<2>(-7, -1, -3);
}

#[test(should_fail)]
fn test_div_pow2_field_quotient() {
    check_div_pow2::<2>(-7, (-7 - 2) / 4, 2);
}

#[test]
fn test_floor_div_const() {
    assert(floor_div_const::<3>(-7) == -3);
    assert(floor_div_const::<3>(7) == 2);
    assert(floor_div_const::<9>(-9) == -1);
    assert(floor_div_const::<256>(-1) == -1);
    check_div_const::<5>(-23, -5, 2);
}

#[test(should_fail)]
fn test_div_const_next_quotient() {
    check_div_const::<5>(-23, -4, -3);
}

#[test(should_fail)]
fn test_div_const_previous_quotient() {
    check_div_const::<5>(-23, -6, 7);
}

#[test(should_fail)]
fn test_div_const_field_quotient() {
    check_div_const::<5>(-23, (-23 - 3) / 5, 3);
}

#[test(should_fail)]
fn test_div_const_remainder_equal_to_divisor() {
    check_div_const::<5>(-20, -5, 5);
}

#[test]
fn test_rescale_honest() {
    check_rescale(-5 * SCALE - 7, -6, SCALE - 7);
}

#[test(should_fail)]
fn test_rescale_next_quotient() {
    check_rescale(-5 * SCALE - 7, -5, -7);
}

#[test(should_fail)]
fn test_rescale_field_quotient() {
    check_rescale(-5 * SCALE - 7, (-5 * SCALE - 7 - (SCALE - 6)) / SCALE, SCALE - 6);
}

#[test(should_fail)]
fn test_sqrt_too_big() {
    check_sqrt(2 * SCALE, 6074000999 + 1);
}

#[test(should_fail)]
fn test_sqrt_too_small() {
    check_sqrt(2 * SCALE, 6074000999 - 1);
}

#[test(should_fail)]
fn test_sqrt_negative_root() {
    check_sqrt(2 * SCALE, -6074000999);
}

#[test(should_fail)]
fn test_parity_field_half() {
    let _ = check_parity(7, 7 / 2);
}

#[test]
fn test_max_of() {
    assert(max_of([-3 * SCALE, 5 * SCALE, 2 * SCALE]) == 5 * SCALE);
    check_max([-3, 5, 2], 5);
}

#[test(should_fail)]
fn test_max_smaller_member() {
    check_max([-3, 5, 2], 2);
}

#[test(should_fail)]
fn test_max_outside_window() {
    check_max([9223372036854775808, 3], 9223372036854775808);
}

#[test(should_fail)]
fn test_max_not_member() {
    check_max([-3, 5, 2], 6);
}

#[test]
fn test_min_of() {
    assert(min_of([-3 * SCALE, 5 * SCALE, 2 * SCALE]) == -3 * SCALE);
    check_min([-3, 5, 2], -3);
    // The window minimum, whose negation -max_of(-xs) cannot take.
    assert(min_of([-OFFSET, 5, -OFFSET + 1]) == -OFFSET);
}

#[test(should_fail)]
fn test_min_larger_member() {
    check_min([-3, 5, 2], 2);
}

#[test(should_fail)]
fn test_min_not_member() {
    check_min([-3, 5, 2], -4);
}

#[test(should_fail)]
fn test_min_outside_window() {
    check_min([-OFFSET - 1, 3], -OFFSET - 1);
}

#[test]
fn test_abs_int64() {
    assert(abs_int64(-5) == 5);
    assert(abs_int64(5) == 5);
    assert(abs_int64(0) == 0);
    assert(abs_int64(OFFSET - 1) == OFFSET - 1);
    assert(abs_int64(-OFFSET + 1) == OFFSET - 1);
}

#[test(should_fail)]
fn test_abs_int64_of_int64_min() {
    let _ = abs_int64(-OFFSET);
}

#[test(should_fail)]
fn test_abs_int64_above_window() {
    let _ = abs_int64(OFFSET);
}

#[test]
fn test_arctan2() {
    // round(atan2(y, x) * 2^32) from f64, compared within 1e-8 (43 units).
    let cases: [(Field, Field, Field); 12] = [
        (0, 0, 0),
        (0, SCALE, 0),
        (0, -SCALE, 13493037705),
        (SCALE, 0, 6746518852),
        (-SCALE, 0, -6746518852),
        (SCALE, SCALE, 3373259426),
        (-SCALE, -SCALE, -10119778278),
        (29 * SCALE, 70 * SCALE, 1686894214),
        (70 * SCALE, 29 * SCALE, 5059624638),
        (3 * SCALE, -4 * SCALE, 10729221487),
        (-OFFSET, -OFFSET, -10119778278),
        (OFFSET - 1, -OFFSET, 10119778278),
    ];
    for i in 0..12 {
        let (y, x, want) = cases[i];
        assert(lt(abs(arctan2(y, x) - want), 43));
    }
    assert(arctan2_nonneg(3 * SCALE, 4 * SCALE) == arctan2(3 * SCALE, 4 * SCALE));
}

// Wrong region bits and quotients are rejected.

#[test(should_fail)]
fn test_atan_wrong_octant() {
    let _ = check_atan_regions(3, 4, 1, 0);
}

#[test(should_fail)]
fn test_atan_wrong_octant_swapped() {
    let _ = check_atan_regions(4, 3, 0, 1);
}

#[test(should_fail)]
fn test_atan_wrong_reduction() {
    // 1 / 4 is below 29/70.
    let _ = check_atan_regions(1, 4, 0, 1);
}

#[test(should_fail)]
fn test_atan_wrong_reduction_over() {
    // 3 / 4 is above 29/70.
    let _ = check_atan_regions(3, 4, 0, 0);
}

#[test]
fn test_atan_regions_honest() {
    let _ = check_atan_regions(3, 4, 0, 1);
    let _ = check_atan_regions(4, 1, 1, 0);
    let _ = check_atan_regions(5, 5, 0, 1);
}

#[test(should_fail)]
fn test_div_half_ratio_next_quotient() {
    // -1 / 3: q = -1431655766, r = 2
    check_div_half_ratio(-1, 3, -1431655765, -1);
}

#[test(should_fail)]
fn test_div_half_ratio_field_quotient() {
    check_div_half_ratio(-1, 3, (-SCALE - 1) / 3, 1);
}

#[test(should_fail)]
fn test_div_half_ratio_half() {
    let _ = div_half_ratio(1, 2);
}

#[test]
fn test_div_half_ratio_honest() {
    check_div_half_ratio(-1, 3, -1431655766, 2);
}

#[test]
fn test_clip_const() {
    assert(clip_const(-5, 2, 10) == 2);
    assert(clip_const(2, 2, 10) == 2);
    assert(clip_const(7, 2, 10) == 7);
    assert(clip_const(10, 2, 10) == 10);
    assert(clip_const(11, 2, 10) == 10);
    assert(clip_const(-OFFSET, 2, 10) == 2);
    assert(clip_const(OFFSET - 1, 2, 10) == 10);
    assert(clip_const(-3 * SCALE, -SCALE / 4, SCALE / 2) == -SCALE / 4);
    assert(clip_const(5, 5, 5) == 5);
    assert(clip_const(OFFSET - 1, OFFSET - 6, OFFSET - 1) == OFFSET - 1);
    assert(check_clip(-OFFSET, -OFFSET, -OFFSET + SCALE - 1, 0, 0) == -OFFSET);
}

// Wrong regions: every alternative to the honest bits is rejected.

#[test(should_fail)]
fn test_clip_middle_claimed_below() {
    let _ = check_clip(1, 2, 10, 0, 0);
}

#[test(should_fail)]
fn test_clip_middle_claimed_above() {
    let _ = check_clip(11, 2, 10, 0, 0);
}

#[test(should_fail)]
fn test_clip_below_claimed_middle() {
    let _ = check_clip(2, 2, 10, 1, 0);
}

#[test(should_fail)]
fn test_clip_above_claimed_middle() {
    let _ = check_clip(10, 2, 10, 0, 1);
}

#[test(should_fail)]
fn test_clip_above_claimed_below() {
    let _ = check_clip(11, 2, 10, 1, 0);
}

#[test(should_fail)]
fn test_clip_below_claimed_above() {
    let _ = check_clip(1, 2, 10, 0, 1);
}

#[test(should_fail)]
fn test_clip_both_regions() {
    let _ = check_clip(1, 2, 10, 1, 1);
}

#[test(should_fail)]
fn test_clip_flag_not_a_bit() {
    let _ = check_clip(5, 2, 10, 2, 0);
}

#[test]
fn test_compact() {
    let keep = [0, 1, 1, 0, 1];
    let w = [0, 11, 12, 0, 14];
    assert(compact(keep, w) == [11, 12, 14, 0, 0]);
    let rows = compact_rows(keep, [[0, 0], [1, 2], [3, 4], [0, 0], [5, 6]]);
    assert(rows == [[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]]);
}

#[test]
fn test_array_index() {
    let a = [10, 11, 12, 13, 14, 15];
    assert(a[array_index::<8>(0, 5)] == 10);
    assert(a[array_index::<8>(5, 5)] == 15);
    assert(a[array_index::<28>(4, 5)] == 14);
}

#[test(should_fail)]
fn test_array_index_above_last() {
    let _ = array_index::<8>(6, 5);
}

#[test(should_fail)]
fn test_array_index_minus_one() {
    // p - 1, the index the pinned bb's memory check admits.
    let _ = array_index::<28>(-1, 5);
}

#[test(should_fail)]
fn test_array_index_half() {
    // (p + 1) / 2 passes the check of 2 idx (= 1), not the one of last - idx.
    let half: Field = 1 / 2;
    let _ = array_index::<8>(half, 5);
}

#[test]
fn test_index_pos() {
    assert(index_pos::<8>(-1, 5) == 4);
    assert(index_pos::<8>(-5, 5) == 0);
    assert(index_pos::<8>(2, 5) == 2);
    assert(index_pos::<28>(4, 5) == 4);
}

#[test]
fn test_pick() {
    let row = [10, 11, 12];
    assert(pick(row, 0) == 10);
    assert(pick(row, 2) == 12);
    assert(pick(row, -1) == 12);
    assert(pick(row, -3) == 10);
}

#[test(should_fail)]
fn test_pick_out_of_bounds() {
    let _ = pick([10, 11, 12], 3);
}

#[test(should_fail)]
fn test_pick_too_negative() {
    let _ = pick([10, 11, 12], -4);
}

#[test]
fn test_input_domain() {
    check_inputs([-9223372036854775808, 9223372036854775807, 0]);
}

#[test(should_fail)]
fn test_input_above_window() {
    check_input(9223372036854775808);
}

#[test(should_fail)]
fn test_input_below_window() {
    check_input(-9223372036854775809);
}

#[test(should_fail)]
fn test_index_out_of_bounds() {
    let _ = index_pos::<8>(5, 5);
}

#[test(should_fail)]
fn test_index_too_negative() {
    let _ = index_pos::<8>(-6, 5);
}


// Strict-comparison primitive APIs: share the frozen compiler's arithmetic
// schedules/coefficients rather than the historical tolerance-only minimax fits.
pub use noir_fpc::{qlt as canonical_lt, qgt as canonical_gt, qle as canonical_le, qge as canonical_ge,
    qdiv as canonical_div_or_zero, qexp as canonical_exp, qlog as canonical_log, qtanh as canonical_tanh,
    qsin as canonical_sin, qcos as canonical_cos, qarctan2 as canonical_arctan2};
// The magnitude check follows the frozen Noir backend's own domain.
// qdiv_nonzero_of's remainder constraints reject a zero divisor.
pub fn canonical_div_nonzero(a: Field, b: Field) -> Field {
    let (_, magnitude) = noir_fpc::sign_abs(b);
    noir_fpc::qdiv_nonzero_of(a, b, magnitude)
}
pub fn canonical_exp_neg(gap: Field) -> Field { canonical_exp(-gap) }
pub fn canonical_arctan2_nonneg(y: Field, x: Field) -> Field { canonical_arctan2(y, x) }

#[test]
fn canonical_frozen_vectors() {
    assert(canonical_exp(-(171798691840 as Field)) == 0);
    assert(canonical_exp(-(4294967296 as Field)) == 1580030168);
    assert(canonical_exp(-(1 as Field)) == 4294967291);
    assert(canonical_exp(0) == 4294967296);
    assert(canonical_exp(1) == 4294967296);
    assert(canonical_exp(429496729) == 4746672948);
    assert(canonical_exp(4294967296) == 11674931554);
    assert(canonical_exp(42949672960) == 94602950262784);
    assert(canonical_exp(90194313216) == 5664270453422686208);
    assert(canonical_log(1) == -(95265423098 as Field));
    assert(canonical_log(2) == -(92288378626 as Field));
    assert(canonical_log(2147483648) == -(2977044472 as Field));
    assert(canonical_log(4294967296) == 0);
    assert(canonical_log(4294967297) == 0);
    assert(canonical_log(8589934591) == 2977044470);
    assert(canonical_log(8589934592) == 2977044472);
    assert(canonical_log(528280977408) == 20668174427);
    assert(canonical_sin(-(85899345920 as Field)) == -(3921069985 as Field));
    assert(canonical_sin(-(13493037705 as Field)) == 0);
    assert(canonical_sin(-(6746518852 as Field)) == -(4294967285 as Field));
    assert(canonical_sin(-(4294967296 as Field)) == -(3614090346 as Field));
    assert(canonical_sin(0) == 0);
    assert(canonical_sin(4294967296) == 3614090346);
    assert(canonical_sin(6746518852) == 4294967285);
    assert(canonical_sin(13493037705) == 0);
    assert(canonical_sin(85899345920) == 3921069985);
    assert(canonical_tanh(-(42949672960 as Field)) == -(4294967279 as Field));
    assert(canonical_tanh(-(4294967296 as Field)) == -(3271021994 as Field));
    assert(canonical_tanh(-(1 as Field)) == -(4 as Field));
    assert(canonical_tanh(0) == 0);
    assert(canonical_tanh(1) == 2);
    assert(canonical_tanh(4294967296) == 3271021993);
    assert(canonical_tanh(42949672960) == 4294967278);
    assert(canonical_arctan2(0, 0) == 0);
    assert(canonical_arctan2(4294967296, 0) == 6746518852);
    assert(canonical_arctan2(0, -(4294967296 as Field)) == 13493037705);
    assert(canonical_arctan2(-(4294967296 as Field), 0) == -(6746518852 as Field));
    assert(canonical_arctan2(4294967296, 4294967296) == 3373259418);
    assert(canonical_arctan2(4294967296, 12884901888) == 1381908105);
    assert(canonical_arctan2(-(4294967296 as Field), 12884901888) == -(1381908105 as Field));
    assert(canonical_arctan2(-(4294967296 as Field), -(12884901888 as Field)) == -(12111129600 as Field));
    assert(canonical_arctan2(4294967296, -(12884901888 as Field)) == 12111129600);
}

#[test]
fn canonical_cosine_boundary_vectors() {
    assert(canonical_cos(0) == 4294967285);
    assert(canonical_cos(-(4294967296 as Field)) == 2320580745);
    assert(canonical_cos(4294967296) == 2320580746);
    assert(canonical_cos(9223372036854775807) == 701407306);
    assert(canonical_cos(-(9223372036854775808 as Field)) == 701407308);
}
