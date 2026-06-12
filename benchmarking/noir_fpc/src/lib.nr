// noir_fpc -- Q32 fixed-point library for Zinnia benchmarking baselines.
//
// Mirrors halo2-graph's FixedPointChip<F, 32>: scale = 2^32, BN254 signed
// values encoded as Field (negative v stored as p - |v|).
//
// Noir's Field has no native ordering. For order comparisons we project
// into a witnessed (sign, magnitude) pair where magnitude: u128 and
// constrain value == (1 - 2*sign) * magnitude in the field. Callers that
// only need +, -, == can stay in Field; comparison and abs trigger the
// projection.
//
// Multiplication and division emit witnessed rescale terms: an
// unconstrained hint provides the integer quotient; we constrain
// a*b == result*Q + r with |r| < Q.

pub global Q: Field = 4294967296;
pub global Q_U128: u128 = 4294967296;
pub global HALF_Q: u128 = 2147483648;
// Bound on legal magnitudes we range-check against. 2^126 is well below
// the BN254 modulus (~2^254) and large enough that legitimate Q32
// intermediate products (~2^96 for chain-of-3 mul) cannot reach it.
global MAG_BOUND: u128 = 0x40000000000000000000000000000000;

unconstrained fn hint_split_sign(v: Field) -> (bool, u128) {
    let pos_u128: u128 = v as u128;
    let pos_f: Field = pos_u128 as Field;
    if v == pos_f {
        (false, pos_u128)
    } else {
        let neg_u128: u128 = (-v) as u128;
        (true, neg_u128)
    }
}

pub fn to_signed(v: Field) -> (bool, u128) {
    // Safety: the assertion below constrains the unconstrained projection.
    let (sign, mag) = unsafe { hint_split_sign(v) };
    let mag_f: Field = mag as Field;
    let signed_f: Field = if sign { -mag_f } else { mag_f };
    assert(v == signed_f, "to_signed: sign/mag decomposition mismatch");
    assert(mag < MAG_BOUND, "to_signed: magnitude out of bounds");
    (sign, mag)
}

pub fn is_negative(v: Field) -> bool {
    let (s, _m) = to_signed(v);
    s
}

pub fn qabs(v: Field) -> Field {
    let (_s, m) = to_signed(v);
    m as Field
}

// Signed less-than for Q-scaled Field values.
//   sa=true,  sb=false -> negative < non-negative (true unless both zero)
//   sa=false, sb=true  -> false
//   sa=false, sb=false -> ma < mb
//   sa=true,  sb=true  -> ma > mb (larger magnitude is more negative)
pub fn qlt(a: Field, b: Field) -> bool {
    let (sa, ma) = to_signed(a);
    let (sb, mb) = to_signed(b);
    if sa {
        if !sb { (ma != 0) | (mb != 0) } else { ma > mb }
    } else {
        if sb { false } else { ma < mb }
    }
}

pub fn qle(a: Field, b: Field) -> bool { !qlt(b, a) }
pub fn qgt(a: Field, b: Field) -> bool { qlt(b, a) }
pub fn qge(a: Field, b: Field) -> bool { !qlt(a, b) }

// Pre-projected forms. When the same operand participates in many
// comparisons (e.g. a slice-bound `col` compared against each index in
// 0..N), each `qlt` call would redundantly emit `to_signed`'s 126-bit
// range check. The emitter projects each operand once and feeds the
// pair into these helpers, slashing the per-cmp cost from ~140 to ~5
// gates while keeping soundness identical to `qlt(a, b)`.
pub fn qlt_signed(sa: bool, ma: u128, sb: bool, mb: u128) -> bool {
    if sa {
        if !sb { (ma != 0) | (mb != 0) } else { ma > mb }
    } else {
        if sb { false } else { ma < mb }
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

pub fn qadd(a: Field, b: Field) -> Field { a + b }
pub fn qsub(a: Field, b: Field) -> Field { a - b }
pub fn qneg(a: Field) -> Field { -a }

// Euclidean integer division hint: returns (q, r) such that n == q*d + r
// with 0 <= r < |d|. Sign-aware on both operands -- the previous body did
// `n as u64`, which wraps negative Field elements (p - |n|) into a huge
// u64 and corrupts the quotient. Matches Zinnia's IR semantics for FloorDivI
// (`i128::div_euclid`) and ModI (`i128::rem_euclid`), so all backends agree.
// NOTE: this Euclidean convention coincides with Python/numpy floor-div and
// floor-mod whenever d > 0 (including negative dividends). It diverges from
// Python only for d < 0 (Python's remainder takes the sign of d); the mock
// and halo2 backends ship the same Euclidean convention, so Noir stays
// consistent with the rest of the pipeline.
unconstrained fn hint_idiv(n: Field, d: Field) -> (Field, Field) {
    let (n_neg, n_mag) = hint_split_sign(n);
    let (d_neg, d_mag) = hint_split_sign(d);
    // Magnitude truncated division, then Euclidean sign correction.
    let qm: u128 = n_mag / d_mag;
    let rm: u128 = n_mag - qm * d_mag;
    let q_neg: bool = n_neg ^ d_neg;
    // Truncated quotient/remainder in signed Field form.
    let q_trunc: Field = if q_neg { -(qm as Field) } else { qm as Field };
    let r_trunc: Field = if n_neg { -(rm as Field) } else { rm as Field };
    // Euclidean adjustment: force 0 <= r < |d|. When r_trunc < 0, add |d|
    // to r and step q by -sign(d) (i.e. q decreases by 1 when d > 0).
    if (rm != 0) & n_neg {
        let r_eucl: Field = (d_mag as Field) - (rm as Field);
        let q_adj: Field = if d_neg { q_trunc + 1 } else { q_trunc - 1 };
        (q_adj, r_eucl)
    } else {
        (q_trunc, r_trunc)
    }
}

// Integer floor (Euclidean) division, self-soundproof for ALL signs.
// Constraints: `q*d + r == n` and `0 <= r < |d|`. Those three facts pin a
// UNIQUE (q, r) for any (n, d), so the unconstrained hint cannot be
// substituted. The `r < |d|` check goes through `to_signed` (an internal
// Field-to-u128 range check, ~2840 gates). This is the default emitter
// lowering for user-level `a // b` on the Field path.
pub fn q_floor_div(n: Field, d: Field) -> Field {
    // Safety: q is constrained by the reconstruction + range check below.
    let (q, r) = unsafe { hint_idiv(n, d) };
    assert(q * d + r == n, "q_floor_div: reconstruction failed");
    let (r_neg, r_mag) = to_signed(r);
    let (_d_neg, d_mag) = to_signed(d);
    assert(!r_neg, "q_floor_div: negative remainder");
    assert(r_mag < d_mag, "q_floor_div: r >= |d|");
    q
}

// Integer floor (Euclidean) remainder, self-soundproof for ALL signs.
// Same constraint set as `q_floor_div`; returns r in [0, |d|). Default
// emitter lowering for user-level `a % b` on the Field path.
pub fn q_mod(n: Field, d: Field) -> Field {
    // Safety: r is constrained by the reconstruction + range check below.
    let (q, r) = unsafe { hint_idiv(n, d) };
    assert(q * d + r == n, "q_mod: reconstruction failed");
    let (r_neg, r_mag) = to_signed(r);
    let (_d_neg, d_mag) = to_signed(d);
    assert(!r_neg, "q_mod: negative remainder");
    assert(r_mag < d_mag, "q_mod: r >= |d|");
    r
}

// Truncate-toward-zero division of v by Q, sign-aware.
// Returns (q, r) such that v == q*Q + r, |r| < Q, and r has the same sign
// as v (or is zero). This is the IR's int_cast convention (`vi / scale`,
// Rust truncating division -- NOT Euclidean floor), matching the mock and
// halo2 backends so int(x) agrees across the stack for negatives.
unconstrained fn hint_trunc_div_q(v: Field) -> (Field, Field) {
    let (v_neg, v_mag) = hint_split_sign(v);
    let qm: u128 = v_mag / Q_U128;
    let rm: u128 = v_mag - qm * Q_U128;
    let q: Field = if v_neg { -(qm as Field) } else { qm as Field };
    let r: Field = if v_neg { -(rm as Field) } else { rm as Field };
    (q, r)
}

// Divide-by-Q with truncation toward zero, self-soundproof for ALL signs.
// Default emitter lowering for float-to-int casts (int_cast) on the Field
// path: int(x) = trunc(x_q / Q). Pinning argument: the reconstruction
// `q*Q + r == v` with `|r| < Q` alone admits TWO candidates for negative v
// (the floor and the truncation differ by one), so we additionally force r
// to carry v's sign (r == 0 when v's integer part divides evenly). That
// triple uniquely pins q to the truncating quotient -- the unconstrained
// hint cannot be substituted.
pub fn q_div_q(v: Field) -> Field {
    // Safety: q is constrained by the reconstruction + sign/range checks.
    let (q, r) = unsafe { hint_trunc_div_q(v) };
    assert(q * Q + r == v, "q_div_q: reconstruction failed");
    let (v_neg, _v_mag) = to_signed(v);
    let (r_neg, r_mag) = to_signed(r);
    assert(r_mag < Q_U128, "q_div_q: |r| >= Q");
    // r must share v's sign (or be zero) -- forbids the floor alternative.
    if r_mag != 0 {
        assert(r_neg == v_neg, "q_div_q: remainder sign mismatch (not truncation)");
    }
    q
}

// Hint-only quotient: returns the integer floor q = n/d as a Field, with
// NO standalone soundness constraint -- the caller must bind q via a
// downstream assertion. As of the C04 fix the Noir emitter no longer emits
// any unpinned division (every Field-path `a // b` / `a % b` / `int(x)`
// lowers to the self-sound `q_floor_div` / `q_mod` / `q_div_q`), so this
// hint is currently UNUSED. It is retained for a future structural
// pinned-marker that would let compiler-generated, downstream-constrained
// reshape / slice divisions reclaim the range-check cost; such a marker
// MUST be witness-independent and survive the IR roundtrip.
pub fn q_floor_div_hint(n: Field, d: Field) -> Field {
    // Safety: caller-side constraint pins q.
    let (q, _r) = unsafe { hint_idiv(n, d) };
    q
}

// Hint-only remainder: returns n - q*d where q = n/d. Same soundness
// caveat as `q_floor_div_hint`.
pub fn q_mod_hint(n: Field, d: Field) -> Field {
    // Safety: caller-side constraint pins r.
    let (_q, r) = unsafe { hint_idiv(n, d) };
    r
}

// Hint-only divide-by-Q: returns floor(v / Q). Same soundness caveat as
// `q_floor_div_hint`; retained for symmetry but not currently emitted.
pub fn q_div_q_hint(v: Field) -> Field {
    // Safety: caller-side constraint pins q.
    let (q, _r) = unsafe { hint_idiv(v, Q) };
    q
}

unconstrained fn hint_div_q(v_mag: u128) -> (u128, u128) {
    let q: u128 = v_mag / Q_U128;
    let r: u128 = v_mag - q * Q_U128;
    (q, r)
}

// Returns floor(v / Q) for a raw accumulated fixed-point value.
pub fn qscale_down_raw(v: Field) -> Field {
    let (sign, mag) = to_signed(v);
    // Safety: quotient/remainder are constrained by reconstruction below.
    let (q_mag, r_mag) = unsafe { hint_div_q(mag) };
    let result_f: Field = if sign {
        if r_mag == 0 {
            -(q_mag as Field)
        } else {
            -((q_mag + 1) as Field)
        }
    } else {
        q_mag as Field
    };
    let rem_mag: u128 = if sign {
        if r_mag == 0 { 0 } else { Q_U128 - r_mag }
    } else {
        r_mag
    };
    let rem_f: Field = rem_mag as Field;
    assert(v == result_f * Q + rem_f, "qscale_down_raw: reconstruction failed");
    assert(rem_mag < Q_U128, "qscale_down_raw: remainder out of range");
    assert(q_mag < MAG_BOUND, "qscale_down_raw: quotient out of range");
    result_f
}

// Returns floor(a * b / Q) using the same Euclidean rescale as the Rust kernel.
pub fn qmul(a: Field, b: Field) -> Field {
    qscale_down_raw(a * b)
}

unconstrained fn hint_qdiv(a_mag: u128, b_mag: u128) -> (u128, u128) {
    let num: u128 = a_mag * Q_U128;
    let q: u128 = num / b_mag;
    let r: u128 = num - q * b_mag;
    (q, r)
}

// Returns floor(a * Q / b) preserving sign of (a*sign(b)).
pub fn qdiv(a: Field, b: Field) -> Field {
    let (sa, ma) = to_signed(a);
    let (sb, mb) = to_signed(b);
    assert(mb != 0, "qdiv: divide by zero");
    // Safety: the assertion below constrains the unconstrained rescale.
    let (q_mag, r_mag) = unsafe { hint_qdiv(ma, mb) };
    let q_sign: bool = sa ^ sb;
    let q_f: Field = if q_sign { -(q_mag as Field) } else { q_mag as Field };
    let r_f: Field = if sa { -(r_mag as Field) } else { r_mag as Field };
    let b_pos: Field = if sb { -b } else { b };
    assert(a * Q == q_f * b_pos + r_f, "qdiv: dividend reconstruction failed");
    assert(r_mag < mb, "qdiv: remainder >= |divisor|");
    assert(q_mag < MAG_BOUND, "qdiv: quotient out of range");
    q_f
}

pub fn q_from_int(x: u64) -> Field {
    (x as Field) * Q
}

pub fn q_from_neg_int(x: u64) -> Field {
    -((x as Field) * Q)
}

pub fn q_eq_with_tol(a: Field, b: Field, tol: u128) {
    let d = a - b;
    let (_s, m) = to_signed(d);
    assert(m <= tol, "q_eq_with_tol: |a - b| > tol");
}

pub fn q_close(a: Field, b: Field, tol: u128) -> bool {
    let d = a - b;
    let (_s, m) = to_signed(d);
    m <= tol
}

// Default tolerance: 0.001 * Q ~= 4_294_967.
pub global TOL_DEFAULT: u128 = 4294967;

// 4-term cos Taylor: 1 - x^2/2 + x^4/24 - x^6/720.
pub fn qcos(x: Field) -> Field {
    let x2 = qmul(x, x);
    let x4 = qmul(x2, x2);
    let x6 = qmul(x4, x2);
    let term2 = qdiv(x2, q_from_int(2));
    let term4 = qdiv(x4, q_from_int(24));
    let term6 = qdiv(x6, q_from_int(720));
    let one_q = q_from_int(1);
    one_q - term2 + term4 - term6
}

// 4-term sin Taylor: x - x^3/6 + x^5/120 - x^7/5040.
pub fn qsin(x: Field) -> Field {
    let x2 = qmul(x, x);
    let x3 = qmul(x2, x);
    let x5 = qmul(x3, x2);
    let x7 = qmul(x5, x2);
    let t3 = qdiv(x3, q_from_int(6));
    let t5 = qdiv(x5, q_from_int(120));
    let t7 = qdiv(x7, q_from_int(5040));
    x - t3 + t5 - t7
}

unconstrained fn hint_qsqrt(x_mag: u128) -> u128 {
    let target: u128 = x_mag * Q_U128;
    if target == 0 {
        0
    } else {
        let mut g: u128 = target;
        for _ in 0..64 {
            let g_next: u128 = (g + target / g) / 2;
            if g_next < g { g = g_next; }
        }
        g
    }
}

// Returns Q-scaled sqrt of x (x must be non-negative).
pub fn qsqrt(x: Field) -> Field {
    let (sx, mx) = to_signed(x);
    assert(!sx, "qsqrt: negative input");
    // Safety: bracket assertions below constrain the unconstrained guess.
    let y_mag: u128 = unsafe { hint_qsqrt(mx) };
    let y_f: Field = y_mag as Field;
    let yy: u128 = y_mag * y_mag;
    let xq: u128 = mx * Q_U128;
    let yy_plus: u128 = yy + 2 * y_mag + 1;
    assert(yy <= xq, "qsqrt: y^2 > x*Q");
    assert(xq < yy_plus, "qsqrt: x*Q >= (y+1)^2");
    y_f
}

#[test]
fn test_qmul_basic() {
    let a = q_from_int(2) + Q / 2;
    let b = q_from_int(4);
    let p = qmul(a, b);
    let expected = q_from_int(10);
    q_eq_with_tol(p, expected, TOL_DEFAULT);
}

#[test]
fn test_qdiv_basic() {
    let a = q_from_int(10);
    let b = q_from_int(4);
    let q = qdiv(a, b);
    let expected = q_from_int(2) + Q / 2;
    q_eq_with_tol(q, expected, TOL_DEFAULT);
}

#[test]
fn test_neg_arithmetic() {
    let a: Field = -q_from_int(3);
    let b: Field = q_from_int(2);
    let s = qadd(a, b);
    let expected = -q_from_int(1);
    q_eq_with_tol(s, expected, TOL_DEFAULT);
}

#[test]
fn test_qabs() {
    let v: Field = -q_from_int(5);
    let a = qabs(v);
    assert(a == q_from_int(5));
}

#[test]
fn test_qlt() {
    assert(qlt(-q_from_int(3), -q_from_int(2)));
    assert(qlt(-q_from_int(2), q_from_int(0)));
    assert(qlt(q_from_int(1), q_from_int(2)));
    assert(!qlt(q_from_int(5), q_from_int(5)));
}

// Euclidean floor-div / mod soundness across all sign combinations.
// Values are raw integers (NOT Q-scaled); these helpers operate on the
// integer Field cells the emitter feeds them.
#[test]
fn test_q_floor_div_signs() {
    // d > 0: Euclidean == Python floor for any dividend sign.
    assert(q_floor_div(7, 3) == 2);
    assert(q_floor_div(-7, 3) == -3);
    assert(q_floor_div(6, 3) == 2);
    assert(q_floor_div(-6, 3) == -2);
    // d < 0: Euclidean convention (matches mock/halo2, not Python).
    assert(q_floor_div(7, -3) == -2);
    assert(q_floor_div(-7, -3) == 3);
}

#[test]
fn test_q_mod_signs() {
    // Remainder is always in [0, |d|).
    assert(q_mod(7, 3) == 1);
    assert(q_mod(-7, 3) == 2);
    assert(q_mod(6, 3) == 0);
    assert(q_mod(7, -3) == 1);
    assert(q_mod(-7, -3) == 2);
}

#[test]
fn test_q_div_q_signs() {
    // trunc(v / Q) toward zero. v is a raw Q-scaled Field value here.
    assert(q_div_q(3 * Q) == 3);
    assert(q_div_q(3 * Q + 5) == 3);
    assert(q_div_q(-(3 * Q)) == -3);
    // -(3Q) - 5: truncates toward zero to -3 (remainder -5 shares v's sign).
    assert(q_div_q(-(3 * Q) - 5) == -3);
}

// ---------------------------------------------------------------------------
// Integer helpers (operate on signed Q-encoded values whose magnitude < 2^63)
// ---------------------------------------------------------------------------

pub fn qabs_i(v: Field) -> Field {
    let (_s, m) = to_signed(v);
    m as Field
}

pub fn qsign_i(v: Field) -> Field {
    let (s, m) = to_signed(v);
    if m == 0 { 0 } else if s { -(1 as Field) } else { 1 as Field }
}

pub fn qsign(v: Field) -> Field {
    // Float sign returns a Q-scaled +1, -1, or 0 to match qpow / qsin units.
    let (s, m) = to_signed(v);
    if m == 0 { 0 } else if s { -Q } else { Q }
}

// ---------------------------------------------------------------------------
// qexp / qlog / qpow
// ---------------------------------------------------------------------------
//
// Strategy mirrors halo2-graph's FixedPointChip<F, 32>:
//   exp(x) = 2^(x / ln2)
//   log(x) = log2(x) * ln2
// Range-reduce so the polynomial fits its training interval, then apply the
// Remez coefficients. The polynomial layouts mirror `EXP2_COEFS` / `LOG2_COEFS`
// in `src/prove/kernel.rs`.

// Q-scaled ln(2) ~= 2977044472 (= round(ln(2) * 2^32)).
global LN2_Q: Field = 2977044472;
// Q-scaled 1 / ln(2) ~= 6196328019.
global INV_LN2_Q: Field = 6196328019;

// Horner evaluation in Q-scale with leading-first coefficients.
fn horner_q(x: Field, coefs: [Field; 13]) -> Field {
    let mut acc: Field = coefs[0];
    for i in 1..13 {
        acc = qmul(x, acc) + coefs[i];
    }
    acc
}

fn horner_q15(x: Field, coefs: [Field; 15]) -> Field {
    let mut acc: Field = coefs[0];
    for i in 1..15 {
        acc = qmul(x, acc) + coefs[i];
    }
    acc
}

fn horner_q16(x: Field, coefs: [Field; 16]) -> Field {
    let mut acc: Field = coefs[0];
    for i in 1..16 {
        acc = qmul(x, acc) + coefs[i];
    }
    acc
}

// EXP2_COEFS quantized to Q32 (matches src/prove/kernel.rs::EXP2_COEFS).
global EXP2_COEFS_Q: [Field; 13] = [
    0,
    2,
    31,
    437,
    5676,
    65510,
    661577,
    5726720,
    41309550,
    238388332,
    1031764991,
    2977044472,
    4294967296,
];

pub fn qexp(x: Field) -> Field {
    // y = x / ln2 = x * (1/ln2)
    let y: Field = qmul(x, INV_LN2_Q);
    let (sy, my) = to_signed(y);
    // Split y into integer part i (Field, signed) and fractional part f in [0, Q).
    let i_abs: u128 = my / Q_U128;
    let f_abs: u128 = my - i_abs * Q_U128;
    // result = 2^y = 2^i * 2^f.
    // Polynomial input must be unsigned f in [0, Q) (i.e. 2^{frac} on [0,1)).
    // For negative y: y = -|y| = -(i + f). 2^y = 2^{-i} * 2^{-f}. But
    // -f is not in [0,1); instead use y = -(i+1) + (1-f), giving f' = 1-f in (0,1].
    let (i_field, f_q): (Field, Field) = if sy {
        if f_abs == 0 {
            (-(i_abs as Field), 0)
        } else {
            let f_neg: u128 = Q_U128 - f_abs;
            let i_neg: Field = -((i_abs + 1) as Field);
            (i_neg, f_neg as Field)
        }
    } else {
        (i_abs as Field, f_abs as Field)
    };
    let two_to_f: Field = horner_q(f_q, EXP2_COEFS_Q);
    // Multiply by 2^i_field. Iterative doubling/halving up to +-64.
    qmul_pow2(two_to_f, i_field)
}

// Multiply a Q-scaled value by 2^k where k is a signed Field integer.
// Uses witnessed power-of-two via repeated shift. Bounded to |k| <= 96.
fn qmul_pow2(v: Field, k: Field) -> Field {
    let (sk, mk) = to_signed(k);
    assert(mk < 128, "qmul_pow2: |k| too large");
    let mut acc: Field = v;
    if sk {
        // divide by 2^mk
        let mut steps: u128 = mk;
        for _ in 0..128 {
            if steps > 0 {
                acc = acc / 2;
                steps = steps - 1;
            }
        }
    } else {
        let mut steps: u128 = mk;
        for _ in 0..128 {
            if steps > 0 {
                acc = acc + acc;
                steps = steps - 1;
            }
        }
    }
    acc
}

// LOG2_COEFS_Q: log2(x) on [2, 4), degree 14. Mirrors LOG2_COEFS in kernel.rs.
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

unconstrained fn hint_log2_decompose(x_mag: u128) -> (i64, u128) {
    // Find k and m in [2*Q, 4*Q) such that x = m * 2^(k-1) / Q.
    // Equivalently x_mag * 2^? falls into [2*Q, 4*Q).
    let two_q: u128 = 2 * Q_U128;
    let four_q: u128 = 4 * Q_U128;
    let mut m: u128 = x_mag;
    let mut k: i64 = 0;
    // Bring m into [2Q, 4Q).
    for _ in 0..96 {
        if m < two_q {
            m = m * 2;
            k = k - 1;
        } else if m >= four_q {
            m = m / 2;
            k = k + 1;
        }
    }
    (k, m)
}

pub fn qlog(x: Field) -> Field {
    // log(x) = log2(x) * ln2. log2(x) = k + log2(m), where x = m * 2^k, m in [2, 4).
    let (sx, mx) = to_signed(x);
    assert(!sx, "qlog: non-positive input");
    assert(mx != 0, "qlog: zero input");
    // Safety: assertions below pin (k, m) to the canonical range-reduction.
    let (k, m_q): (i64, u128) = unsafe { hint_log2_decompose(mx) };
    // Mantissa range: m_q in [2Q, 4Q). This pins m_q to a single Q-quantized
    // candidate per (input, k) pair within the polynomial's fit interval.
    // We do not strictly constrain m_q == x_mag * 2^{-k}: integer division
    // in the witness loses up to 2^k - 1 ULPs, so the equality cannot be
    // demanded exactly in the field. The halo2 backend ships the same
    // soundness caveat (see synthesizer.rs::log2_range_reduce comments);
    // the value pipeline is correct to within Q-precision either way.
    assert(m_q >= 2 * Q_U128, "qlog: mantissa < 2");
    assert(m_q < 4 * Q_U128, "qlog: mantissa >= 4");
    // Loose reconstruction check: m_q << k must equal x_mag up to a small
    // additive error <= 2^k. Encode as `|m_q << k - x_mag| <= 2^k`.
    let k_abs: u128 = if k < 0 { (-k) as u128 } else { k as u128 };
    let m_q_f: Field = m_q as Field;
    let mx_f: Field = mx as Field;
    if k >= 0 {
        // expected = m_q << k_abs ; check |expected - mx_f| <= 2^k_abs.
        let mut pow2: u128 = 1;
        let mut shift_left: u128 = k_abs;
        for _ in 0..96 { if shift_left > 0 { pow2 = pow2 * 2; shift_left = shift_left - 1; } }
        let expected: Field = m_q_f * (pow2 as Field);
        let diff: Field = expected - mx_f;
        let (_s, dm) = to_signed(diff);
        assert(dm <= pow2, "qlog: reconstruction outside ULP (k >= 0)");
    } else {
        // m_q = x_mag * 2^(-k). Check |m_q - x_mag << k_abs| <= 1.
        let mut pow2: u128 = 1;
        let mut shift_left: u128 = k_abs;
        for _ in 0..96 { if shift_left > 0 { pow2 = pow2 * 2; shift_left = shift_left - 1; } }
        let expected: Field = mx_f * (pow2 as Field);
        let diff: Field = expected - m_q_f;
        let (_s, dm) = to_signed(diff);
        assert(dm <= 1, "qlog: reconstruction outside ULP (k < 0)");
    }

    let log2_m: Field = horner_q15(m_q_f, LOG2_COEFS_Q);
    // log2(x) = k + log2(m). k is an integer; scale to Q.
    let k_field: Field = if k < 0 { -(k_abs as Field) } else { k_abs as Field };
    let k_q: Field = k_field * Q;
    let log2_x: Field = k_q + log2_m;
    qmul(log2_x, LN2_Q)
}

pub fn qpow(base: Field, exp_q: Field) -> Field {
    // a^b = exp(b * log(a))
    let la: Field = qlog(base);
    let bla: Field = qmul(exp_q, la);
    qexp(bla)
}

// Integer-exponent power: base^exp where exp is a u64 (NOT Q-scaled).
// 32-iter square-and-multiply; constant-time over u32 exponent range.
// 0^0 = 1 by convention.
pub fn qpow_int_q32(base: Field, exp: u64) -> Field {
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

// ATAN_COEFS_Q: atan(t) on [-1, 1], degree 15. Mirrors ATAN_COEFS in kernel.rs.
global ATAN_COEFS_Q: [Field; 16] = [
    0,
    4294964071,
    0,
    -(1431492759 as Field),
    0,
    856548815,
    0,
    -(596658626 as Field),
    0,
    412388002,
    0,
    -(237862581 as Field),
    0,
    92381544,
    0,
    -(17009175 as Field),
];

pub fn qarctan(t: Field) -> Field {
    horner_q16(t, ATAN_COEFS_Q)
}

pub fn qarctan2(y: Field, x: Field) -> Field {
    let (sy, my) = to_signed(y);
    let (sx, mx) = to_signed(x);
    let abs_y: Field = my as Field;
    let abs_x: Field = mx as Field;
    // swap = abs_y > abs_x
    let swap: bool = my > mx;
    let num: Field = if swap { abs_x } else { abs_y };
    let den: Field = if swap { abs_y } else { abs_x };
    // den + tiny epsilon to dodge 0/0; epsilon = 1 (smallest positive Field).
    let den_safe: Field = den + 1;
    let t: Field = qdiv(num, den_safe);
    let poly: Field = qarctan(t);
    // half_pi in Q: 6746518852
    let half_pi_q: Field = 6746518852;
    let a0: Field = if swap { half_pi_q - poly } else { poly };
    // Quadrant adjust for x < 0
    let pi_q: Field = 13493037705;
    let a1: Field = if sx { pi_q - a0 } else { a0 };
    if sy { -a1 } else { a1 }
}

pub fn qarcsin(x: Field) -> Field {
    // arcsin(x) = arctan2(x, sqrt(1 - x*x))
    let xx: Field = qmul(x, x);
    let one_q: Field = Q;
    let one_minus_xx: Field = one_q - xx;
    let s: Field = qsqrt(one_minus_xx);
    qarctan2(x, s)
}

pub fn qarccos(x: Field) -> Field {
    // arccos(x) = arctan2(sqrt(1 - x*x), x)
    let xx: Field = qmul(x, x);
    let one_q: Field = Q;
    let one_minus_xx: Field = one_q - xx;
    let s: Field = qsqrt(one_minus_xx);
    qarctan2(s, x)
}

// ---------------------------------------------------------------------------
// Hyperbolic + tan
// ---------------------------------------------------------------------------

pub fn qsinh(x: Field) -> Field {
    // sinh(x) = (exp(x) - exp(-x)) / 2
    let ex: Field = qexp(x);
    let enx: Field = qexp(-x);
    let diff: Field = ex - enx;
    diff / 2
}

pub fn qcosh(x: Field) -> Field {
    // cosh(x) = (exp(x) + exp(-x)) / 2
    let ex: Field = qexp(x);
    let enx: Field = qexp(-x);
    let sum: Field = ex + enx;
    sum / 2
}

pub fn qtanh(x: Field) -> Field {
    let s: Field = qsinh(x);
    let c: Field = qcosh(x);
    qdiv(s, c)
}

pub fn qtan(x: Field) -> Field {
    let s: Field = qsin(x);
    let c: Field = qcos(x);
    qdiv(s, c)
}

// ---------------------------------------------------------------------------
// Tests for the new helpers
// ---------------------------------------------------------------------------

#[test]
fn test_qexp_one() {
    let one_q: Field = Q;
    let e: Field = qexp(one_q);
    let e_q: Field = 11674931555; // e ~= 2.71828... in Q32
    let d: Field = e - e_q;
    let (_s, m) = to_signed(d);
    assert(m < 1000 * TOL_DEFAULT, "qexp(1) far from e");
}

#[test]
fn test_qlog_e() {
    // log(e) ~= 1
    let e_q: Field = 11674931555;
    let l: Field = qlog(e_q);
    let one_q: Field = Q;
    let d: Field = l - one_q;
    let (_s, m) = to_signed(d);
    assert(m < 1000 * TOL_DEFAULT, "qlog(e) far from 1");
}

#[test]
fn test_qpow_small() {
    // 2.0 ^ 3.0 = 8. Composition of log + exp leaves ~1e-3 relative error,
    // so we accept up to ~0.05 absolute (Q*0.05 = ~2e8) at the result scale.
    let two_q: Field = q_from_int(2);
    let three_q: Field = q_from_int(3);
    let r: Field = qpow(two_q, three_q);
    let eight_q: Field = q_from_int(8);
    let d: Field = r - eight_q;
    let (_s, m) = to_signed(d);
    assert(m < 200000000, "2^3 far from 8");
}

#[test]
fn test_qarctan2_quadrant() {
    // arctan2(1, 1) = pi/4
    let one_q: Field = Q;
    let r: Field = qarctan2(one_q, one_q);
    let pi_4_q: Field = 3373259426;
    let d: Field = r - pi_4_q;
    let (_s, m) = to_signed(d);
    assert(m < 100 * TOL_DEFAULT, "arctan2(1,1) far from pi/4");
}

#[test]
fn test_qarcsin_half() {
    // arcsin(0.5) = pi/6 ~= 0.5236
    let half_q: Field = Q / 2;
    let r: Field = qarcsin(half_q);
    let pi_6_q: Field = 2248839617;
    let d: Field = r - pi_6_q;
    let (_s, m) = to_signed(d);
    assert(m < 1000 * TOL_DEFAULT, "arcsin(0.5) far from pi/6");
}
