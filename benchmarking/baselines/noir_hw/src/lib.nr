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

pub fn abs(x: Field) -> Field {
    if is_neg(x) { -x } else { x }
}

pub fn max(a: Field, b: Field) -> Field {
    if window(a) < window(b) { b } else { a }
}

pub fn min(a: Field, b: Field) -> Field {
    if window(b) < window(a) { b } else { a }
}

unconstrained fn max_hint<let N: u32>(xs: [Field; N]) -> Field {
    let mut m = xs[0];
    for i in 1..N {
        if window(xs[i]) > window(m) {
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
    let m = unsafe { max_hint(xs) };
    check_max(xs, m);
    m
}

fn check_max<let N: u32>(xs: [Field; N], m: Field) {
    let _ = window(m);
    let mut prod: Field = 1;
    for i in 0..N {
        let gap = m - xs[i];
        gap.assert_max_bit_size::<64>();
        prod *= gap;
    }
    assert(prod == 0, "max: not one of the values");
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

// Position of a NumPy index in [-len, len) (negative counts from the end);
// any other index has no witness, as NumPy raises IndexError. Shifting by len
// maps the valid indices onto [0, 2 len): one 32-bit range check of the
// shifted Field (so the u32 cast is exact) and one u32 comparison prove it.
pub fn index_pos(idx: Field, len: u32) -> u32 {
    let shifted = idx + (len as Field);
    shifted.assert_max_bit_size::<32>();
    let s = shifted as u32;
    assert(s < 2 * len, "index out of bounds");
    if s >= len { s - len } else { s }
}

// row[idx] with NumPy semantics by a one-hot over the 2K valid index values
// -K..K: position p is hit when idx is p or p - K, i.e. when
// (idx - p) (idx - p + K) is zero (no zero divisors in a field), one zero test
// per position; the assertion that exactly one position is hit pins idx to a
// valid index (no range check). Cheaper than `index_pos` for short rows.
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
fn test_compact() {
    let keep = [0, 1, 1, 0, 1];
    let w = [0, 11, 12, 0, 14];
    assert(compact(keep, w) == [11, 12, 14, 0, 0]);
    let rows = compact_rows(keep, [[0, 0], [1, 2], [3, 4], [0, 0], [5, 6]]);
    assert(rows == [[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]]);
}

#[test]
fn test_index_pos() {
    assert(index_pos(-1, 5) == 4);
    assert(index_pos(2, 5) == 2);
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

#[test(should_fail)]
fn test_index_out_of_bounds() {
    let _ = index_pos(5, 5);
}

#[test(should_fail)]
fn test_index_too_negative() {
    let _ = index_pos(-6, 5);
}

